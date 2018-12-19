import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import math

from network import Network
from losses import WalkerVisitLosses
from input_pipeline import get_datasets
from utils import evaluate, write_logs


BATCH_SIZE = 1000
NUM_EPOCHS = 1
EMBEDDING_DIM = 64
DELAY = 500
GROWTH_STEPS = 300
BETA1 = 1.0
BETA2 = 0.5
DEVICE = torch.device('cuda:0')
SAVE_PATH = 'models/run00.pth'


def train_and_evaluate():

    svhn, mnist = get_datasets(is_training=True)
    source_loader = DataLoader(svhn, BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    target_loader = DataLoader(mnist, BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    val_svhn, val_mnist = get_datasets(is_training=False)
    val_source_loader = DataLoader(val_svhn, BATCH_SIZE, shuffle=False, drop_last=False)
    val_target_loader = DataLoader(val_mnist, BATCH_SIZE, shuffle=False, drop_last=False)

    num_steps_per_epoch = math.floor(min(len(svhn), len(mnist)) / BATCH_SIZE)
    embedder = Network(image_size=(32, 32), embedding_dim=EMBEDDING_DIM).to(DEVICE)
    classifier = nn.Linear(EMBEDDING_DIM, 10).to(DEVICE)
    model = nn.Sequential(embedder, classifier)

    optimizer = optim.Adam(lr=1e-2, params=model.parameters())
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * NUM_EPOCHS, eta_min=1e-5)

    cross_entropy = nn.CrossEntropyLoss()
    association = WalkerVisitLosses()

    text = 'e:{0:2d}, i:{1:3d}, classification loss: {2:.3f}, ' +\
        'walker loss: {3:.3f}, visit loss: {4:.3f}, ' +\
        'total loss: {5:.3f}, lr: {6:.6f}'
    logs, val_logs = [], []
    i = 0  # iteration

    for e in range(NUM_EPOCHS):
        for (x_source, y_source), (x_target, _) in zip(source_loader, target_loader):

            x_source = x_source.to(DEVICE)
            x_target = x_target.to(DEVICE)
            y_source = y_source.to(DEVICE)

            x = torch.cat([x_source, x_target], dim=0)
            embeddings = embedder(x)
            a, b = torch.split(embeddings, BATCH_SIZE, dim=0)
            logits = classifier(a)
            usual_loss = cross_entropy(logits, y_source)
            walker_loss, visit_loss = association(a, b, y_source)

            if i > DELAY:
                growth = min((i - DELAY)/GROWTH_STEPS, 1.0)
                loss = usual_loss + growth * (BETA1 * walker_loss + BETA2 * visit_loss)
            else:
                loss = usual_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            lr = scheduler.get_lr()[0]

            log = (e, i, usual_loss.item(), walker_loss.item(), visit_loss.item(), loss.item(), lr)
            print(text.format(*log))
            logs.append(log)
            i += 1

        result1 = evaluate(model, cross_entropy, val_source_loader, DEVICE)
        result2 = evaluate(model, cross_entropy, val_target_loader, DEVICE)
        print('source loss {0:.3f} and accuracy {1:.3f}'.format(*result1))
        print('target loss {0:.3f} and accuracy {1:.3f}'.format(*result2))
        val_logs.append(result1 + result2)

    torch.save(model.state_dict(), SAVE_PATH)
    write_logs(logs, val_logs)


train_and_evaluate()
