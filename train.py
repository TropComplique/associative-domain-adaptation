import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import math

from network import Network
from losses import WalkerVisitLosses
from input_pipeline import get_datasets
from utils import evaluate, write_logs, make_weights_for_balanced_classes


"""
The purpose of this script is to train a simple
CNN on mnist and svhn using associative domain adaptation.
"""


BATCH_SIZE = 200
NUM_EPOCHS = 25
EMBEDDING_DIM = 64

DELAY = 1000  # number of steps before turning on additional losses
GROWTH_STEPS = 1000  # number of steps of linear growth of additional losses
# so domain adaptation losses are in full strength after `DELAY + GROWTH_STEPS` steps

BETA1, BETA2 = 1.0, 0.5
DEVICE = torch.device('cuda:0')
SOURCE_DATA = 'svhn'  # 'svhn' or 'mnist'
SAVE_PATH = 'models/svhn_source'
LOGS_PATH = 'logs/svhn_source.json'


def train_and_evaluate():

    svhn, mnist = get_datasets(is_training=True)
    source_dataset = svhn if SOURCE_DATA == 'svhn' else mnist
    target_dataset = mnist if SOURCE_DATA == 'svhn' else svhn

    weights = make_weights_for_balanced_classes(source_dataset, num_classes=10)
    sampler = WeightedRandomSampler(weights, len(weights))
    source_loader = DataLoader(source_dataset, BATCH_SIZE, sampler=sampler, pin_memory=True, drop_last=True)
    target_loader = DataLoader(target_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    val_svhn, val_mnist = get_datasets(is_training=False)
    val_svhn_loader = DataLoader(val_svhn, BATCH_SIZE, shuffle=False, drop_last=False)
    val_mnist_loader = DataLoader(val_mnist, BATCH_SIZE, shuffle=False, drop_last=False)
    print('\nsource dataset is', SOURCE_DATA, '\n')

    num_steps_per_epoch = math.floor(min(len(svhn), len(mnist)) / BATCH_SIZE)
    embedder = Network(image_size=(32, 32), embedding_dim=EMBEDDING_DIM).to(DEVICE)
    classifier = nn.Linear(EMBEDDING_DIM, 10).to(DEVICE)
    model = nn.Sequential(embedder, classifier)
    model.train()

    optimizer = optim.Adam(lr=1e-3, params=model.parameters(), weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * NUM_EPOCHS - DELAY, eta_min=1e-6)

    cross_entropy = nn.CrossEntropyLoss()
    association = WalkerVisitLosses()

    text = 'e:{0:2d}, i:{1:3d}, classification loss: {2:.3f}, ' +\
        'walker loss: {3:.3f}, visit loss: {4:.4f}, ' +\
        'total loss: {5:.3f}, lr: {6:.6f}'
    logs, val_logs = [], []
    i = 0  # iteration

    for e in range(NUM_EPOCHS):
        model.train()
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
                growth = torch.clamp(torch.tensor((i - DELAY)/GROWTH_STEPS).to(DEVICE), 0.0, 1.0)
                loss = usual_loss + growth * (BETA1 * walker_loss + BETA2 * visit_loss)
            else:
                loss = usual_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i > DELAY:
                scheduler.step()
            lr = scheduler.get_lr()[0]

            log = (e, i, usual_loss.item(), walker_loss.item(), visit_loss.item(), loss.item(), lr)
            print(text.format(*log))
            logs.append(log)
            i += 1

        result1 = evaluate(model, cross_entropy, val_svhn_loader, DEVICE)
        result2 = evaluate(model, cross_entropy, val_mnist_loader, DEVICE)
        print('\nsvhn loss {0:.3f} and accuracy {1:.3f}'.format(*result1))
        print('mnist loss {0:.3f} and accuracy {1:.3f}\n'.format(*result2))
        val_logs.append((i,) + result1 + result2)

    torch.save(model.state_dict(), SAVE_PATH)
    write_logs(logs, val_logs, LOGS_PATH)


train_and_evaluate()
