import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import math

from network import Network
from losses import WalkerVisitLosses
from input_pipeline import get_datasets


BATCH_SIZE = 1000
NUM_EPOCHS = 100
EMBEDDING_DIM = 64
DELAY = 500
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

    optimizer = optim.Adam(lr=1e-3, params=model.parameters())
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * NUM_EPOCHS, eta_min=1e-6)

    cross_entropy = nn.CrossEntropyLoss()
    association = WalkerVisitLosses()

    text = 'e:{0:2d}, i:{1:3d}, classification loss: {2:.3f}, ' +\
        'walker loss: {3:.3f}, visit loss: {4:.3f}, ' +\
        'total loss: {5:.3f}, lr: {6:.6f}'
    logs = []
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
                loss = usual_loss + BETA1 * walker_loss + BETA2 * visit_loss
            else:
                loss = usual_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            lr = scheduler.get_lr()

            log = (e, i, usual_loss, walker_loss, visit_loss, loss, lr)
            print(text.format(*log))
            logs.append(log)
            i += 1

        l, a = evaluate(model, cross_entropy, val_source_loader)
        print('source loss {0:.3f} and accuracy {1:.3f}'.format(l, a))
        l, a = evaluate(model, cross_entropy, val_target_loader)
        print('target loss {0:.3f} and accuracy {1:.3f}'.format(l, a))

    torch.save(model.state_dict(), SAVE_PATH)


def evaluate(model, criterion, loader):

    total_loss = 0.0
    num_hits = 0
    num_samples = 0

    for images, targets in loader:

        batch_size = images.size(0)
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.set_grad_enabled(False):
            logits = model(images)
            loss = criterion(logits, targets)

        _, predicted_labels = logits.max(1)
        num_hits += (targets == predicted_labels).float().sum()
        total_loss += loss * batch_size
        num_samples += batch_size

    loss = total_loss.item() / num_samples
    accuracy = num_hits.item() / num_samples
    return loss, accuracy


train_and_evaluate()
