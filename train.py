import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from network import Network
from losses import WalkerVisitLosses
from input_pipeline import get_datasets


BATCH_SIZE = 1000
NUM_EPOCHS = 100
DELAY = 500
BETA1 = 1.0
BETA2 = 0.5
DEVICE = torch.device('cuda:0')
SAVE_PATH = 'models/run00.pth'


def train_and_evaluate(source_loader, target_loader, val_source_loader, val_target_loader):

    embedder = Network(image_size=(32, 32), embedding_dim=64).to(DEVICE)
    classifier = nn.Linear(64, 10).to(DEVICE)
    model = nn.Sequential(embedder, classifier)
    optimizer = optim.Adam(lr=5e-4, params=model.parameters())

    model.train()
    cross_entropy = nn.CrossEntropyLoss()
    association = WalkerVisitLosses()

    text = 'e:{0:2d}, i:{1:3d}, classification loss: {2:.3f}, ' +\
        'walker loss: {3:.3f}, visit loss: {4:.3f}, total loss: {5:.3f}'
    logs = []
    i = 0  # iteration

    for e in range(NUM_EPOCHS):
        for (x_source, y_source), (x_target, _) in zip(source_loader, target_loader):

            x_source = x_source.to(DEVICE)
            x_target = x_target.to(DEVICE)
            y_source = y_source.to(DEVICE)

            x = torch.cat([x_source, x_target], dim=0)
            embeddings = embedder(x)
            a, b = torch.split(embeddings, batch_size, dim=0)
            logits = classifier(a)

            usual_loss = cross_entropy(logits, y_source)
            loss = usual_loss

            walker_loss, visit_loss = association(a, b, y_source)
            if i > DELAY:
                loss += BETA1 * walker_loss
                loss += BETA2 * visit_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log = (e, i, usual_loss, walker_loss, visit_loss, loss)
            print(text.format(*log))
            logs.append(log)
            i += 1

        l, a = evaluate(model, cross_entropy, val_source_loader)
        print('source loss {0:.3f} and accuracy {1:.3f}'.format(l, a))
        l, a = evaluate(model, cross_entropy, val_target_loader)
        print('target loss {0:.3f} and accuracy {1:.3f}'.format(l, a))

    torch.save(model.state_dict(), SAVE_PATH)


def evaluate(model, criterion, loader):

    model.eval()
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


source_dataset, target_dataset = get_datasets(is_training=True)
source_loader = DataLoader(source_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
target_loader = DataLoader(target_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

val_source_dataset, val_target_dataset = get_datasets(is_training=False)
val_source_loader = DataLoader(val_source_dataset, BATCH_SIZE, shuffle=False, drop_last=False)
val_target_loader = DataLoader(val_target_dataset, BATCH_SIZE, shuffle=False, drop_last=False)

train_and_evaluate(source_loader, target_loader, val_source_loader, val_target_loader)
