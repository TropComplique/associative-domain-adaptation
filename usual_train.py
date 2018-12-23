import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import math

from network import Network
from input_pipeline import get_datasets
from utils import evaluate


"""
The purpose of this script is to train a simple
CNN on mnist and svhn using a usual training method.

This is needed for comparison with
models trained with domain adaptation.
"""


BATCH_SIZE = 32
NUM_EPOCHS = 15
EMBEDDING_DIM = 64
DEVICE = torch.device('cuda:0')
DATA = 'mnist'  # 'svhn' or 'mnist'
SAVE_PATH = 'models/just_mnist'


def get_loaders():

    svhn, mnist = get_datasets(is_training=True)
    val_svhn, val_mnist = get_datasets(is_training=False)

    train_dataset = svhn if DATA == 'svhn' else mnist
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    val_svhn_loader = DataLoader(val_svhn, BATCH_SIZE, shuffle=False, drop_last=False)
    val_mnist_loader = DataLoader(val_mnist, BATCH_SIZE, shuffle=False, drop_last=False)
    return train_loader, val_svhn_loader, val_mnist_loader


def train_and_evaluate():

    train_loader, val_svhn_loader, val_mnist_loader = get_loaders()
    num_steps_per_epoch = math.floor(len(train_loader.dataset) / BATCH_SIZE)
    print('\ntraining is on', DATA, '\n')

    embedder = Network(image_size=(32, 32), embedding_dim=EMBEDDING_DIM).to(DEVICE)
    classifier = nn.Linear(EMBEDDING_DIM, 10).to(DEVICE)
    model = nn.Sequential(embedder, classifier)

    optimizer = optim.Adam(lr=1e-3, params=model.parameters(), weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * NUM_EPOCHS, eta_min=1e-6)
    cross_entropy = nn.CrossEntropyLoss()
    i = 0  # iteration

    for e in range(NUM_EPOCHS):
        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            i += 1

        result1 = evaluate(model, cross_entropy, val_svhn_loader, DEVICE)
        result2 = evaluate(model, cross_entropy, val_mnist_loader, DEVICE)
        print('iteration', i)
        print('svhn validation loss {0:.3f} and accuracy {1:.3f}'.format(*result1))
        print('mnist validation loss {0:.3f} and accuracy {1:.3f}\n'.format(*result2))

    torch.save(model.state_dict(), SAVE_PATH)


train_and_evaluate()
