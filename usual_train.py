import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from network import Network
from input_pipeline import get_datasets
from train import evaluate


BATCH_SIZE = 1000
NUM_EPOCHS = 100
EMBEDDING_DIM = 64
DEVICE = torch.device('cuda:0')
SAVE_PATH = 'models/run00.pth'


def train_and_evaluate():

    svhn, _ = get_datasets(is_training=True)
    train_loader = DataLoader(svhn, BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    val_svhn, _ = get_datasets(is_training=False)
    val_loader = DataLoader(val_svhn, BATCH_SIZE, shuffle=False, drop_last=False)

    embedder = Network(image_size=(32, 32), embedding_dim=EMBEDDING_DIM).to(DEVICE)
    classifier = nn.Linear(EMBEDDING_DIM, 10).to(DEVICE)
    model = nn.Sequential(embedder, classifier)

    optimizer = optim.Adam(lr=1e-3, params=model.parameters())
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps_per_epoch * NUM_EPOCHS, eta_min=1e-6)
    cross_entropy = nn.CrossEntropyLoss()

    text = 'e:{0:2d}, i:{1:3d}, classification loss: {2:.3f}, lr: {3:.6f}'
    logs = []
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
            lr = scheduler.get_lr()

            log = (e, i, lossб lr)
            print(text.format(*log))
            logs.append(log)
            i += 1

        l, a = evaluate(model, cross_entropy, val_svhn)
        print('validation loss {0:.3f} and accuracy {1:.3f}'.format(l, a))

    torch.save(model.state_dict(), SAVE_PATH)


train_and_evaluate()
