import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from network import Network
from losses import WalkerVisitLosses
from input_pipeline import get_datasets


is_training = False
batch_size = 100
source_dataset, target_dataset = get_datasets(is_training)

source_loader = DataLoader(source_dataset, batch_size, shuffle=True, pin_memory=True, drop_last=True)
target_loader = DataLoader(target_dataset, batch_size, shuffle=True, pin_memory=True, drop_last=True)

DEVICE = torch.device('cuda:0')
model = Network(image_size=32, embedding_dim=64).to(DEVICE)
classifier = nn.Linear(64, 10).to(DEVICE)
cross_entropy = nn.CrossEntropyLoss()
association = WalkerVisitLosses()
params = [p for p in model.parameters()] + [p for p in classifier.parameters()]
optimizer = optim.Adam(lr=5e-4, params=params)

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

def train_and_evaluate():

    model.train()

    text = 'e:{0:2d}, i:{1:3d}, classification loss: {2:.3f}, ' +\
        'walker loss: {3:.3f}, visit loss: {4:.3f}, total loss: {5:.3f}'
    logs = []
    num_epochs = 10
    for e in range(num_epochs):
        i = 0
        for (x_source, y_source), (x_target, _) in zip(source_loader, target_loader):

            x_source = x_source.to(DEVICE)
            x_target = x_target.to(DEVICE)
            y_source = y_source.to(DEVICE)

            x = torch.cat([x_source, x_target], dim=0)
            embeddings = model(x)
            a, b = torch.split(embeddings, batch_size, dim=0)
            logits = classifier(a)

            usual_loss = cross_entropy(logits, y_source)
            loss = usual_loss

            if False:
                walker_loss, visit_loss = association(a, b, y_source)
                loss += walker_loss
                loss += visit_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log = (e, i, usual_loss, walker_loss, visit_loss, loss)
            print(text.format(*log))
            logs.append(log)
            i += 1

        r = evaluate(nn.Sequential(model, classifier), cross_entropy, source_loader)
        print(r)
        r = evaluate(nn.Sequential(model, classifier), cross_entropy, target_loader)
        print(r)

    torch.save(model.state_dict(), PATH)