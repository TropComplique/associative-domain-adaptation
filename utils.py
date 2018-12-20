import torch
import json


def evaluate(model, criterion, loader, device):

    total_loss = 0.0
    num_hits = 0
    num_samples = 0

    for images, targets in loader:

        batch_size = images.size(0)
        images = images.to(device)
        targets = targets.to(device)

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


def write_logs(logs, val_logs, path):
    keys = [
        'step', 'classification_loss',
        'walker_loss', 'visit_loss'
    ]
    val_keys = [
        'val_step', 'svhn_logloss', 'svhn_accuracy',
        'mnist_logloss', 'mnist_accuracy'
    ]
    logs = {k: [] for k in keys + val_keys}

    for t in logs:
        for i, k in enumerate(keys, 1):
            logs[k].append(t[i])

    for t in val_logs:
        for i, k in enumerate(val_keys):
            logs[k].append(t[i])

    with open(path, 'w') as f:
        json.dump(logs, f)
