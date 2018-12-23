import torch
import json


def evaluate(model, criterion, loader, device):
    model.eval()
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


def make_weights_for_balanced_classes(dataset, num_classes):

    count = [0] * num_classes
    for _, label in dataset:
        count[label] += 1

    weight_per_class = [0.0] * num_classes
    N = float(sum(count))

    for i in range(num_classes):
        weight_per_class[i] = N/float(count[i])

    weights = [0.0] * len(dataset)
    for i, (_, label) in enumerate(dataset):
        weights[i] = weight_per_class[label]

    return torch.DoubleTensor(weights)


def write_logs(logs, val_logs, path):
    keys = [
        'step', 'classification_loss',
        'walker_loss', 'visit_loss'
    ]
    val_keys = [
        'val_step', 'svhn_logloss', 'svhn_accuracy',
        'mnist_logloss', 'mnist_accuracy'
    ]
    d = {k: [] for k in keys + val_keys}

    for t in logs:
        for i, k in enumerate(keys, 1):
            d[k].append(t[i])

    for t in val_logs:
        for i, k in enumerate(val_keys):
            d[k].append(t[i])

    with open(path, 'w') as f:
        json.dump(d, f)
