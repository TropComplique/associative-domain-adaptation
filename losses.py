import torch
import torch.nn as nn
import torch.nn.functional as F


# a small value
EPSILON = 1e-8


class WalkerVisitLosses(nn.Module):

    def __init__(self):
        super(WalkerVisitLosses, self).__init__()

    def forward(self, a, b, labels_for_a):
        """
        Arguments:
            a: a float tensor with shape [n, d].
            b: a float tensor with shape [m, d].
            labels_for_a: a long tensor with shape [n],
                it has values in {0, 1, ..., num_labels - 1}.
        Returns:
            two float tensors with shape [].
        """
        d = a.size(1)
        p = torch.matmul(a, b.t())  # shape [n, m]
        p /= torch.tensor(d).float().sqrt()

        ab = F.softmax(p, dim=1)  # shape [n, m]
        ba = F.softmax(p.t(), dim=1)  # shape [m, n]
        aba = torch.matmul(ab, ba)  # shape [n, n]
        # note that all rows in `aba` sum to one

        labels = labels_for_a.unsqueeze(0)  # shape [1, n]
        is_same_label = (labels == labels.t()).float()  # shape [n, n]
        label_count = is_same_label.sum(1).unsqueeze(1)  # shape [n, 1]
        targets = is_same_label/label_count  # shape [n, n]
        # note that all rows in `targets` sum to one

        walker_loss = targets * torch.log(EPSILON + aba)  # shape [n, n]
        walker_loss = walker_loss.sum(1).mean(0).neg()

        visit_probability = ab.mean(0)  # shape [m]
        # note that visit_probability.sum() = 1

        m = b.size(0)
        targets = (1.0 / m) * torch.ones_like(visit_probability)
        visit_loss = targets * torch.log(EPSILON + visit_probability)  # shape [m]
        visit_loss = visit_loss.sum(0).neg()

        return walker_loss, visit_loss
