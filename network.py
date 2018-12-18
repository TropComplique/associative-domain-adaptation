import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, image_size, embedding_dim):
        super(Network, self).__init__()

        feature_extractor = [
            nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        self.feature_extractor = nn.Sequential(*feature_extractor)
    
        assert image_size % 8 == 0
        final_area = (image_size // 8) ** 2  # final spatial image size
        self.embedding = nn.Linear(128 * final_area, embedding_dim)
        self.final_area = final_area
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents RGB images with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [b, embedding_dim].
        """
        b = x.size(0)
        x = 2.0*x - 1.0
        x = self.feature_extractor(x)
        x = x.view(b, 128 * self.final_area)
        x = self.embedding(x).view(b, self.embedding_dim)
        return x
