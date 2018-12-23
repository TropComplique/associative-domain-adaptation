import torch
import torch.nn.init
import torch.nn as nn
    

class Network(nn.Module):

    def __init__(self, image_size, embedding_dim):
        super(Network, self).__init__()

        feature_extractor = [
            nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        self.feature_extractor = nn.Sequential(*feature_extractor)

        width, height = image_size
        assert width % 8 == 0 and height % 8 == 0
        final_area = (width // 8) * (height // 8)
        self.embedding = nn.Sequential(
            nn.Linear(128 * final_area, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.final_area = final_area
        self.embedding_dim = embedding_dim

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(weights_init)

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
