import torch
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN
import numpy as np
from PIL import Image


def get_datasets(is_training):
    """
    Arguments:
        is_training: a boolean.
    Returns:
        two datasets with RGB images of size 32x32,
        pixel values are in range [0, 1].
        Possible labels are {0, 1, 2, ..., 9}.
    """
    svhn = SVHN(
        'datasets/svhn/', split='train' if is_training else 'test',
        download=True, transform=transforms.ToTensor()
    )
    mnist_transform = transforms.Compose([
        # randomly color digit and background:
        transforms.Lambda(to_random_rgb),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    mnist = MNIST(
        'datasets/mnist/', train=is_training,
        download=True, transform=mnist_transform
    )
    return svhn, mnist


def to_random_rgb(x):
    color1 = np.random.randint(0, 256, size=3, dtype='uint8')
    color2 = np.random.randint(0, 256, size=3, dtype='uint8')
    x = np.array(x)
    x = x.astype('float32')/255.0
    x = np.expand_dims(x, 2)
    x = (1.0 - x) * color1 + x * color2
    return Image.fromarray(x.astype('uint8'))
