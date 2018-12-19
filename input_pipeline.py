import torch
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN


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
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # to RGB
    ])
    mnist = MNIST(
        'datasets/mnist/', train=is_training,
        download=True, transform=mnist_transform
    )
    return svhn, mnist
