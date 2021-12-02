import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize


def get_mnist_transformer():
    # The operations which are being applied to each image before training
    return torchvision.transforms.Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])


def get_mnist_inverse_transformer():
    """
    This function reverses the transformations of 'get_mnist_transformer'.
    This is required for the PGD algorithm
    """
    return torchvision.transforms.Compose([
        ToTensor(),
        Normalize((0.,), (1/0.3081,)),
        Normalize((-0.1307,), (1.,))
    ])


def get_mnist_dataloader(split, batch_size=64, shuffle=True):
    if split == "train":
        dataloader = DataLoader(
            MNIST('./data/MNIST/', train=True, download=True,
                  transform=get_mnist_transformer()),
            batch_size=batch_size, shuffle=True)

    elif split == "test":
        dataloader = DataLoader(
            MNIST('./data/MNIST/', train=False, download=True,
                  transform=get_mnist_transformer()),
            batch_size=batch_size, shuffle=True)

    else:
        raise Exception("The specified split '" +
                        str(split) + "' is not supported. Please use one of ['train', 'test']")

    return dataloader
