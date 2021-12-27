import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor, Normalize


class DataGetter():
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

    def get_transformer(self):
        # The operations which are being applied to each image before training
        if self.name == 'MNIST':
            return torchvision.transforms.Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])
        elif self.name == 'CIFAR10':
            return torchvision.transforms.Compose([
                ToTensor(),
                # Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
                # Normalize((125.30691805, 122.95039414, 113.86538318), (62.99321928, 62.08870764, 66.70489964))
                Normalize(0.5, 0.5)
            ])

    def get_inverse_transformer(self):
        """
        This function reverses the transformations of 'get_transformer'.
        This is required for the PGD algorithm
        """
        if self.name == 'CIFAR10':
            return torchvision.transforms.Compose([
                ToTensor(),
                Normalize((0.,), (1/0.3081,)),  # Std of CIFAR10 /255 = 0.3081
                Normalize((-0.2516,), (1.,))   # Mean of CIFAR10 /255 = 0.2516
            ])
        if self.name == 'MNIST':
            return torchvision.transforms.Compose([
                ToTensor(),
                Normalize((0.,), (1/0.3081,)),  # Std of MNIST /255 = 0.3081
                Normalize((-0.1307,), (1.,))   # Mean of MNIST /255 = 0.1307
            ])

    def get_dataloader(self, split, batch_size=64, shuffle=True):
        if split == "train":
            dataloader = DataLoader(
                self.dataset('./data/'+self.name+'/', train=True, download=True,
                             transform=self.get_transformer()),
                batch_size=batch_size, shuffle=True)

        elif split == "test":
            dataloader = DataLoader(
                self.dataset('./data/'+self.name+'/', train=False, download=True,
                             transform=self.get_transformer()),
                batch_size=batch_size, shuffle=True)

        else:
            raise Exception("The specified split '" +
                            str(split) + "' is not supported. Please use one of ['train', 'test']")
        return dataloader
