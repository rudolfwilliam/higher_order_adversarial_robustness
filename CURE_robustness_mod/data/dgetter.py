from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor, Normalize


class DataGetter():
    """Helper class for getting various torchvision transformations and data loaders
    
       Args:
            self.dataset (torch.utils.data.Dataset): data set from which to load the data
            self.name ("MNIST" or "CIFAR10"): Name of data set used
    """
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

    def get_transformer(self):
        """Get transformer depending on self.name

        Returns:
            torchvision.transforms.Transforms: Normalization based on data set depending on attribute self.name"""
        # The operations which are being applied to each image before training
        if self.name == 'MNIST':
            return torchvision.transforms.Compose([
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
            ])
        elif self.name == 'CIFAR10':
            return torchvision.transforms.Compose([
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010))
            ])

    def get_inverse_transformer(self):
        """Reverses the transformations of 'get_transformer'. This is required for the PGD algorithm"""
        if self.name == 'CIFAR10':
            return torchvision.transforms.Compose([
                Normalize((0, 0, 0),
                          (1 / 0.2023, 1 / 0.1994, 1 / 0.2010)),
                Normalize((-0.4914, -0.4822, -0.4465),
                          (1, 1, 1)),
            ])
        if self.name == 'MNIST':
            return torchvision.transforms.Compose([
                ToTensor(),
                Normalize((0.,), (1/0.3081,)),  # Std of MNIST /255 = 0.3081
                Normalize((-0.1307,), (1.,))   # Mean of MNIST /255 = 0.1307
            ])

    def get_dataloader(self, split, batch_size=64, shuffle=True):
        """Get a dataloader for specified parameters.

           Args:
                split ("train" ord "test"): split for data set to use
                batch_size (int): batch size of data set
                shuffle (bool: if True, will shuffle the data set, otherwise not
            
            Returns:
                torch.utils.data.DataLoader: The dataloader
            Raises:
                Exception: If specified split is neither "train" nor "test"
           """

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
