from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from data.dgetter import DataGetter


def getter(dataset, modelname):
    """Getter function for returning functions for getting dataloader, transformer, inverse_transformer and model

    Args:
        dataset ('MNIST or 'CIFAR10'): Dataset
        modelname ('SimpleModel, ResNet18, ResNet20, AlexNet or VGG'): Model name
    
    Returns:
        getter functions for dataloader, transformer, inverse_transformer and model"""
        
    if dataset == 'CIFAR10':
        print(CIFAR10)
        dgetter = DataGetter(CIFAR10, 'CIFAR10')
        get_dataloader = dgetter.get_dataloader
        get_transformer = dgetter.get_transformer
        get_inverse_transformer = dgetter.get_inverse_transformer

        print(modelname)
        if modelname == 'SimpleModel':
            from models.cifar_simple_model import ModelGetter
            get_model = ModelGetter(dgetter, 'cifar_simple_model').get_simple_model
        elif modelname == 'ResNet18':
            from models.cifar_resnet18 import ModelGetter
            get_model = ModelGetter(dgetter, 'cifar10_resnet18').get_model
        elif modelname == 'ResNet20':
            from models.cifar_resnet20 import ModelGetter
            get_model = ModelGetter(dgetter, 'cifar10_resnet20').get_model
        elif modelname == 'AlexNet':
            from models.cifar_alexnet import ModelGetter
            get_model = ModelGetter(dgetter,'cifar10_alexnet').get_model
        elif modelname == 'VGG':
            from models.cifar_vgg import ModelGetter
            get_model = ModelGetter(dgetter,'cifar10_vgg').get_model
    elif dataset == 'MNIST':
        print(MNIST)
        dgetter = DataGetter(MNIST, 'MNIST')
        get_dataloader = dgetter.get_dataloader
        get_transformer = dgetter.get_transformer
        get_inverse_transformer = dgetter.get_inverse_transformer
        if modelname == 'SimpleModel':
            from models.mnist_simple_model import ModelGetter
            get_model = ModelGetter(dgetter, 'mnist_simple_model').get_simple_model

    return get_dataloader, get_transformer, get_inverse_transformer, get_model
