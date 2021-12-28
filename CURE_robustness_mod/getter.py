from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from data.dgetter import DataGetter


def getter(dataset,modelname):
    if dataset == 'CIFAR10':
        print(CIFAR10)
        dgetter = DataGetter(CIFAR10,'CIFAR10')
        get_dataloader = dgetter.get_dataloader
        get_transformer = dgetter.get_transformer
        get_inverse_transformer = dgetter.get_inverse_transformer

        if modelname == 'SimpleModel':
            from models.cifar_simple_model import ModelGetter
            get_model = ModelGetter(dgetter,'cifar_simple_model').get_simple_model
        if modelname == 'ResNet18':
            from models.cifar_resnet18 import ModelGetter
            get_model = ModelGetter(dgetter,'cifar10_resnet18').get_model
            #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        if modelname == 'AlexNet':
            from models.cifar_alexnet import ModelGetter
            get_model = ModelGetter(dgetter,'cifar10_alexnet').get_model
            #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        if modelname == 'VGG':
            from models.cifar_vgg import ModelGetter
            get_model = ModelGetter(dgetter,'cifar10_vgg').get_model
            #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    if dataset == 'MNIST':
        print(MNIST)
        dgetter = DataGetter(MNIST,'MNIST')
        get_dataloader = dgetter.get_dataloader
        get_transformer = dgetter.get_transformer
        get_inverse_transformer = dgetter.get_inverse_transformer
        if modelname == 'SimpleModel':
            from models.mnist_simple_model import ModelGetter
            get_model = ModelGetter(dgetter,'mnist_simple_model').get_simple_model


    return get_dataloader, get_transformer, get_inverse_transformer, get_model
