from pathlib import Path

import torch.optim as optim
import torchvision

import torch
from torch import nn
import torch.nn.functional as F

# Load PyTorch's ResNet18 that has been pretrained on the ImageNet dataset.
ResNet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

class ModelGetter():
    """Class that constructs model, trains it if necessary and can return it.

    Attributes:
            data_getter (DataGetter): Object that gathers loads dataset.
                                Defined in './data/dgetter.py' and passed in
                                './getter.py'.
            model_name (str): Name of the model to be loaded.
    Args:
            data_getter (obj): DataGetter object that gathers loads dataset.
                                Defined in './data/dgetter.py' and passed in
                                './getter.py'.
            model_name (str): Name of the model to be loaded.
    """
    def __init__(self, data_getter, model_name):
        self.data_getter = data_getter
        self.model_name = model_name

    def get_model(self):
        """
        Constructs neural network and calls training if it has not been pretrained.

        Returns:
            ResNet18: The ResNet18 model finetuned to the dataset.
        """
        model_path = Path("./models/pretrained/"+self.model_name+".pth")

        # If the model has not been finetuned to new dataset, finetune it
        if not model_path.exists():
            print("Train model")
            self.train_model(n_epochs=10, learning_rate=0.01,
                                     momentum=0.5, seed=1, log_interval=10)

        # Finally, load the model
        network = ResNet18
        network.load_state_dict(torch.load(model_path))

        return network

    def train_model(self, n_epochs=10, learning_rate=0.01, momentum=0.5, seed=1, log_interval=10):
        """Fuction that normally trains the neural network constructed by this class.

        Args:
            n_epochs (int): Number of epochs, i.e. passes through the dataset.
            learning_rate (float): Learning rate of the optimization algorithm.
            momentum (float): Momentum of the optimization algorithm.
            seed (int): Seed to make the training deterministic.
            log_interval (int): Number of steps to skip before logging progress.

        """
        # Get the dataloader for the traing phase.
        trainloader = self.data_getter.get_dataloader(split="train")

        # Define necessary variables for the training loop.
        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i * len(trainloader.dataset) for i in range(n_epochs + 1)]

        # Set seed for deterministic results.
        torch.manual_seed(seed)

        # State which model to train.
        network = ResNet18

        # State which optimizer to use.
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=momentum)

        def train(epoch, trainloader, optimizer, log_interval, train_losses, train_counter):
            """
            Function defining training loop of a single epoch.
            """
            network.train()
            for batch_idx, (data, target) in enumerate(trainloader):
                optimizer.zero_grad()
                output = network(data)
                #loss = nn.NLLLoss(output, target)
                #loss = nn.CrossEntropyLoss
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item()))
                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(trainloader.dataset)))
                    torch.save(network.state_dict(),
                               "./models//pretrained/"+self.model_name+".pth")
        # Loop calling the training loop for each epoch.
        for epoch in range(1, n_epochs + 1):
            train(epoch, trainloader=trainloader, optimizer=optimizer,
                  log_interval=log_interval, train_losses=train_losses, train_counter=train_counter)
