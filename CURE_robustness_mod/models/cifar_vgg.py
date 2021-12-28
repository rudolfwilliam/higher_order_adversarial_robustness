from pathlib import Path

import torch.optim as optim
import torchvision

import torch
from torch import nn
import torch.nn.functional as F
from data.dgetter import DataGetter

VGG = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)

class ModelGetter():
    def __init__(self, data_getter, model_name):
        self.data_getter = data_getter
        self.model_name = model_name

    def get_model(self):
        model_path = Path("./models/pretrained/"+self.model_name+".pth")

        # If the model does not yet exists, train it
        if not model_path.exists():
            print("Train model")
            self.train_model(n_epochs=10, learning_rate=0.01,
                                     momentum=0.5, seed=1, log_interval=10)

        # Finally, load the model
        network = VGG
        network.load_state_dict(torch.load(model_path))

        return network

    def train_model(self, n_epochs=10, learning_rate=0.01, momentum=0.5, seed=1, log_interval=10):
        trainloader = self.data_getter.get_dataloader(split="train")

        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i * len(trainloader.dataset) for i in range(n_epochs + 1)]

        torch.manual_seed(seed)

        network = VGG
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=momentum)

        def train(epoch, trainloader, optimizer, log_interval, train_losses, train_counter):
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

        for epoch in range(1, n_epochs + 1):
            train(epoch, trainloader=trainloader, optimizer=optimizer,
                  log_interval=log_interval, train_losses=train_losses, train_counter=train_counter)
