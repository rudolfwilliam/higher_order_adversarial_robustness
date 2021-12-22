from pathlib import Path

import torch.optim as optim
import torchvision

import torch
from torch import nn
import torch.nn.functional as F
from data.getter import DataGetter


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class ModelGetter():
    def __init__(self, data_getter, model_name):
        self.data_getter = data_getter
        self.model_name = model_name

    def get_simple_model(self):
        model_path = Path("./models/pretrained/"+self.model_name+".pth")

        # If the model does not yet exists, train it
        if not model_path.exists():
            print("Train model")
            self.train_simple_model(n_epochs=10, learning_rate=0.01,
                                     momentum=0.5, seed=1, log_interval=10)

        # Finally, load the model
        network = SimpleModel()
        network.load_state_dict(torch.load(model_path))

        return network

    def train_simple_model(self, n_epochs=10, learning_rate=0.01, momentum=0.5, seed=1, log_interval=10):
        trainloader = self.data_getter.get_dataloader(split="train")

        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i * len(trainloader.dataset) for i in range(n_epochs + 1)]

        torch.manual_seed(seed)

        network = SimpleModel()
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=momentum)

        def train(epoch, trainloader, optimizer, log_interval, train_losses, train_counter):
            network.train()
            for batch_idx, (data, target) in enumerate(trainloader):
                optimizer.zero_grad()
                output = network(data)
                loss = F.nll_loss(output, target)
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
