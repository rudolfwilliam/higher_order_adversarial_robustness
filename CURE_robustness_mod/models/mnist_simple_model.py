from pathlib import Path

import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
import torch
from torch import nn
import torch.nn.functional as F
import data.mnist as mnist


def get_simple_mnist_model():
    model_path = Path("./models/pretrained/MNIST_model.pth")

    # If the model does not yet exists, train it
    if not model_path.exists():
        print("Train MNIST model")
        train_simple_mnist_model(n_epochs=10, learning_rate=0.01,
                                 momentum=0.5, seed=1, log_interval=10)

    # Finally, load the model
    network = SimpleModel()
    network.load_state_dict(torch.load(model_path))

    return network


def train_simple_mnist_model(n_epochs=10, learning_rate=0.01, momentum=0.5, seed=1, log_interval=10):

    trainloader = mnist.get_mnist_dataloader(split="train")

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
                           './models/pretrained/MNIST_model.pth')

    for epoch in range(1, n_epochs + 1):
        train(epoch, trainloader=trainloader, optimizer=optimizer,
              log_interval=log_interval, train_losses=train_losses, train_counter=train_counter)


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
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
