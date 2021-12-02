#!/usr/bin/env python
# coding: utf-8

# # Robustness via curvature regularization, and vice versa
# This notebooks demonstrates how to use the CURE algorithm for training a robust network.
import sys
print(sys.version_info)

import torch
print(torch.__version__)

import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')


from CURE.CURE_mod import ModCURELearner
from simple_MNIST_model import Net
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import MNIST
import torch.nn.functional as F
import torch.optim as optim
import torch

# if model does not exist yet, it needs to be trained once
train = False
make_robust = True

checkpoint_file = 'checkpoint/checkpoint_02.data'

batch_size_train = 64
batch_size_test = 1000

if train:
    n_epochs = 1
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    trainloader = torch.utils.data.DataLoader(
        MNIST('./files/', train=True, download=True,
              transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize((0.1307,), (0.3081,))
              ])),
        batch_size=batch_size_train, shuffle=True)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(trainloader.dataset) for i in range(n_epochs + 1)]

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

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
                torch.save(network.state_dict(), './pretrained/MNIST_model.pth')
    for epoch in range(1, n_epochs + 1):
        train(epoch, trainloader=trainloader, optimizer=optimizer, log_interval=log_interval, train_losses=train_losses, train_counter=train_counter)
else:
    network = Net()
    network.load_state_dict(torch.load("./pretrained/MNIST_model.pth"))

trainloader = torch.utils.data.DataLoader(
              MNIST('./files/', train=True, download=True,
              transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize((0.1307,), (0.3081,))
              ])),
              batch_size=batch_size_train, shuffle=True)

testloader = torch.utils.data.DataLoader(
              MNIST('./files/', train=False, download=True,
              transform=torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize((0.1307,), (0.3081,))
              ])),
              batch_size=batch_size_test, shuffle=True)


net_CURE = ModCURELearner(network, trainloader, testloader, lambda_=1, device='cpu', path="./checkpoint/best_model.data")

# **Set the optimizer**

net_CURE.set_optimizer(optim_alg='SGD', args={'lr':1e-4})

# **Train the model**
if make_robust:
    h = [0.1]
    net_CURE.train(epochs=1, h=h)

    net_CURE.save_state(checkpoint_file)
else:
    net_CURE.import_state(checkpoint_file)

net_CURE.plot_results()
