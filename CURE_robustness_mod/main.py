#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
import torch
import os
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from CURE.CURE import CURELearner
from data.mnist import get_mnist_dataloader, get_mnist_transformer, get_mnist_inverse_transformer
from models.mnist_simple_model import get_simple_mnist_model

# # Robustness via curvature regularization, and vice versa
# This notebooks demonstrates how to use the CURE algorithm for training a robust network.


print(os.getcwd())


############################################################################################
# BEGIN CONFIGURATION

# Getter functions
get_model = get_simple_mnist_model
get_dataloader = get_mnist_dataloader
get_transformer = get_mnist_transformer
get_inverse_transformer = get_mnist_inverse_transformer

# Constants
device = "cpu"

batch_size_train = 64
batch_size_test = 1000

shuffle_train = True

# CURE configurations
lambda_ = 1
h = [0.1, 0.4, 0.8, 1.8, 3]
optimization_algorithm = 'SGD'
optimizer_arguments = {
    'lr': 1e-4
}
epochs = 10

use_checkpoint = False
checkpoint_file = 'checkpoint_01.data'

# END CONFIGURATION
############################################################################################

checkpoint_path = Path("./data/checkpoints/")

# Load the base model (train it if needed)
model = get_model()

# Load the dataset
trainloader = get_dataloader(
    split="train", batch_size=batch_size_train, shuffle=shuffle_train)
testloader = get_dataloader(split="test", batch_size=batch_size_test)

# Create the net_cure model
transformer = get_transformer()
inverse_transformer = get_inverse_transformer()
net_CURE = CURELearner(model, trainloader, testloader, lambda_=lambda_, transformer=transformer,
                       inverse_transformer=inverse_transformer, device=device, path=checkpoint_path / "best_model.data")

# Set the optimizer
net_CURE.set_optimizer(optim_alg=optimization_algorithm,
                       args=optimizer_arguments)

# Train the net-cure model
if use_checkpoint:
    net_CURE.import_state(checkpoint_file)

else:
    net_CURE.train(epochs=epochs, h=h)

    net_CURE.save_state(checkpoint_path / checkpoint_file)

net_CURE.plot_results()
