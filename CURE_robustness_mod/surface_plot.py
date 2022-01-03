import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from CURE.CURE import CURELearner
from utils.config import CIFAR_CONFIG
from getter import getter

import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

device = "cuda"


PATH = "./data/checkpoints/checkpoint_01.data"
config = CIFAR_CONFIG
get_dataloader, get_transformer, get_inverse_transformer, get_model = getter(
    config["dataset"], config["model_name"])
model = get_model()
model = model.to(device)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: {}".format(total_params))
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: {}".format(trainable_params))
transformer = get_transformer()
inverse_transformer = get_inverse_transformer()
trainloader = get_dataloader(split="train", batch_size=config["batch_size_train"], shuffle=config["shuffle_train"])
testloader = get_dataloader(split="test", batch_size=config["batch_size_test"],shuffle=False)

L = nn.CrossEntropyLoss()

if not torch.cuda.is_available() and device == 'cuda':
    raise ValueError("cuda is not available")

img_shape = (3,32,32)


flatten = False
delta = 10
n_points = 10000
res = int(np.sqrt(n_points))

for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to(device), targets.to(device)
    inputs.requires_grad_()
    outputs = model.eval()(inputs)

    # TODO: Don't calculate whole gradient here but in next loop per image
    loss = L(outputs, targets)
    gradient = torch.autograd.grad(loss, inputs, create_graph=True)[0]
    if flatten:
        gradient = torch.flatten(gradient,start_dim=1)
    normal = F.normalize(gradient,dim=1)

    v = torch.rand_like(normal)
    v = F.normalize(v,dim=1)

    for k, x in enumerate(inputs):
        scalars = np.linspace(-delta,delta,res)
        grid = torch.empty(res,res)
        for i in range(res):
            for j in range(res):
                x_star = x
                if flatten:
                    x_star = torch.flatten(x,start_dim=0)
                x_star = x_star + scalars[i]*normal[k] + scalars[j]*v[k]
                x_star = x_star.reshape((1,)+img_shape)
                y_star = model.eval()(x_star)
                y_true = torch.zeros_like(y_star)
                grid[i,j] = L(y_star, targets[k].unsqueeze(0))
        grid = grid.detach().numpy()

        scalars = np.outer(scalars,np.ones(res))
        masks = [scalars, scalars.T]
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(masks[0],masks[1],grid,cmap='viridis',edgecolor='none')
        ax.set_xlabel('Surface Normal Direction')
        ax.set_ylabel('Random Direction')
        plt.show()

        if k > 3:
            exit()
