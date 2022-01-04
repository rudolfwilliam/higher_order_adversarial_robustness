import torch
import torch.nn as nn
import torch.nn.functional as F
from getter import getter
from CURE.CURE import CURELearner

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def lossplot(config: dict, save_path: str = None) -> None:
    """Plots the negative of the loss surface. One axis represents the normal direction; the other is a random direction."""
    device = config["device"]

    get_dataloader, get_transformer, get_inverse_transformer, get_model = getter(
        config["dataset"], config["model_name"])
    trainloader = get_dataloader(split="train", batch_size=config["batch_size_train"], shuffle=config["shuffle_train"])
    testloader = get_dataloader(split="test", batch_size=config["batch_size_test"], shuffle=False)
    
    model = get_model()

    if config["use_checkpoint"]:
        checkpoint_path = Path("./data/checkpoints/")
        
        transformer = get_transformer()
        net_CURE = CURELearner(model, trainloader, testloader, lambda_0=config["lambda_0"], lambda_1=config["lambda_1"], lambda_2=config["lambda_2"], transformer=transformer, trial=None,
                            image_min=config["image_min"], image_max=config["image_max"], device=config["device"], path=checkpoint_path / "best_model.data", acc=config["accuracy"])

        net_CURE.set_optimizer(optim_alg=config["optimization_algorithm"],
                            args=config["optimizer_arguments"])
        
        net_CURE.import_state(checkpoint_path / config["checkpoint_file"])

        model = net_CURE.net


    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: {}".format(total_params))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}".format(trainable_params))
    transformer = get_transformer()
    inverse_transformer = get_inverse_transformer()


    L = nn.CrossEntropyLoss()

    img_shape = (3, 32, 32)


    flatten = False
    delta = 30
    res = 101
    n_points = res**2

    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad_()
        outputs = model.eval()(inputs)

        loss = L(outputs, targets)
        gradient = torch.autograd.grad(loss, inputs, create_graph=True)[0]
        gradient = torch.flatten(gradient, start_dim=1)
        normal = F.normalize(gradient, dim=1)

        v = torch.rand_like(torch.zeros(normal.shape), device=device)
        v = F.normalize(v, dim=1)

        if not flatten:
            normal = normal.reshape(inputs.shape)
            v = v.reshape(inputs.shape)

        for k, x in enumerate(inputs):
            scalars = np.linspace(-delta, delta, res)
            grid = torch.empty(res, res)
            for i in range(res):
                for j in range(res):
                    x_star = x
                    if flatten:
                        x_star = torch.flatten(x, start_dim=0)
                    x_star = x_star + scalars[i]*normal[k] + scalars[j]*v[k]
                    x_star = x_star.reshape((1,)+img_shape)
                    y_star = model.eval()(x_star)
                    y_true = torch.zeros_like(y_star)
                    grid[i, j] = L(y_star, targets[k].unsqueeze(0)).detach()
            grid = grid.detach().numpy()

            scalars = np.outer(scalars, np.ones(res))
            masks = [scalars, scalars.T]
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(masks[0], masks[1], grid, cmap='viridis', edgecolor='none')
            ax.scatter(0, 0, grid[res // 2, res // 2])
            ax.set_xlabel('Surface Normal Direction')
            ax.set_ylabel('Random Direction')
            if save_path is not None: plt.savefig(save_path + f"loss_{k}")
            plt.show()
            plt.pause(.001) # Prevents blocking

            if k > 3:
                exit()
