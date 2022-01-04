#!/usr/bin/env python
# coding: utf-8

from getter import getter
from pathlib import Path
from CURE.CURE import CURELearner
from utils.config import CIFAR_CONFIG  # , CIFAR_CONFIG_RESNET20


def train_CURE(config, plot_results=True, trial=None):
    """
    The training function.

    Args:
        config (dict): Configuration dictionary that provides training procedure with all necessary hyper parameters. 
        plot_results (bool): If True, results will be plotted.
        trial (optuna.trial.Trial): Necessary for hyper parameter optimization using Optuna (https://optuna.org/). 
                                    A process for evaluating an objective function. 
                                    See (https://optuna.readthedocs.io/en/stable/reference/trial.html?highlight=trial) for details.
    Returns:
        CURELearner: The trained model.
    """

    get_dataloader, get_transformer, _, get_model = getter(
        config["dataset"], config["model_name"])

    # Construct (and if needed train) model
    model = get_model()

    checkpoint_path = Path("./data/checkpoints/")

    # Load the dataset
    trainloader = get_dataloader(
        split="train", batch_size=config["batch_size_train"], shuffle=config["shuffle_train"])
    testloader = get_dataloader(split="test", batch_size=config["batch_size_test"])

    # Create the net_cure model
    transformer = get_transformer()
    net_CURE = CURELearner(model, trainloader, testloader, lambda_0=config["lambda_0"], lambda_1=config["lambda_1"], lambda_2=config["lambda_2"], transformer=transformer, trial=trial,
                           image_min=config["image_min"], image_max=config["image_max"], device=config["device"], path=checkpoint_path / "best_model.data", acc=config["accuracy"])

    # Set the optimizer
    net_CURE.set_optimizer(optim_alg=config["optimization_algorithm"],
                           args=config["optimizer_arguments"])

    # Train the net-cure model
    if config["use_checkpoint"]:
        net_CURE.import_state(checkpoint_path / config["checkpoint_file"])

    else:
        net_CURE.train(epochs=config["epochs"], h=config["h"], epsilon=config["epsilon"])

        net_CURE.save_state(checkpoint_path / config["checkpoint_file"])

    if plot_results:
        net_CURE.plot_results()

    return net_CURE


if __name__ == "__main__":
    train_CURE(CIFAR_CONFIG)
