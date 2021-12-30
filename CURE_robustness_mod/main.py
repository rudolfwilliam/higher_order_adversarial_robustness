#!/usr/bin/env python
# coding: utf-8

from getter import getter
from pathlib import Path
from CURE.CURE import CURELearner
from utils.config import CIFAR_CONFIG, CIFAR_CONFIG_RESNET20


def train_CURE(config, plot_results=True, trial=None):
    """
    The main function.
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
    net_CURE = CURELearner(model, trainloader, testloader, lambda_=config["lambda_"], transformer=transformer, trial=trial,
                           image_min=config["image_min"], image_max=config["image_max"], device=config["device"], path=checkpoint_path / "best_model.data")

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

    return net_CURE.test_acc_adv


if __name__ == "__main__":
    train_CURE(CIFAR_CONFIG_RESNET20)
