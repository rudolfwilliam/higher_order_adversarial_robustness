#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Compose
import joblib

from getter import getter
from pathlib import Path
from CURE.CURE import CURELearner
from utils.config import CIFAR_CONFIG  # , CIFAR_CONFIG_RESNET20
from utils.utils import pgd

from datetime import datetime


def train_epoch(dataloader, model, optimizer=None, device="cpu", attack=None, **kwargs):
    """
    Based on this excellent tutorial: https://adversarial-ml-tutorial.org/adversarial_training/
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()

    model = model.to(device)
    total_loss, total_err = 0., 0.
    for X, y_true in dataloader:
        X, y_true = X.to(device), y_true.to(device)

        if attack is not None:
            X_input = attack(X, model, targets=y_true, device=device, **kwargs)
        else:
            X_input = X

        y_pred = model(X_input)
        loss = nn.CrossEntropyLoss()(y_pred, y_true)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_err += (y_pred.max(dim=1)[1] != y_true).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(dataloader.dataset), total_loss / len(dataloader.dataset)


def adversarial_training(config, use_pretrained_model=False, mixed_training=False, plot_results=True):

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
    if type(transformer.transforms[0]) == ToTensor:
        transformer = Compose(transformer.transforms[1:])

    if use_pretrained_model:
        net_CURE = CURELearner(model, trainloader, testloader, lambda_=config["lambda_"], transformer=transformer, trial=None,
                               image_min=config["image_min"], image_max=config["image_max"], device=config["device"], path=checkpoint_path / "best_model.data")

        # Load the checkpoint
        net_CURE.import_state(checkpoint_path / config["checkpoint_file"])

        model = net_CURE.net

    # Set the optimizer
    optimizer = getattr(optim, config["optimization_algorithm"])(
        model.parameters(), **config["optimizer_arguments"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10**6, gamma=1)

    # epsilon=8, targets=None, step_size=0.04, num_steps=20, normalizer=None, clip_min=0, clip_max=255, device="cpu"
    pgd_args = {
        'epsilon': config['epsilon'],
        'normalizer': transformer,
        'clip_min': config['image_min'],
        'clip_max': config['image_max'],
    }

    clean_err_train, clean_err_test = [], []
    clean_loss_train, clean_loss_test = [], []
    adv_err_train, adv_err_test = [], []
    adv_loss_train, adv_loss_test = [], []

    # Perform adversarial training
    for epoch in range(config["epochs"]):
        print("\nStarting epoch ", epoch, ": ", datetime.now())
        adv_err_train_epoch, adv_loss_train_epoch = train_epoch(trainloader, model, optimizer=optimizer, device=config['device'],
                                                                attack=pgd, **pgd_args)
        print("After adversarial step: ", datetime.now())
        if mixed_training:
            clean_err_train_epoch, clean_loss_train_epoch = train_epoch(
                trainloader, model, optimizer=optimizer, device=config['device'])
            print("After normal step: ", datetime.now())
            clean_err_train.append(clean_err_train_epoch)
            clean_loss_train.append(clean_loss_train_epoch)

        clean_err_test_epoch, clean_loss_test_epoch = train_epoch(testloader, model, device=config['device'])
        print("After normal test step: ", datetime.now())
        adv_err_test_epoch, adv_loss_test_epoch = train_epoch(trainloader, model, device=config['device'], attack=pgd, **pgd_args)
        print("After adversarial test step: ", datetime.now())

        adv_err_train.append(adv_err_train_epoch)
        adv_loss_train.append(adv_loss_train_epoch)
        clean_err_test.append(clean_err_test_epoch)
        clean_loss_test.append(clean_loss_test_epoch)
        adv_err_test.append(adv_err_test_epoch)
        adv_loss_test.append(adv_loss_test_epoch)

        scheduler.step()

    return clean_err_train, clean_err_test, clean_loss_train, clean_loss_test, adv_err_train, adv_err_test, adv_loss_train, adv_loss_test


if __name__ == "__main__":
    config = CIFAR_CONFIG
    config["epochs"] = 20
    results = adversarial_training(config)
    print("RESULTS = ", results)
    checkpoint_path = Path("./data/checkpoints/")
    joblib.dump(results, checkpoint_path / "cached_adv_training.data")
