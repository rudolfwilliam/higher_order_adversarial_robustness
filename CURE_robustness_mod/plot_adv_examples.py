#!/usr/bin/env python
# coding: utf-8
import torch
import os
import numpy as np
from torchvision.transforms import ToTensor, Compose, ToPILImage
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from getter import getter
from pathlib import Path
from CURE.CURE import CURELearner
from utils.config import CIFAR_CONFIG, CIFAR_CONFIG_RESNET20
from utils.utils import pgd


def plot_adv_examples(config):
    """
    The main function.
    """
    checkpoint_path = Path("./data/checkpoints/")

    get_dataloader, get_transformer, get_inverse_transformer, get_model = getter(
        config["dataset"], config["model_name"])

    # Construct (and if needed train) model
    model = get_model()

    # Load the test dataset
    testloader = get_dataloader(split="test", batch_size=config["batch_size_test"])

    # Create the net_cure model (required to load the checkpoint data)
    transformer = get_transformer()
    inverse_transformer = get_inverse_transformer()

    if type(transformer.transforms[0]) == ToTensor:
        transformer = Compose(transformer.transforms[1:])

    net_CURE = CURELearner(model, None, testloader, lambda_=config["lambda_"], transformer=transformer, trial=None,
                           image_min=config["image_min"], image_max=config["image_max"], device=config["device"], path=checkpoint_path / "best_model.data")

    # Load the checkpoint
    net_CURE.import_state(checkpoint_path / config["checkpoint_file"])

    model = net_CURE.net
    model = model.module.to(config["device"])
    model.eval()

    if (checkpoint_path / "inputs.data").is_file():
        inputs = torch.load(checkpoint_path / "inputs.data")
        inputs_pert = torch.load(checkpoint_path / "inputs_pert.data")
        orig_correct = torch.load(checkpoint_path / "orig_correct.data")
        pert_correct = torch.load(checkpoint_path / "pert_correct.data")
    else:
        inputs = None
        inputs_pert = None
        orig_correct = None
        pert_correct = None

        for batch_idx, (inputs_batch, targets) in enumerate(testloader):
            print("Processing batch ", batch_idx)
            inputs_batch = inputs_batch.to(config["device"])
            targets = targets.to(config["device"])

            outputs_orig = model(inputs_batch)
            probs_orig, predicted_orig = outputs_orig.max(1)
            orig_correct_batch = predicted_orig.eq(targets)

            inputs_pert_batch = pgd(inputs_batch, model, epsilon=config["epsilon"], targets=targets, step_size=0.04, num_steps=20,
                                    normalizer=transformer, device=config["device"], clip_min=config["image_min"], clip_max=config["image_max"])

            outputs_pert = model(inputs_pert_batch)
            probs_pert, predicted_pert = outputs_pert.max(1)
            pert_correct_batch = predicted_pert.eq(targets)

            if inputs is None:
                inputs = inputs_batch.cpu()
                inputs_pert = inputs_pert_batch.cpu()
                orig_correct = orig_correct_batch.cpu()
                pert_correct = pert_correct_batch.cpu()
            else:
                inputs = torch.cat([inputs, inputs_batch.cpu()], dim=0)
                inputs_pert = torch.cat([inputs_pert, inputs_pert_batch.cpu()], dim=0)
                orig_correct = torch.cat([orig_correct, orig_correct_batch.cpu()], dim=0)
                pert_correct = torch.cat([pert_correct, pert_correct_batch.cpu()], dim=0)

        torch.save(inputs, checkpoint_path / "inputs.data")
        torch.save(inputs_pert, checkpoint_path / "inputs_pert.data")
        torch.save(orig_correct, checkpoint_path / "orig_correct.data")
        torch.save(pert_correct, checkpoint_path / "pert_correct.data")

    class ImagePlotter:
        def __init__(self, start_index=0):
            self.fig = plt.figure()
            self.ax = self.fig.subplots()
            self.index = start_index

        def next_img(self, val):
            self.index = (self.index + 1) % inputs.shape[0]
            self.plot_image()

        def prev_img(self, val):
            self.index = (self.index - 1) % inputs.shape[0]
            self.plot_image()

        def plot_image(self):
            image = inputs[self.index]
            adv_image = inputs_pert[self.index]
            image_correct = orig_correct[self.index]
            adv_image_correct = pert_correct[self.index]

            image = inverse_transformer(image)
            adv_image = inverse_transformer(adv_image)

            real_max_error = (image - adv_image).abs().max() * 255

            combined = torch.cat([image, adv_image], dim=2)
            combined = ToPILImage(mode="RGB")(combined)

            self.ax.imshow(combined)
            orig_title = "" if image_correct == 1 else "in"
            adv_title = "" if adv_image_correct == 1 else "in"
            self.ax.set_title(f"Original image {orig_title}correctly classified\nAdversarial image {adv_title}correctly classified\nMaximal difference: {real_max_error}", fontdict={
                "fontsize": 25
            })

    plotter = ImagePlotter()
    axes1 = plt.axes([0.81, 0.000001, 0.1, 0.075])
    axes2 = plt.axes([0.71, 0.000001, 0.1, 0.075])
    bnext = Button(axes1, 'Next')
    bprev = Button(axes2, "Previous")
    bnext.on_clicked(plotter.next_img)
    bprev.on_clicked(plotter.prev_img)

    plotter.plot_image()
    plt.show()

    return net_CURE.test_acc_adv


if __name__ == "__main__":
    config = CIFAR_CONFIG_RESNET20
    config["device"] = "cpu"

    plot_adv_examples(config)
