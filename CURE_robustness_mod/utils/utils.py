import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
# from torch.autograd.gradcheck import zero_gradients
import time
import shutil
import sys
import numpy as np


def read_vision_dataset(path, batch_size=128, num_workers=4, dataset='CIFAR10', transform=None):
    '''
    Read dataset available in torchvision

    Arguments:
        dataset : string
            The name of dataset, it should be available in torchvision
        transform_train : torchvision.transforms
            train image transformation
            if not given, the transformation for CIFAR10 is used
        transform_test : torchvision.transforms
            train image transformation
            if not given, the transformation for CIFAR10 is used
    Return:
        trainloader, testloader
    '''
    if not transform and dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    trainset = getattr(datasets, dataset)(
        root=path, train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = getattr(datasets, dataset)(
        root=path, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def zero_gradients(tensor):
    if tensor.grad is not None:
        tensor.grad.detach_()
        tensor.grad.zero_()


def pgd(inputs, net, epsilon=8, targets=None, step_size=0.04, num_steps=20, normalizer=None, clip_min=0, clip_max=255, device="cpu"):
    # We don't want to change the parameters of the NN
    net.eval()

    # If the images are normalized, normalize also 'epsilon' and the boundaries of the image domain
    # ('clip_min' and 'clip_max')
    if normalizer is not None:
        epsilon = epsilon / (clip_max - clip_min)
        #clip_min = normalizer(torch.tensor(clip_min, dtype=torch.float32).view(1, 1, 1, 1)).to(device).view([])
        #clip_max = normalizer(torch.tensor(clip_max, dtype=torch.float32).view(1, 1, 1, 1)).to(device).view([])
        clip_min = normalizer(torch.repeat_interleave(torch.tensor(clip_min, dtype=torch.float32), 3).view(1, 3, 1, 1)).to(device)
        clip_max = normalizer(torch.repeat_interleave(torch.tensor(clip_max, dtype=torch.float32), 3).view(1, 3, 1, 1)).to(device)
        epsilon = epsilon * (clip_max - clip_min)

    # Compute the maximum perturbation
    pert_max = inputs.detach().clone().to(device).requires_grad_(False) + epsilon
    pert_min = inputs.detach().clone().to(device).requires_grad_(False) - epsilon

    # The perturbed images
    pert_images = inputs.detach().clone().to(device).requires_grad_(True)

    # Randomize the starting point of PGD for each image
    pert_images = pert_images + (torch.rand(inputs.shape, device=device) - 0.5) * 2 * epsilon
    pert_images.clamp_(min=clip_min, max=clip_max)

    # Perform the 'num_steps' projected gradient descent steps
    for step in range(num_steps):
        # Reset the gradient of the pertubed images
        zero_gradients(pert_images)
        pert_images.requires_grad_()

        # Compute the predictions for the perturbed images
        predictions = net(pert_images)

        # Compute the gradient w.r.t the input
        loss_wrt_label = nn.CrossEntropyLoss()(predictions, targets)
        grad = torch.autograd.grad(
            loss_wrt_label, pert_images, only_inputs=True, create_graph=False, retain_graph=False)[0]

        # Update the perturbed images with small perturbations
        pert_images = pert_images + step_size * grad.sign()

        # Project the images back into the image-domain
        pert_images.clamp_(min=clip_min, max=clip_max)

        # Project the images back into the epsilon-box around the input images
        pert_images.clamp_(min=pert_min, max=pert_max)

    # Project the images back into the image-domain
    pert_images.clamp_(min=clip_min, max=clip_max)

    return pert_images


def pgd_old(inputs, net, epsilon=[1.], targets=None, step_size=0.04, num_steps=20, epsil=5./255.*8, transformer=None, inverse_transformer=None, device="cpu"):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and
       perturbed image
    """
    input_shape = inputs.shape
    pert_image = torch.tensor(inputs.detach().clone(),
                              requires_grad=True).to(device)
    w = torch.zeros(input_shape)
    r_tot = torch.zeros(input_shape)

    pert_image = pert_image + (torch.rand(inputs.shape).to(device)-0.5)*2*epsil
    pert_image = pert_image.to(device)

    for ii in range(num_steps):
        pert_image.requires_grad_()
        # zero_gradients(pert_image)
        if pert_image.grad is not None:
            pert_image.grad.detach_()
            pert_image.grad.zero_()
        fs = net.eval()(pert_image)
        loss_wrt_label = nn.CrossEntropyLoss()(fs, targets)
        grad = torch.autograd.grad(
            loss_wrt_label, pert_image, only_inputs=True, create_graph=True, retain_graph=False)[0]

        dr = torch.sign(grad.data)
        pert_image.detach_()
        pert_image += dr * step_size
        for i in range(inputs.size(0)):
            pert_image[i] = torch.min(
                torch.max(pert_image[i], inputs[i] - epsil), inputs[i] + epsil)
            if inverse_transformer is not None:
                pert_image[i] = inverse_transformer(pert_image[i])
            pert_image[i] = torch.clamp(pert_image[i], 0., 1.)
            if transformer is not None:
                pert_image[i] = transformer(pert_image[i])

    r_tot = pert_image - inputs
    regul = np.linalg.norm(r_tot.cpu().flatten(
        start_dim=1, end_dim=-1), np.inf, axis=1)
    regul = torch.Tensor(regul).view(-1, 1, 1, 1).to(device)
    r_tot = r_tot / regul

    return r_tot.cpu()


def pgd_original(inputs, net, targets=None, step_size=0.04, num_steps=20, epsilon=5./255.*8, normalizer=None, clip_min=0, clip_max=255, device="cpu"):
    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and
       perturbed image
    """
    input_shape = inputs.shape
    pert_image = inputs.clone().to(device)
    w = torch.zeros(input_shape)
    r_tot = torch.zeros(input_shape)

    denormal = transforms.Compose([transforms.Normalize((0., 0., 0.), (1/0.2023, 1/0.1994, 1/0.2010)),
                                   transforms.Normalize((-0.4914, -0.4822, -0.4465), (1., 1., 1.))])
    normal = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    pert_image = pert_image + (torch.rand(inputs.shape).to(device)-0.5)*2*epsilon
    pert_image = pert_image.to(device)

    for ii in range(num_steps):
        pert_image.requires_grad_()
        zero_gradients(pert_image)
        fs = net.eval()(pert_image)
        loss_wrt_label = nn.CrossEntropyLoss()(fs, targets)
        grad = torch.autograd.grad(loss_wrt_label, pert_image, only_inputs=True, create_graph=True, retain_graph=False)[0]

        dr = torch.sign(grad.data)
        pert_image.detach_()
        pert_image += dr * step_size
        for i in range(inputs.size(0)):
            pert_image[i] = torch.min(torch.max(pert_image[i], inputs[i] - epsilon), inputs[i] + epsilon)
            pert_image[i] = pert_image[i] / torch.Tensor([1/0.2023, 1/0.1994, 1/0.2010]).view(3, 1, 1).to(device)
            pert_image[i] -= torch.Tensor([-0.4914, -0.4822, -0.4465]).view(3, 1, 1).to(device)
            pert_image[i] = torch.clamp(pert_image[i], 0., 1.)
            pert_image[i] = (pert_image[i] - torch.Tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device))
            pert_image[i] /= torch.Tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)

            #pert_image[i] = normal(torch.clamp(pert_image[i], 0., 1.))[None, :, :, :]

    r_tot = pert_image - inputs
    regul = np.linalg.norm(r_tot.cpu().flatten(start_dim=1, end_dim=-1), np.inf, axis=1)
    regul = torch.Tensor(regul).view(-1, 1, 1, 1).to(device)
    r_tot = r_tot / regul

    inputs_pert = inputs.to(device) + epsilon * r_tot.to(device)

    return inputs_pert


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
term_width, _ = shutil.get_terminal_size()
term_width = int(term_width)


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
