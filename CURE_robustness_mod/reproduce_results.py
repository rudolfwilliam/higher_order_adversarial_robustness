#!/usr/bin/env python
# coding: utf-8

# # Robustness via curvature regularization, and vice versa
# This notebooks demonstrates how to use the CURE algorithm for training a robust network.



# In[2]:


import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')


# In[3]:


import sys
import numpy as np
from CURE.CURE import CURELearner
import matplotlib.pyplot as plt
from utils.utils import read_vision_dataset
from utils.resnet import ResNet18
import torchvision.transforms as T
from torch import nn
from torchvision.models import resnet18


# **Read the DataLoader**

# In[7]:


# Build custom transforms for our custom model
transforms = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# In[9]:


trainloader, testloader = read_vision_dataset('../data', transform=transforms)


# **Import the network**

# In[10]:


# network = ResNet18()
network = resnet18(pretrained=True)


# **Initialize the class**

# In[11]:


net_CURE = CURELearner(network, trainloader, testloader, lambda_=1, device='cpu')

# **Set the optimizer**

# In[12]:


net_CURE.set_optimizer(optim_alg='Adam', args={'lr':1e-4})


# **Import the pre-trained model**

# In[8]:


#net_CURE.import_model('../checkpoint/ckpt.t7')


# **Train the model**

# In[13]:


h = [0.1, 0.4, 0.8, 1.8, 3]
net_CURE.train(epochs=10, h=h)

net_CURE.save_state('../checkpoint/checkpoint_00.data')

# **Plot the results**

# In[ ]:


#net_CURE.plot_results()


# In[ ]:




