import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import preprocess
import os

"""
Class: Flatten
==============
Flattens a tensor of shape (N, C, H, W) to be of shape (N, C*H*W).
==============
"""
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)

"""
Class: Unflatten
==============
Unflattens a tensor of shape (N, C*H*W) to be of shape (N, C, H, W).
==============
"""
class Unflatten(nn.Module):
    def __init__(self, N=-1, C=128, H=7, W=7):
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initializeWeights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)

def eyeModel():
    m = nn.Squential(

    )

    return m

"""
Function: getOptimizer
======================
Returns an optimizer for the type specified by optimType.
======================
input:
    m: model to create optimizer for
    optimType: type of optimizer
        'adam', 'sparseadam', 'adamax', 'rmsprop', 'sgd', 'nesterovsgd'
    lr: learning rate
    alpha: alpha value for optimizer
    betas: beta1 and beta2 values for optimizer
    momentum: momentum for optimizer
output:
    optimizer: optimizer for the model m
"""
def getOptimizer(m, optimType='adam', lr=1e-3, alpha=0.9, betas=(0.5, 0.999), momentum=0.9):
    optimizer = None

    if (optimType == 'adam'):
        optimizer = optim.Adam(m.parameters(), lr=lr, betas=betas)
    elif (optimType == 'sparseadam'):
        optimizer = optim.SparseAdam(m.parameters(), lr=lr, betas=(beta1, beta2))
    elif (optimType == 'adamax'):
        optimizer = optim.Adamax(m.parameters(), lr=lr, betas=(beta1, beta2))
    elif (optimType == 'rmsprop'):
        optimizer = optim.RMSprop(m.parameters(), lr=lr, alpha=alpha, momentum=momentum)
    elif (optimType == 'sgd'):
        optimizer = optim.SGD(m.parameters(), lr=lr, momentum=momentum)
    elif (optimType == 'nesterovsgd'):
        optimizer = optim.SGD(m.parameters(), lr=lr, momentum=momentum, nesterov=True)
    else:
        print("Unsupported optimizer type")

    return optimizer

def train(m, data_train, loss_function, batch_size=128, num_epochs=10, show_every=100):
    iter_count = 0
    initializeWeights(m)
    solver = getOptimizer(m)

    for epoch in range(num_epochs):
        for c, (x, y) in enumerate(data_train):
            # Print the update of the loss
            if (iter_count % show_every == 0):
                print("Iter: {}, Loss: {}".format(iter_count, loss))

            iter_count += 1

def setup(path_to_data, data_to_retrieve):
    N_data = countData(path_to_data)
    m = np.zeros((N_data), 1)
    for path, subdirs, files in os.walk(path_to_data):
        for data in files:
            if (data.endswith('.png')):
            elif (data.endswith('.pkl')):
