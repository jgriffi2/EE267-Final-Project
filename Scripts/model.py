import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
from preprocess import *
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

# TODO: may need to remove this
def initializeWeights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)

def eyeModel(model_num, H, W, num_classes):
    """
    Archtitectures:
    ex.1
    Convolutional layer (filter size 7x7 | 24 filters)
    ReLU
    MaxPool (2x2)
    Convolutional layer (filter size 5x5 | 24 filters)
    ReLU
    MaxPool (2x2)
    Convolutional layer (filter size 3x3 | 24 filters)
    ReLU
    MaxPool (2x2)
    Flatten
    Fully Connected

    ex.2
    For each eye
    Convolutional layer (filter size 3x3 | 32 filters)
    Convolutional layer (filter size 3x3 | 32 filters)
    Convolutional layer (filter size 3x3 | 64 filters)
    Flatten
    get fully connected layer (128 neurons)
    """

    m = None

    if (model_num == 0): # Architecture 0
        m = nn.Squential(
            nn.Conv2d(3, 24, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 24, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(24*H*W / np.pow(2, 6), num_classes)
        )
    elif (model_num == 1): # Architecture 1
        m = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            Flatten(),
            nn.Linear(64*H*W, num_classes)
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

"""
Function: checkAccuracy
=======================
Checks the accuracy of the model m on the validation data data_val.
=======================
input:
    m: model whose accuracy we are checking
    data_val: validation set of the data
output:
    None
"""
def checkAccuracy(m, data_val):
    num_correct = 0
    num_samples = 0

    # Set model to evaluation mode
    m.eval()

    with torch.no_grad():
        for x, y in data_val:
            # Convert x to correct data structure
            C, H, W = x.shape
            x = torch.tensor(x.reshape(1, C, H, W))
            x = x.to(dtype=torch.float32)
            y = torch.tensor([y], dtype=torch.long)

            # Get scores
            scores = m(x)

            # Predictions of correct response
            _, preds = scores.max(1)

            # Determine the number of correct values
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

"""
Function: train
===============
Trains the model m and saves it to path_to_model.
===============
input:
    m: model to be trained
    data_train: training set of the data
    data_val: validation set of the data
    path_to_model: directory where the model will be saved
    num_epochs: number of epochs to run
    show_every: number to print statistics every show_every iteration
output:
    None
"""
def train(m, data_train, data_val, path_to_model, num_epochs=10, show_every=100):
    print("=====Training=====")

    iter_count = 0
    # initializeWeights(m) TODO: may need to remove this
    optimizer = getOptimizer(m)

    for epoch in range(num_epochs):
        for c, (x, y) in enumerate(data_train):
            # Convert x to correct data structure
            C, H, W = x.shape
            x = torch.tensor(x.reshape(1, C, H, W))
            x = x.to(dtype=torch.float32)
            y = torch.tensor([y], dtype=torch.long)

            # Put model into training mode
            m.train()

            # Determine scores from model
            scores = m(x)

            # Determine loss
            loss = F.cross_entropy(scores, y)

            # Zero out all grads in the optimizer
            optimizer.zero_grad()

            # Perform backward pass from loss
            loss.backward()

            # Update parameters of the model
            optimizer.step()

            # Print the update of the loss
            if (iter_count % show_every == 0):
                print('Iteration %d, loss = %.4f' % (iter_count, loss.item()))
                checkAccuracy(m, data_val)
                print()

            iter_count += 1

    saveData(m, path_to_model)

"""
Function: model
===============
Trains or tests a model defined by mode and model_num.
===============
input:
    path_to_model: path where model is located or where it will be saved
    mode: mode to determine if we train or test
        'train': states we will be training
        'test': states we will be testing
    model_num: number that specifies the model we'll be using
    path_to_data: path to where the array data is located
    path_to_unique: path to where the unique_y data is located
    setup_mode: mode to use for setup function
        'load': states we will be loading data
        'save': states we will be creating and saving the data
output:
    None
"""
def model(path_to_model, mode, model_num, path_to_data, path_to_unique, setup_mode='load'):
    data, y = setup(path_to_data, path_to_unique, mode=setup_mode)
    H, W, _ = data[0][0].shape
    data = reformulateData(data)
    data_train, data_val, data_test = splitData(data)

    num_classes = len(y)

    model_to_use = eyeModel(model_num, H, W, num_classes) if (mode == 'train') else loadData(path_to_model)

    if (mode == 'train'):
        train(model_to_use, data_train, data_val, path_to_model)
    elif (mode == 'test'):
        checkAccuracy(model_to_use, data_test)
