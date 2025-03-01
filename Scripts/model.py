import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
from preprocess import *
import os
from plotter import *
import random

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
Function: initializeWeights
===========================
Initializes the weights of the model using the xavier uniform method.
===========================
input:
    m: model
output:
    None
"""
def initializeWeights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)

"""
Function: eyeModel
==================
Constructs a model for eye tracking.
==================
input:
    H: height of imgs
    W: width of imgs
    num_classes: number of classes
output:
    m: model created
"""
def eyeModel(H, W, num_classes):
    """
    Archtitectures:
    Convolutional layer (filter size 7x7 | 32 filters)
    ReLU
    MaxPool (2x2)
    Convolutional layer (filter size 5x5 | 32 filters)
    ReLU
    MaxPool (2x2)
    Convolutional layer (filter size 3x3 | 64 filters)
    ReLU
    MaxPool (2x2)
    Flatten
    Fully Connected
    """

    m = nn.Sequential(
        nn.Conv2d(3, 32, 7, stride=1, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(H*W, num_classes)
    )

    m.apply(initializeWeights)

    return m

"""
Function: getOptimizer
======================
Returns an optimizer for the type specified by optimType.
======================
input:
    m: model to create optimizer for
    optimType: type of optimizer
        'adam', 'rmsprop', 'sgd'
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
    elif (optimType == 'rmsprop'):
        optimizer = optim.RMSprop(m.parameters(), lr=lr, alpha=alpha, momentum=momentum)
    elif (optimType == 'sgd'):
        optimizer = optim.SGD(m.parameters(), lr=lr, momentum=momentum)
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
    acc: accuracy of m on data_val
"""
def checkAccuracy(m, data_val):
    num_correct = 0
    num_samples = 0

    # Set model to evaluation mode
    m.eval()

    with torch.no_grad():
        for x, y in data_val:
            # Get scores
            scores = m(x)

            # Predictions of correct response
            _, preds = scores.max(1)

            # Determine the number of correct values
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

    return acc

"""
Function: train
===============
Trains the model m and saves it to path_to_model.
===============
input:
    m: model to be trained
    data_train: training set of the data
    data_val: validation set of the data
    opt_params: parameters for getOptimizer
        (type, lr, alpha, betas, momentum)
    model_name: name of the model running
    path_to_model: directory where the model will be saved
    path_to_loss: directory to save loss figure
    path_to_acc: directory to save accuracy figure
    num_epochs: number of epochs to run
    show_every: number to print statistics every show_every iteration
output:
    None
"""
def train(m, data_train, data_val, opt_params, model_name, path_to_model="../Models/",
          path_to_loss="../Plots/Loss/", path_to_acc="../Plots/Accuracy/", num_epochs=10, show_every=500):
    print("=====Training=====")

    iter_count = 0
    type, lr, alpha, betas, momentum = opt_params
    optimizer = getOptimizer(m, optimType=type, lr=lr, betas=betas, momentum=momentum)

    loss_array = np.zeros(data_train.shape[0] * num_epochs, dtype=np.float)
    acc_train = np.zeros(num_epochs, dtype=np.float)
    acc_val = np.zeros(num_epochs, dtype=np.float)

    for epoch in range(num_epochs):
        for c, (x, y) in enumerate(data_train):
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

            # Store loss in loss_array
            loss_array[iter_count] = loss.item()

            # Print the update of the loss
            if (iter_count % show_every == 0):
                print('Iteration %d, loss = %.4f' % (iter_count, loss.item()))
                checkAccuracy(m, data_val)
                print()

            iter_count += 1

        acc_train[epoch] = checkAccuracy(m, data_train)
        acc_val[epoch] = checkAccuracy(m, data_val)

    # Save model
    saveData(m, path_to_model + model_name)

    # Plot the loss
    plotLoss(loss_array, path_to_loss + model_name + ".png")

    # Plot the accuracy
    plotAccuracy(acc_train, acc_val, path_to_acc + model_name + ".png")

"""
Function: reformData
====================
Converts the tuple of numpy arrays to tuple of tensors.
====================
input:
    data: tuple of numpy arrays
output:
    newData: tuple of tensors
"""
def reformData(data):
    N = data.shape[0]
    C, H, W = data[0][0].shape

    newData = np.zeros((N, 2), dtype=tuple)

    for c, (x, y) in enumerate(data):
        # Convert x and y to correct data structure
        x = torch.tensor(x.reshape(1, C, H, W))
        x = x.to(dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.long)

        newData[c] = (x, y)

    return newData

"""
Function: model
===============
Trains or tests a model defined by mode and model_num.
===============
input:
    mode: mode to determine if we train or test
        'train': states we will be training
        'test': states we will be testing
    path_to_model: path where model is located or where it will be saved
    opt_params: parameters for getOptimizer
        (type, lr, alpha, betas, momentum)
    normalize: determines whether to normalize data
    path_to_samples: path to where the array data is located
    path_to_uniques: path to where the unique_y data is located
    setup_mode: mode to use for setup function
        'load': states we will be loading data
        'save': states we will be creating and saving the data
output:
    None
"""
def model(mode, path_to_model="../Models/", opt_params=('adam', 1e-3, 0.9, (0.5, 0.999), 0.9),
          normalize=True, path_to_samples="../samples/samples", path_to_uniques="../Uniques/uniques",
          setup_mode='load'):
    data, y = setup(path_to_samples, path_to_uniques, mode=setup_mode)
    C, H, W = data[0][0].shape

    data = normalizeData(data) if (normalize) else data

    data_train, data_val, data_test = splitData(data)
    data_train, data_val, data_test = reformData(data_train), reformData(data_val), reformData(data_test)

    num_classes = len(y)

    type, lr, alpha, betas, momentum = opt_params
    model_name = "model_" + type + "_" + str(normalize) + "_" + str(lr) + "_" + str(alpha) + "_" + str(betas) + "_" + str(momentum)

    model_to_use = eyeModel(H, W, num_classes) if (mode == 'train') else loadData(path_to_model + model_name)

    if (mode == 'train'):
        train(model_to_use, data_train, data_val, opt_params, model_name)
    elif (mode == 'test'):
        checkAccuracy(model_to_use, data_test)

"""
Function: testHyperParameters
=============================
Tests the hyper parameters of normalization and type of optimizer with standard
learning rate, alpha, betas, and momentum.
=============================
input:
    mode: mode to determine if we train or test
        'train': states we will be training
        'test': states we will be testing
    normalize: determines whether to normalize data
    type: type of optimizer
        'adam', 'rmsprop', 'sgd'
output:
    None
"""
def testHyperParameters(mode, normalize, type):
    lr, alpha, betas, momentum = 1e-3, 0.9, (0.5, 0.999), 0.9
    opt_params = (type, lr, alpha, betas, momentum)

    print("=====%s=====" % type)

    model(mode, opt_params=opt_params, normalize=normalize)
