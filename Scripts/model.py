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
import cv2

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

def train(m, data_train, data_val, num_epochs=10, show_every=100):
    iter_count = 0
    initializeWeights(m)
    optimizer = getOptimizer(m)

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

            # Print the update of the loss
            if (iter_count % show_every == 0):
                print('Iteration %d, loss = %.4f' % (iter_count, loss.item()))
                checkAccuracy(m, data_val)
                print()

            iter_count += 1

"""
Function: setup
===============
Sets up the data in an array format.
===============
input:
    path_to_data: path where pickled data is located
    data_to_retrieve: the data the user wants from path_to_data in a list
        'look_vec': the 3D gaze direction in camera space
        'head_pose': a 3x3 matrix rotation from world space to camera space
        'ldmks': a dict containing the following 2D and 3D landmarks
            'ldmks_lids_2d', 'ldmks_iris_2d', 'ldmks_pupil_2d' in screen space
            'ldmks_lids_3d', 'ldmks_iris_3d', 'ldmks_pupil_3d' in camera space
output:
    array_data: the data in an array format
"""
def setup(path_to_data, data_to_retrieve):
    N_data = countData(path_to_data)

    list_data = []

    for path, subdirs, files in os.walk(path_to_data):
        for data in files:
            if (data.endswith('.png')):
                x = cv2.imread(path + '/' + data, cv2.IMREAD_COLOR)
                y = getData(path + '/' + data[:-3] + 'pkl', data_to_retrieve)
                list_data.append((x, y))

    array_data = np.asarray(list_data)

    return array_data

"""
Function: splitData
===================
Splits data into training, validation, and testing sets with a 60, 20, 20 split.
===================
input:
    data: data to split
output:
    data_train: training set
    data_val: validation set
    data_test: testing set
"""
def splitData(data):
    data_train, data_val, data_test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])

    return torch.from_numpy(data_train), torch.from_numpy(data_val), torch.from_numpy(data_test)

# TODO: finish this function
def model(inputModel, data, train, test):
    model_to_use = eyeModel if (train) else inputModel

    data_train, data_val, data_test = splitData(data)

    if (train):
        train(model_to_use, data_train, data_val)
    if (test):
        checkAccuracy(model_to_use, data_test)
