import pickle
import os
import numpy as np
import operator
import cv2
from scipy.spatial.distance import cdist

"""
Function: saveData
==================
Saves the data to the specified location path_to_data. The directory must already
be created.
==================
input:
    data: data to be saved
    path_to_data: path where data should be saved
output:
    None
"""
def saveData(data, path_to_data):
    pickle.dump(data, open(path_to_data, 'wb'))

"""
Function: loadData
======================
Unpickles the data from the file located in path_to_data.
======================
input:
    path_to_data: path where pickled data is located
output:
    data: dict of unpickled data
"""
def loadData(path_to_data):
    data = pickle.load(open(path_to_data, 'rb'))

    return data

"""
Function: getData
=================
Retrieves the look_vec data from the pickled data in path_to_data.
=================
input:
    path_to_data: path where pickled data is located
output:
    desired_data: array of the look_vec
"""
def getData(path_to_data="../SynthEyes_data"):
    data = loadData(path_to_data)
    desired_data = np.asarray(data.get('look_vec'))

    return desired_data

"""
Function: gatherData
====================
Gathers all of the necessary data and saves it in path_to_save.
====================
input:
    path_to_data: path where pickled data is located
    path_to_save: directory where to save the data
output:
    y: data in an array format
"""
def gatherData(path_to_data="../SynthEyes_data", path_to_save="../LookVecs/look_vecs"):
    y = []

    for path, subdirs, files in os.walk(path_to_data):
        for data in files:
            if (data.endswith('.pkl')):
                d = getData(path + '/' + data)
                y.append(d)

    y = np.asarray(y)

    saveData(y, path_to_save)

    return y

"""
Function: saveDistance
======================
Creates data object that holds the distance values between 'look_vec's.
======================
input:
    path_to_vecs: location where y values are located
    path_to_save: location to save distance values
output:
    yDist: distance values
"""
def saveDistance(path_to_vecs="../LookVecs/look_vecs", path_to_save="../Dists/dists"):
    y = loadData(path_to_vecs)

    yDist = np.sum(cdist(y, y), axis=0)

    saveData(yDist, path_to_save)

    return yDist

"""
Function: createUniqueYs
========================
Finds the most unique values of the 'look_vec' data based on a multiplier.
========================
input:
    path_to_uniques: directory where unique data will be loaded from or saved to
    path_to_vecs: path where ys is located
    path_to_dists: directory where distance values are located
    mode: states whether we will be loading data or creating data
        'load': states we will be loading data
        'save': states we will be creating and saving the data
    mult: multiplier to determine size of uniques, should be in range (0, 1)
output:
    unique_ys: the M most unique values of the data we have based on threshold
"""
def createUniqueYs(path_to_uniques="../Uniques/uniques", path_to_vecs="../LookVecs/look_vecs",
                   path_to_dists="../Dists/dists", mode='load', mult=0.005):
    unique_ys = None

    if (mode == 'load'):
        print("=====Loading unique_ys=====")
        unique_ys = loadData(path_to_uniques)
    elif (mode == 'save'):
        print("=====Creating unique_ys=====")

        # Load all needed data
        y = loadData(path_to_vecs)
        yDist = loadData(path_to_dists)

        # Compute M, based on constant value determined by us: 0.005
        M = int(y.shape[0] * mult)

        # Sort distances from largest to smallest and take M largest
        unique_y_indices = np.argsort(yDist)[::-1][:M]
        unique_ys = y[unique_y_indices]

        saveData(unique_ys, path_to_uniques)
    else:
        print("Incorrect mode type")

    return unique_ys

"""
Function: findClosestData
=========================
Finds the closest datapoint to y in unique_ys.
=========================
input:
    unique_ys: values of the most unique look vectors
    y: a look vector
output:
    closest_y: the closest value in unique_ys to y
"""
def findClosestData(unique_ys, y):

    dists = np.linalg.norm(unique_ys - y[None, :], axis=1)
    closest_y = np.argsort(dists)[0]

    return closest_y

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
    print("=====Splitting Data=====")
    data_train, data_val, data_test = np.split(data, [int(.6*len(data)), int(.8*len(data))], axis=0)

    return data_train, data_val, data_test

"""
Function: normalizeData
=======================
Normalizes each image in data to correct for lighting and shadows.
=======================
input:
    data: array of tuples, (img, class)
output:
    newData: array of tuples where img is normalized
"""
def normalizeData(data):
    N = data.shape[0]
    C, H, W = data[0][0].shape

    newData = np.copy(data)

    for n in range(N):
        mean_img = np.mean(data[n][0], axis=(1, 2))
        std_img = np.std(data[n][0], axis=(1, 2))
        std_img = np.where(std_img == 0, 1, std_img)
        newData[n][0] = (data[n][0] - mean_img[:, None, None]) / std_img[:, None, None]

    return newData

"""
Function: setup
===============
Sets up the data in an array format.
===============
input:
    path_to_samples: location of samples
    path_to_ys: location of ys
    path_to_uniques: directory where unique data will be loaded from
    path_to_data: path where pickled data is located
    mode: states whether we will be loading data or creating data
        'load': states we will be loading data
        'save': states we will be creating and saving the data
output:
    array_data: the data in an array format
"""
def setup(path_to_samples="../Samples/samples", path_to_ys="../Ys/ys",
          path_to_uniques="../Uniques/uniques", path_to_data="../SynthEyes_data", mode='load'):
    print("=====In setup=====")

    array_data, unique_ys = None, None

    if (mode == 'load'):
        array_data = loadData(path_to_samples)
        unique_ys = loadData(path_to_uniques)
    elif (mode == 'save'):
        unique_ys = loadData(path_to_uniques)

        list_data = []
        actual_ys = []

        for path, subdirs, files in os.walk(path_to_data):
            for data in files:
                if (data.endswith('.png')):
                    x = cv2.imread(path + '/' + data, cv2.IMREAD_COLOR)
                    x = np.transpose(x, axes=(2, 0, 1))

                    y = getData(path + '/' + data[:-3] + 'pkl')
                    actual_ys.append(y)

                    y = findClosestData(unique_ys, y)
                    list_data.append((x, y))

        array_data = np.asarray(list_data)
        actual_ys = np.asarray(actual_ys)

        saveData(array_data, path_to_samples)
        saveData(actual_ys, path_to_ys)
    else:
        print("Incorrect mode type")

    return array_data, unique_ys
