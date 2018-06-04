import pickle
import os
import numpy as np
import operator
import cv2

"""
Function: saveData
==================
Saves the data to the specified location path_to_data. The location must already
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
Retrieves the data the user wants from the pickled data in path_to_data.
=================
input:
    path_to_data: path where pickled data is located
    data_to_retrieve: the data the user wants from path_to_data in a list
        'look_vec': the 3D gaze direction in camera space
        'head_pose': a 3x3 matrix rotation from world space to camera space
output:
    desired_data: dict of the data the user wants, determined by data_to_retrieve
"""
def getData(path_to_data, data_to_retrieve):
    data = loadData(path_to_data)

    desired_data = {}

    for d in data_to_retrieve:
        cur_data = data.get(d)

        # Error check to see if valid data to retrieve
        if (cur_data == None):
            print(d + " is not valid data to retrieve.")
            continue

        desired_data[d] = cur_data

    return desired_data

"""
Function: getMetaData
===================
Counts the number of unique images we have in our dataset, as well as the height
and width of each image.
===================
input:
    path_to_data: path where pickled data is located
output:
    count: number of unique images we have in our dataset
    H: height of each image
    W: width of each image
"""
def getMetaData(path_to_data):
    count = 0
    H, W = None, None
    for path, subdirs, files in os.walk(path_to_data):
        for data in files:
            if (data.endswith('.png')):
                if (count == 0):
                    H, W = cv2.imread(path + '/' + data, 0).shape
                count += 1

    return count, H, W

"""
Function: countUniqueData
=========================
Counts the number of unique data points based on 'look_vec' from the pikled data.
=========================
input:
    path_to_data: path where pickled data is located
output:
    None
"""
def countUniqueData(path_to_data):
    count = 0
    unique = []
    for path, subdirs, files in os.walk(path_to_data):
        for data in files:
            if (data.endswith('.pkl')):
                d = getData(path + '/' + data, {'look_vec'})
                if (isUnique(unique, d)):
                    unique.append(d)
                    count += 1

    print("%d unique data based on 'look_vec'" % count)

"""
Function: isUnique
==================
Returns True if value is unique, False otherwise.
==================
input:
    unique: unique values from 'look_vec' in pickled data
    d: 'look_vec' data to compare to current unique values
output:
    boolean: True if unique, False otherwise
"""
def isUnique(unique, d):
    keys = d.keys()
    for i in range(len(unique)):
        u = unique[i]

        uVals = u.get('look_vec')
        dVals = d.get('look_vec')
        diff = False

        for uc, uVal in enumerate(uVals):
            for dc, dVal in enumerate(dVals):
                if (uc != dc):
                    continue
                if (abs(uVal - dVal) > 1e-5):
                    diff = True
        if (diff == False):
            return False

    return True

"""
Function: computeDist
=====================
Computes the distance from direction y1 to direction y2.
=====================
input:
    y1: direction from 'look_vec'
    y2: direction from 'look_vec'
output:
    dist: distance from y1 to y2
"""
def computeDist(y1, y2):
    dist = 0
    for i in range(len(y1)):
        dist += abs(y1[i] - y2[i])

    return dist

"""
Function: gatherData
====================
Gathers all of the necessary data and save in in path_to_save.
====================
input:
    path_to_data: path where pickled data is located
    data_to_retrieve: the data the user wants from path_to_data in a list
        'look_vec': the 3D gaze direction in camera space
        'head_pose': a 3x3 matrix rotation from world space to camera space
    path_to_save: directory where to save the data
output:
    y: the data we just saved
"""
def gatherData(path_to_data, data_to_retrieve, path_to_save):
    y = []

    for path, subdirs, files in os.walk(path_to_data):
        for data in files:
            if (data.endswith('.pkl')):
                d = getData(path + '/' + data, data_to_retrieve)
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
    path_to_ys: location where y values are located
    path_to_save: location to save distance values
output:
    yDist: distance values
"""
def saveDistance(path_to_ys, path_to_save):
    y = loadData(path_to_ys)

    N = y.shape[0]
    yDist = np.zeros(N)

    for n1 in range(N):
        for n2 in range(N):
            if (n1 == n2):
                continue

            y1 = y[n1].get('look_vec')
            y2 = y[n2].get('look_vec')

            yDist[n1] += computeDist(y1, y2)

    saveData(yDist, path_to_save)

    return yDist

"""
Function: createUniqueYs
========================
Finds the most unique values of the 'look_vec' data based on threshold.
========================
input:
    path_to_unique: directory where unique data will be loaded from or saved to
    path_to_ys: path where ys is located
    path_to_dist: directory where distance values are located
    mode: states whether we will be loading data or creating data
        'load': states we will be loading data
        'save': states we will be creating and saving the data
output:
    unique_ys: the M most unique values of the data we have based on threshold
"""
def createUniqueYs(path_to_unique, path_to_ys="../../Ys/ys", path_to_dist="../../Dists/dists", mode='load'):
    unique_ys = None

    if (mode == 'load'):
        print("=====Loading unique_ys=====")
        unique_ys = loadData(path_to_unique)
    elif (mode == 'save'):
        print("=====Creating unique_ys=====")

        y = loadData(path_to_ys)
        yDist = loadData(path_to_dist)
        N = y.shape[0]

        unique_y_indices = np.argsort(yDist)[::-1]

        # M = 1
        #
        # for n in range(1, N, 1):
        #     y1 = yDist[unique_y_indices[n-1]]
        #     y2 = yDist[unique_y_indices[n]]
        #
        #     compVal = y1 if (y2 == 0) else y1 / y2
        #
        #     if (compVal < threshold):
        #         break
        #
        #     M += 1

        M = int(N * 0.005) # 0.005 is a constant value we determined

        unique_y_indices = unique_y_indices[:M]
        unique_ys = y[unique_y_indices]

        saveData(unique_ys, path_to_unique)
        print("M = %d" % M)
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
    M = unique_ys.shape[0]

    y_to_compare = y.get('look_vec')
    closest_y = 0
    closest_dist = computeDist(unique_ys[closest_y].get('look_vec'), y_to_compare)

    for m in range(M):
        cur_y = unique_ys[m].get('look_vec')
        dist = computeDist(cur_y, y_to_compare)
        if (dist < closest_dist):
            closest_dist = dist
            closest_y = m

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
    path_to_save: location to save data array
    path_to_unique: directory where unique data will be loaded from
    path_to_data: path where pickled data is located
    data_to_retrieve: the data the user wants from path_to_data in a list
        'look_vec': the 3D gaze direction in camera space
        'head_pose': a 3x3 matrix rotation from world space to camera space
    mode: states whether we will be loading data or creating data
        'load': states we will be loading data
        'save': states we will be creating and saving the data
output:
    array_data: the data in an array format
"""
def setup(path_to_save, path_to_unique, path_to_data="../SynthEyes_data", data_to_retrieve={'look_vec'}, mode='load'):
    print("=====In setup=====")

    array_data, unique_ys = None, None

    if (mode == 'load'):
        array_data = loadData(path_to_save)
        unique_ys = loadData(path_to_unique)
    elif (mode == 'save'):
        unique_ys = loadData(path_to_unique)

        list_data = []
        actual_ys = []

        for path, subdirs, files in os.walk(path_to_data):
            for data in files:
                if (data.endswith('.png')):
                    x = cv2.imread(path + '/' + data, cv2.IMREAD_COLOR)
                    x = np.transpose(x, axes=(2, 0, 1))
                    y = getData(path + '/' + data[:-3] + 'pkl', data_to_retrieve)
                    actual_ys.append(y)
                    y = findClosestData(unique_ys, y)
                    list_data.append((x, y))

        array_data = np.asarray(list_data)
        actual_ys = np.asarray(actual_ys)

        saveData(array_data, path_to_save)
        saveData(actual_ys, "../../ActualYs/actual_ys_test")
    else:
        print("Incorrect mode type")

    return array_data, unique_ys
