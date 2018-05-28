import pickle
import os

"""
Function: unpickleData
======================
Unpickles the data from the file located in path_to_data.
======================
input:
    path_to_data: path where pickled data is located
output:
    data: dict of unpickled data
"""
def unpickleData(path_to_data):
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
        'ldmks': a dict containing the following 2D and 3D landmarks
            'ldmks_lids_2d', 'ldmks_iris_2d', 'ldmks_pupil_2d' in screen space
            'ldmks_lids_3d', 'ldmks_iris_3d', 'ldmks_pupil_3d' in camera space
output:
    desired_data: dict of the data the user wants, determined by data_to_retrieve
"""
def getData(path_to_data, data_to_retrieve):
    data = unpickleData(path_to_data)

    desired_data = {}

    for d in data_to_retrieve:
        cur_data = data.get(d)

        # Error check to see if valid data to retrieve
        if (cur_data == None):
            print(d + " is not valid data to retrieve.")
            continue

        desired_data[d] = cur_data

    return desired_data

def countData(path_to_data):
    count = 0
    for path, subdirs, files in os.walk(path_to_data):
        for data in files:
            if (data.endswith('.png')):
                count += 1

    return count
