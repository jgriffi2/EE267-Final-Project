import numpy as np
from model import *
from preprocess import *

def scoreData(m, datapoint):
    return m(datapoint)

def interpScores(scores, y):
    norm = np.linalg.norm(scores)
    norm = 1 if (norm == 0) else norm

    reshapeScores = scores.reshape(scores.shape[1])

    # All values in normScores are in range [0, 1)
    normScores = np.sort(reshapeScores)[::-1] / norm

    scores_to_sum = normScores[:, None] * y

    summed_scores = np.sum(scores_to_sum, axis=0)

    norm = np.linalg.norm(summed_scores)
    norm = 1 if (norm == 0) else norm

    # Make look_vec a unit vector
    interp_scores = summed_scores / norm

    return interp_scores

def structureData(path_to_data):
    data = loadData(path_to_data)

    N = data.shape[0]

    newData = np.zeros((N, 3))

    for n in range(N):
        newData[n] = np.asarray(data[n].get('look_vec'))

    return newData

def interp(path_to_model, path_to_data, path_to_unique, path_to_samples):
    m = loadData(path_to_model)
    d = structureData(path_to_data)
    u = structureData(path_to_unique)
    samples = loadData(path_to_samples)

    N = samples.shape[0]

    num_correct = 0

    for c, (x, y) in enumerate(samples):
        C, H, W = x.shape
        x = torch.tensor(x.reshape(1, C, H, W))
        x = x.to(dtype=torch.float32)

        scores = scoreData(m, x)

        interp_d = interpScores(scores.data.numpy(), u)

        correct_d = d[c]

        if (np.linalg.norm(interp_d - correct_d) < 1e-5):
            num_correct += 1

    print('Accuracy is %d out of %d: %.4f' % (num_correct, N, num_correct / N))
