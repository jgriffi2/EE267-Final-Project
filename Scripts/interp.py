import numpy as np
from model import *
from preprocess import *

"""
Function: interpScores
======================
Interpolates the scores using probability distribution.
======================
input:
    scores: scores of sample based on model
    y: unique datapoints to interpolate
output:
    interp_scores: interpolated look_vec
"""
def interpScores(scores, y):
    reshapeScores = scores.reshape(scores.shape[1])

    # All values in normScores are in range [0, 1)
    max_scores = reshapeScores - np.max(reshapeScores)
    normScores = np.exp(max_scores) / np.sum(np.exp(max_scores))

    # Make look_vec a unit vector
    scores_to_sum = normScores[:, None] * y
    summed_scores = np.sum(scores_to_sum, axis=0)
    norm = np.linalg.norm(summed_scores)
    norm = 1 if (norm == 0) else norm
    interp_scores = summed_scores / norm

    return interp_scores

"""
Function: interp
================
Calculates the accuracy of interpolating between unique datapoints.
================
input:
    path_to_model: location of model to use for interpolation
    path_to_ys: location of ys
    path_to_uniques: location of most uniques ys
    path_to_samples: location of samples to interpolate
output:
    None
"""
def interp(path_to_model, path_to_ys="../Ys/ys", path_to_uniques="../Uniques/uniques",
           path_to_samples="../Samples/samples"):
    m = loadData(path_to_model)
    d = loadData(path_to_ys)
    u = loadData(path_to_uniques)
    samples = reformData(loadData(path_to_samples))

    N = samples.shape[0]

    num_correct = 0

    differences = np.zeros(N)

    for c, (x, y) in enumerate(samples):
        scores = m(x)

        interp_d = interpScores(scores.data.numpy(), u)

        correct_d = d[c]

        differences[c] = np.linalg.norm(interp_d - correct_d)

        if (differences[c] < 1e-5):
            num_correct += 1

    print('Accuracy is %d out of %d: %.4f' % (num_correct, N, num_correct / N))
