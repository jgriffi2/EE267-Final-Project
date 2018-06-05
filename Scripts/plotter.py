import matplotlib.pyplot as plt

"""
Function: plotLoss
==============
Plots the loss and save it at the location given.
==============
input:
    data: loss data
    save_location: location to save figure
output:
    None
"""
def plotLoss(data, save_location):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(data, 'o')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    fig.savefig(save_location)
    plt.close(fig)

"""
Function: plotLoss
==============
Plots the loss and save it at the location given.
==============
input:
    acc_train: accuracy of training data
    acc_val: accuracy of validation data
    save_location: location to save figure
output:
    None
"""
def plotAccuracy(acc_train, acc_val, save_location):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(acc_train, '-o', label='train')
    ax.plot(acc_val, '-o', label='val')
    ax.plot([0.5] * len(acc_val), 'k--')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    fig.savefig(save_location)
    plt.close(fig)
