import matplotlib.pyplot as plt

"""
Function: plotLoss
==============
Plots the data and save it at the location given.
==============
input:
    data:
    save_location:
output:
"""
def plotLoss(data, save_location):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(data, 'o')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    fig.savefig(save_location)
    plt.close(fig)

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
