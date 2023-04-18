import numpy as np


def KNN_precision(indices_NN, instance_mask, include_first=False):

    instance_NN = instance_mask[indices_NN]

    # TP = 0
    # total_NN = np.multiply(instance_NN.shape[0], instance_NN.shape[1] - 1)
    correct_mask = np.zeros(instance_NN.shape)

    for col in range(0, instance_NN.shape[1]):  # exluding the original NN (always correct)
        correct_mask[:, col] = instance_NN[:, 0] == instance_NN[:, col]

    Precision = np.mean(correct_mask)
    at_least_one_incorrect = correct_mask.all(axis=1) == False

    return Precision, at_least_one_incorrect
