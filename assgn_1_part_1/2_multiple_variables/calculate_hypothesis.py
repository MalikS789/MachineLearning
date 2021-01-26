import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """

    hypothesis = 0.0
    for x in range(0, len(theta)):
        hypothesis += X[i, x] * theta[x]

    return hypothesis
