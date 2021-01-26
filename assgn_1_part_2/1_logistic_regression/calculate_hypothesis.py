import numpy as np
from sigmoid import *


def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    hypothesis = 0.0
    for x in range(0, len(theta)):
        hypothesis += theta[x] * X[i][x]
    result = sigmoid([hypothesis])

    return result
