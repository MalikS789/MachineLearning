import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """

    hypothesis = 0.0
    for x in range(0, len(theta)):
        if x > 1:
            hypothesis += pow(X[i, 1], x) * theta[x]
        else:
            hypothesis += X[i, x] * theta[x]

    return hypothesis
