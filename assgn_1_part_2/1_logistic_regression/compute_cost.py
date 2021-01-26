import numpy as np
from calculate_hypothesis import *


def compute_cost(X, y, theta):
    """
        :param X            : 2D array of our dataset
        :param y            : 1D array of the groundtruth labels of the dataset
        :param theta        : 1D array of the trainable parameters
    """

    # initialize cost
    J = 0.0
    # get number of training examples
    m = y.shape[0]

    # Compute cost for logistic regression.
    for i in range(0, m):
        hypothesis = calculate_hypothesis(X, theta, i)[0]
        output = y[i]
        cost = 0.0

        if i > 0:
            for j in range(1, len(X[i])):
                cost += (-output * np.log(hypothesis * X[i][j])) - ((1 - output) * np.log(1 - hypothesis * X[i][j]))
        if not np.math.isnan(cost):
            J += cost
    J = J / m
    return J
