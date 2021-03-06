import numpy as np


def sigmoid(z):
    try:
        y = z.copy()
        for x in range(0, len(y)):
            y[x] = 1 / (1 + np.math.exp(-y[x]))
        return y
    except:
        return y
