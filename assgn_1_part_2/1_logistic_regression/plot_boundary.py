import numpy as np
from calculate_hypothesis import *


def plot_boundary(X_normalized, theta_final, ax1):
    min_x1 = -3.0
    max_x1 = 3.0
    x2_on_min_x1 = 0.0
    x2_on_max_x1 = 0.0

    x2_on_min_x1 = theta_final[0] + theta_final[1] * min_x1
    x2_on_max_x1 = theta_final[0] + theta_final[1] * max_x1

    x_array = np.array([min_x1, max_x1])
    y_array = np.array([x2_on_min_x1, x2_on_max_x1])
    ax1.plot(x_array, y_array, c='black', label='decision boundary')

    # add legend to the subplot
    ax1.legend()
