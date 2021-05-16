import numpy as np


def epsilon_function(e_max, e_min, episodes, period):
    number = episodes // period + 1

    epsilon = np.linspace(e_max, e_min, period)

    return np.tile(epsilon, number)[:episodes]
