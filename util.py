import math
import numpy as np


def get_index_of(arr, item):
    for i in range(len(arr)):
        if arr[i] == item:
            return i

    return -1


def to_precision(number, precision):
    if precision == 0:
        return int(number)

    number *= 10. ** precision
    number = int(number)
    number /= 10. ** precision

    return number


def get_angle_between(p1, p2):
    y = p2["y"] - p1["y"]
    x = p2["x"] - p1["x"]

    return math.atan2(y, x)


def get_distance_between(p1, p2):
    terminal = False

    a, b = p2["x"] - p1["x"], p2["y"] - p1["y"]
    c = math.sqrt(a ** 2 + b ** 2)

    if c < .5:
        terminal = True

    return c, terminal


def flatten(state, state_dim):
    state_flattened = np.zeros(state_dim)
    last_element = len(state_flattened) - 1

    state_flattened[0] = state[0]
    state_flattened[1:last_element] = state[1].reshape(1, -1)
    state_flattened[last_element] = state[2]

    return state_flattened
