from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    # res[200:, :] = -0.3
    res[250:, :] = -0.0
    res[250:, :320] = 0.5
    res[320:, 100:320] = 1
    res[250:, 200:320] = 1
    # res[100:, :140] =1
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    # res[200:, :] = -0.3
    res[250:, :] = -0.0
    res[250:, 300:] = 0.5
    # res[300:, 220:] = 0.5
    res[320:, 300:540] = 1
    res[250:, 300:420] = 1
    # res[100:, 500:] =1
    # res[350:, 250:] = 1
    return res
