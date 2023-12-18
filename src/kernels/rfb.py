import numpy as np
from typing import Callable

def RFB(gamma: np.float64 = 1) -> Callable[[ np.ndarray, np.ndarray ], np.float64 ]:
    def rfb(u: np.ndarray, v: np.ndarray) -> np.float64:
        sum: np.float64 = 0

        for i in range(0, u.shape[0]):
            sum += (u[i] - v[i]) ** 2

        return np.exp( -sum / (2 * gamma ** 2))

    return rfb
