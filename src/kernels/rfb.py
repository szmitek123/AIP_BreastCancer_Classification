import numpy as np
from typing import Callable

def RFB(gamma: np.float64 = 1) -> Callable[[ np.ndarray, np.ndarray ], np.float64 ]:
    def rfb(u: np.ndarray, v: np.ndarray) -> np.ndarray | np.float64:
        if v.ndim == 2: return np.exp(-np.sum((u - v) ** 2, axis=1) / (2 * gamma ** 2))

        # default
        return np.exp(-np.sum((u - v) ** 2) / (2 * gamma ** 2))


    return rfb
