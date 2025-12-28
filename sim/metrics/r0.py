from __future__ import annotations

import numpy as np


def r0_like(new_adopters: np.ndarray, next_new_adopters: np.ndarray) -> np.ndarray:
    denom = np.maximum(new_adopters, 1)
    return next_new_adopters / denom
