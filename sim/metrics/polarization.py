from __future__ import annotations

import numpy as np


def polarization_score(beliefs: np.ndarray) -> np.ndarray:
    mean = beliefs.mean(axis=0)
    var = beliefs.var(axis=0)
    extremes = ((beliefs < 0.2) | (beliefs > 0.8)).mean(axis=0)
    score = 0.5 * extremes + 0.5 * np.clip(4 * var, 0, 1)
    return np.clip(score, 0, 1)
