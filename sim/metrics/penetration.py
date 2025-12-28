from __future__ import annotations

import numpy as np


def cluster_penetration(adoption_mask: np.ndarray, communities: np.ndarray) -> np.ndarray:
    n_claims = adoption_mask.shape[1]
    penetration = np.zeros(n_claims, dtype=np.float32)
    unique_comms = np.unique(communities)
    for k in range(n_claims):
        reached = 0
        for comm in unique_comms:
            members = communities == comm
            if adoption_mask[members, k].any():
                reached += 1
        penetration[k] = reached / max(len(unique_comms), 1)
    return penetration
