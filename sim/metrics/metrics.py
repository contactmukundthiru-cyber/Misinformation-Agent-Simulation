from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from sim.metrics.penetration import cluster_penetration
from sim.metrics.polarization import polarization_score
from sim.metrics.r0 import r0_like


def compute_daily_metrics(
    day: int,
    beliefs: np.ndarray,
    prev_beliefs: np.ndarray,
    trust: Dict[str, np.ndarray],
    adoption_threshold: float,
    communities: np.ndarray | None,
    prev_new_adopters: np.ndarray,
    cluster_penetration_enabled: bool = True,
) -> Tuple[List[Dict[str, float]], np.ndarray]:
    adoption_mask = beliefs >= adoption_threshold
    adoption_fraction = adoption_mask.mean(axis=0)
    mean_belief = beliefs.mean(axis=0)
    variance = beliefs.var(axis=0)
    polarization = polarization_score(beliefs)
    new_adopters = ((beliefs >= adoption_threshold) & (prev_beliefs < adoption_threshold)).sum(axis=0)
    r0_vals = r0_like(prev_new_adopters, new_adopters)
    if communities is not None and cluster_penetration_enabled:
        penetration = cluster_penetration(adoption_mask, communities)
    else:
        penetration = np.zeros(beliefs.shape[1], dtype=np.float32)

    metrics = []
    for k in range(beliefs.shape[1]):
        metrics.append(
            {
                "day": day,
                "claim": k,
                "adoption_fraction": float(adoption_fraction[k]),
                "mean_belief": float(mean_belief[k]),
                "variance": float(variance[k]),
                "polarization": float(polarization[k]),
                "r0_like": float(r0_vals[k]),
                "cluster_penetration": float(penetration[k]),
                "trust_gov": float(trust["trust_gov"].mean()),
                "trust_church": float(trust["trust_church"].mean()),
                "trust_local_news": float(trust["trust_local_news"].mean()),
                "trust_national_news": float(trust["trust_national_news"].mean()),
                "trust_friends": float(trust["trust_friends"].mean()),
            }
        )
    return metrics, new_adopters


def compute_daily_metrics_torch(
    day: int,
    beliefs: torch.Tensor,
    prev_beliefs: torch.Tensor,
    trust: Dict[str, torch.Tensor],
    adoption_threshold: float,
    communities: np.ndarray | None,
    prev_new_adopters: torch.Tensor,
    cluster_penetration_enabled: bool,
) -> Tuple[List[Dict[str, float]], torch.Tensor]:
    adoption_mask = beliefs >= adoption_threshold
    adoption_fraction = adoption_mask.float().mean(dim=0)
    mean_belief = beliefs.mean(dim=0)
    variance = beliefs.var(dim=0, unbiased=False)
    extremes = ((beliefs < 0.2) | (beliefs > 0.8)).float().mean(dim=0)
    polarization = 0.5 * extremes + 0.5 * torch.clamp(4 * variance, 0, 1)
    new_adopters = ((beliefs >= adoption_threshold) & (prev_beliefs < adoption_threshold)).sum(dim=0)
    r0_vals = new_adopters / torch.clamp(prev_new_adopters, min=1)

    if communities is not None and cluster_penetration_enabled:
        adoption_mask_cpu = adoption_mask.detach().cpu().numpy()
        penetration = cluster_penetration(adoption_mask_cpu, communities)
    else:
        penetration = np.zeros(beliefs.shape[1], dtype=np.float32)

    metrics = []
    trust_means = {k: float(v.mean().item()) for k, v in trust.items()}
    for k in range(beliefs.shape[1]):
        metrics.append(
            {
                "day": day,
                "claim": k,
                "adoption_fraction": float(adoption_fraction[k].item()),
                "mean_belief": float(mean_belief[k].item()),
                "variance": float(variance[k].item()),
                "polarization": float(polarization[k].item()),
                "r0_like": float(r0_vals[k].item()),
                "cluster_penetration": float(penetration[k]),
                "trust_gov": trust_means["trust_gov"],
                "trust_church": trust_means["trust_church"],
                "trust_local_news": trust_means["trust_local_news"],
                "trust_national_news": trust_means["trust_national_news"],
                "trust_friends": trust_means["trust_friends"],
            }
        )
    return metrics, new_adopters
