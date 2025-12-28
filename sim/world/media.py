from __future__ import annotations

from typing import List

import torch

from sim.config import WorldConfig
from sim.disease.strains import Strain


def feed_injection(
    media_diet: torch.Tensor,
    strains: List[Strain],
    world_cfg: WorldConfig,
) -> torch.Tensor:
    device = media_diet.device
    dtype = media_diet.dtype
    n_agents = media_diet.shape[0]
    n_claims = len(strains)

    local_social = media_diet[:, 0]
    national_social = media_diet[:, 1]
    base = world_cfg.feed_injection_rate * (local_social + national_social)

    injections = torch.zeros((n_agents, n_claims), device=device, dtype=dtype)
    for k, strain in enumerate(strains):
        outrage = strain.emotional_profile.get("anger", 0.0)
        injections[:, k] = base * (1 + world_cfg.algorithmic_amplification + world_cfg.outrage_amplification * outrage)
    return injections
