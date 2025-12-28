from __future__ import annotations

from typing import List, Tuple

import torch

from sim.config import ModerationConfig, WorldConfig
from sim.disease.strains import Strain


def apply_moderation(
    share_probs: torch.Tensor,
    strains: List[Strain],
    world_cfg: WorldConfig,
    moderation_cfg: ModerationConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    risk = torch.tensor(
        [s.violation_risk * (1 - s.stealth) for s in strains],
        device=share_probs.device,
        dtype=share_probs.dtype,
    )
    strictness = world_cfg.moderation_strictness
    downrank = 1 - moderation_cfg.downrank_effect * strictness * risk
    warning = moderation_cfg.warning_effect * strictness * risk

    adjusted = share_probs * downrank
    return adjusted.clamp(0.0, 1.0), warning
