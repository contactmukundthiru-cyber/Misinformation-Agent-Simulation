from __future__ import annotations

from typing import Dict, List

import torch

from sim.config import SharingConfig, WorldConfig
from sim.disease.strains import Strain


def compute_share_probabilities(
    beliefs: torch.Tensor,
    traits: Dict[str, torch.Tensor],
    emotions: Dict[str, torch.Tensor],
    sharing_cfg: SharingConfig,
    world_cfg: WorldConfig,
    strains: List[Strain],
) -> torch.Tensor:
    """Compute per-agent share probabilities for each claim."""
    base = torch.full_like(beliefs, fill_value=sharing_cfg.base_share_rate)
    logit = torch.log(base / (1 - base))

    logit = logit + sharing_cfg.belief_sensitivity * (beliefs - 0.5)
    logit = logit + sharing_cfg.status_sensitivity * traits["status_seeking"].unsqueeze(1)
    logit = logit + sharing_cfg.conformity_sensitivity * traits["conformity"].unsqueeze(1)

    if emotions:
        fear = emotions["fear"].unsqueeze(1)
        anger = emotions["anger"].unsqueeze(1)
        hope = emotions["hope"].unsqueeze(1)
        weights = torch.tensor(
            [[s.emotional_profile.get("fear", 0.0),
              s.emotional_profile.get("anger", 0.0),
              s.emotional_profile.get("hope", 0.0)] for s in strains],
            device=beliefs.device,
            dtype=beliefs.dtype,
        )
        emotion_score = fear * weights[:, 0] + anger * weights[:, 1] + hope * weights[:, 2]
        logit = logit + sharing_cfg.emotion_sensitivity * emotion_score

    violation = torch.tensor([s.violation_risk for s in strains], device=beliefs.device, dtype=beliefs.dtype)
    moderation_penalty = sharing_cfg.moderation_risk_sensitivity * violation * world_cfg.moderation_strictness
    logit = logit - moderation_penalty
    logit = logit - world_cfg.platform_friction

    return torch.sigmoid(logit)
