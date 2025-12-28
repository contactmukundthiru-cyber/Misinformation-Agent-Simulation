from __future__ import annotations

from typing import Dict

import torch

from sim.config import WorldConfig


def update_trust(
    trust: Dict[str, torch.Tensor],
    belief: torch.Tensor,
    debunk_pressure: torch.Tensor,
    world_cfg: WorldConfig,
) -> Dict[str, torch.Tensor]:
    if not world_cfg.trust_update_enabled:
        return trust
    erosion = world_cfg.trust_erosion_rate * debunk_pressure.mean(dim=1)
    adjust = (belief.mean(dim=1) - 0.5).clamp(min=0)
    trust["trust_gov"] = (trust["trust_gov"] - erosion * adjust).clamp(0.0, 1.0)
    trust["trust_local_news"] = (trust["trust_local_news"] - erosion * adjust).clamp(0.0, 1.0)
    trust["trust_national_news"] = (trust["trust_national_news"] - erosion * adjust).clamp(0.0, 1.0)
    return trust
