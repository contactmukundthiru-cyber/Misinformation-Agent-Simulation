from __future__ import annotations

from typing import Tuple

import torch

from sim.config import BeliefUpdateConfig


def update_beliefs(
    beliefs: torch.Tensor,
    exposure: torch.Tensor,
    trust_signal: torch.Tensor,
    social_proof: torch.Tensor,
    debunk_pressure: torch.Tensor,
    skepticism: torch.Tensor,
    match: torch.Tensor,
    exposure_memory: torch.Tensor,
    baseline: torch.Tensor,
    cfg: BeliefUpdateConfig,
    reactance_enabled: bool,
    reactance: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized belief update with decay, repetition, and corrections."""
    exposure_memory = cfg.exposure_memory_decay * exposure_memory + (1 - cfg.exposure_memory_decay) * exposure

    p = torch.sigmoid(
        cfg.alpha * exposure_memory
        + cfg.beta * trust_signal
        + cfg.gamma * match
        + cfg.delta * social_proof
        - cfg.lambda_skepticism * skepticism.unsqueeze(1)
        - cfg.mu_debunk * debunk_pressure
    )

    correction = debunk_pressure
    if reactance_enabled:
        correction = correction * (1 - cfg.reactance_strength * reactance.unsqueeze(1))

    beliefs = beliefs + cfg.eta * p * (1 - beliefs) - cfg.rho * correction
    beliefs = beliefs + cfg.belief_decay * (baseline - beliefs)
    beliefs = torch.clamp(beliefs, 0.0, 1.0)

    return beliefs, exposure_memory
