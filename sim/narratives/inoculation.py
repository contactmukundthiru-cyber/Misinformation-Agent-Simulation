"""
Inoculation Theory Implementation
==================================
Models how pre-exposure to weakened misinformation builds resistance.

Based on:
- McGuire's inoculation theory (1961)
- van der Linden's psychological inoculation (2017)
- Prebunking interventions (Roozenbeek & van der Linden, 2019)

Key mechanisms:
1. Forewarning: Alerting people that manipulation may occur
2. Weakened exposure: Presenting weakened form of argument
3. Refutational preemption: Teaching counter-arguments
4. Active refutation: Generating own counter-arguments
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from pydantic import BaseModel


class InoculationConfig(BaseModel):
    """Configuration for inoculation mechanics."""

    # Inoculation effectiveness
    inoculation_strength: float = 0.6
    inoculation_decay_rate: float = 0.02

    # Forewarning effect
    forewarning_strength: float = 0.3
    forewarning_duration: int = 30  # Days of protection

    # Prebunking intervention
    prebunking_reach: float = 0.3
    prebunking_day: Optional[int] = None

    # Active inoculation (game-based)
    active_inoculation_multiplier: float = 1.5

    # Transfer effects
    cross_claim_transfer: float = 0.3  # Inoculation against one helps against similar


@dataclass
class InoculationState:
    """Tracks inoculation status for agents."""

    n_agents: int
    n_claims: int
    device: torch.device

    # Inoculation level per agent per claim (0 = none, 1 = fully inoculated)
    inoculation_level: torch.Tensor

    # Days since inoculation (for decay)
    days_since_inoculation: torch.Tensor

    # Type of inoculation received
    # 0 = none, 1 = passive (exposure), 2 = active (refutation training)
    inoculation_type: torch.Tensor

    # Forewarning status
    forewarned: torch.Tensor

    # Antibody-like resistance developed from fighting off exposure
    natural_resistance: torch.Tensor

    @classmethod
    def initialize(
        cls,
        n_agents: int,
        n_claims: int,
        device: torch.device,
    ) -> "InoculationState":
        return cls(
            n_agents=n_agents,
            n_claims=n_claims,
            device=device,
            inoculation_level=torch.zeros(n_agents, n_claims, device=device),
            days_since_inoculation=torch.zeros(n_agents, n_claims, device=device),
            inoculation_type=torch.zeros(n_agents, n_claims, dtype=torch.long, device=device),
            forewarned=torch.zeros(n_agents, dtype=torch.bool, device=device),
            natural_resistance=torch.zeros(n_agents, n_claims, device=device),
        )


def initialize_inoculation_state(
    n_agents: int,
    n_claims: int,
    device: torch.device,
) -> InoculationState:
    """Initialize inoculation tracking."""
    return InoculationState.initialize(n_agents, n_claims, device)


def apply_prebunking(
    state: InoculationState,
    target_claims: List[int],
    target_fraction: float,
    inoculation_type: int,
    cfg: InoculationConfig,
    rng: torch.Generator,
) -> int:
    """
    Apply prebunking intervention to subset of population.

    Returns number of agents inoculated.
    """
    n_to_inoculate = int(state.n_agents * target_fraction)

    # Select random agents
    indices = torch.randperm(state.n_agents, generator=rng, device=state.device)[:n_to_inoculate]

    for claim in target_claims:
        # Set inoculation level
        base_level = cfg.inoculation_strength
        if inoculation_type == 2:  # Active inoculation
            base_level *= cfg.active_inoculation_multiplier

        state.inoculation_level[indices, claim] = torch.clamp(
            state.inoculation_level[indices, claim] + base_level,
            0.0, 1.0
        )

        state.days_since_inoculation[indices, claim] = 0
        state.inoculation_type[indices, claim] = inoculation_type

    return n_to_inoculate


def apply_forewarning(
    state: InoculationState,
    target_fraction: float,
    rng: torch.Generator,
) -> int:
    """
    Apply forewarning to subset of population.

    Forewarning alerts people that they may be exposed to manipulation,
    increasing their resistance even without specific inoculation.
    """
    n_to_warn = int(state.n_agents * target_fraction)
    indices = torch.randperm(state.n_agents, generator=rng, device=state.device)[:n_to_warn]

    state.forewarned[indices] = True

    return n_to_warn


def compute_inoculation_resistance(
    state: InoculationState,
    exposure: torch.Tensor,
    cfg: InoculationConfig,
) -> torch.Tensor:
    """
    Compute resistance to misinformation from inoculation.

    Returns resistance factor (0 = no resistance, 1 = full resistance).
    Shape: (n_agents, n_claims)
    """
    # Base resistance from inoculation level
    resistance = state.inoculation_level.clone()

    # Active inoculation is more effective
    active_mask = state.inoculation_type == 2
    resistance = torch.where(
        active_mask,
        resistance * cfg.active_inoculation_multiplier,
        resistance
    )

    # Forewarning adds baseline resistance
    forewarned_expanded = state.forewarned.unsqueeze(1).expand(-1, state.n_claims)
    forewarning_bonus = forewarned_expanded.float() * cfg.forewarning_strength
    resistance = resistance + forewarning_bonus

    # Natural resistance from surviving exposure
    resistance = resistance + state.natural_resistance

    # Cross-claim transfer: inoculation against similar claims helps
    mean_inoculation = state.inoculation_level.mean(dim=1, keepdim=True)
    transfer_resistance = cfg.cross_claim_transfer * mean_inoculation
    resistance = resistance + transfer_resistance

    return torch.clamp(resistance, 0.0, 0.95)


def update_inoculation_decay(
    state: InoculationState,
    cfg: InoculationConfig,
) -> None:
    """
    Update inoculation state with time decay.

    Inoculation effectiveness decays over time unless reinforced.
    """
    # Increment days since inoculation
    inoculated_mask = state.inoculation_level > 0.01
    state.days_since_inoculation = torch.where(
        inoculated_mask,
        state.days_since_inoculation + 1,
        state.days_since_inoculation
    )

    # Decay inoculation level
    decay = cfg.inoculation_decay_rate
    state.inoculation_level = state.inoculation_level * (1 - decay)

    # Forewarning wears off after duration
    # (This is a simplification - in reality would track per-agent timing)
    state.forewarned = state.forewarned & (
        torch.rand(state.n_agents, device=state.device) > 0.01
    )


def build_natural_resistance(
    state: InoculationState,
    beliefs: torch.Tensor,
    prev_beliefs: torch.Tensor,
    exposure: torch.Tensor,
) -> None:
    """
    Build natural resistance from successfully resisting exposure.

    When an agent is exposed but doesn't increase belief,
    they build some natural resistance.
    """
    # Exposed but didn't increase belief
    exposed = exposure > 0.1
    resisted = beliefs <= prev_beliefs

    resistance_gain = exposed & resisted

    # Build resistance gradually
    gain_amount = 0.02 * resistance_gain.float()
    state.natural_resistance = torch.clamp(
        state.natural_resistance + gain_amount,
        0.0, 0.5
    )


def compute_inoculation_spillover(
    state: InoculationState,
    claim_topics: List[str],
) -> torch.Tensor:
    """
    Compute spillover effects between similar claims.

    Inoculation against one claim type helps against similar types.
    """
    # Build topic similarity matrix
    n_claims = len(claim_topics)
    similarity = torch.zeros(n_claims, n_claims, device=state.device)

    topic_groups = {
        "conspiracy": ["tech_conspiracy", "health_rumor"],
        "threat": ["outsider_threat", "economic_panic"],
        "moral": ["moral_spiral"],
    }

    for i, topic_i in enumerate(claim_topics):
        for j, topic_j in enumerate(claim_topics):
            if i != j:
                for group, members in topic_groups.items():
                    if topic_i in members and topic_j in members:
                        similarity[i, j] = 0.5

    # Spillover: inoculation against i helps against j
    spillover = torch.matmul(state.inoculation_level, similarity.T)

    return spillover * 0.3  # Scale down spillover effect
