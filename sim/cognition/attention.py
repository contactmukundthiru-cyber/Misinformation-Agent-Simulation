"""
Attention Economics and Cognitive Capacity
===========================================
Models the limited attention capacity of human cognition.

Key concepts:
1. Attention Budget: Agents have finite attention per timestep
2. Salience: Some stimuli capture attention more than others
3. Cognitive Fatigue: Attention capacity decreases with use
4. Selective Exposure: Agents preferentially attend to chosen content
5. Attention Capture: Emotional/novel content captures attention

References:
- Kahneman, D. (1973). Attention and Effort
- Falkinger, J. (2008). Limited attention as a scarce resource
- Wu, T. (2017). The Attention Merchants
- Marwick, A. (2018). Why do people share fake news?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from pydantic import BaseModel


class AttentionConfig(BaseModel):
    """Configuration for attention allocation model."""

    # Base attention parameters
    base_attention_budget: float = 1.0
    min_attention_per_item: float = 0.05
    max_attention_per_item: float = 0.5

    # Salience factors
    emotional_salience_weight: float = 0.35
    novelty_salience_weight: float = 0.25
    social_salience_weight: float = 0.25
    personal_relevance_weight: float = 0.15

    # Fatigue dynamics
    fatigue_rate: float = 0.15
    recovery_rate: float = 0.08
    fatigue_attention_penalty: float = 0.4

    # Selective exposure
    preference_alignment_bonus: float = 0.3
    avoidance_threshold: float = 0.7
    avoidance_strength: float = 0.5

    # Attention thresholds (lowered to allow more processing)
    processing_threshold: float = 0.02  # Was 0.15 - too aggressive
    deep_processing_threshold: float = 0.15  # Was 0.4


@dataclass
class AttentionState:
    """Tracks attention-related state for agents."""

    n_agents: int
    n_claims: int
    device: torch.device

    # Current attention budget (0 = depleted, 1 = full)
    attention_budget: torch.Tensor

    # Fatigue level (builds up, reduces effective attention)
    fatigue: torch.Tensor

    # Attention allocated to each claim in current timestep
    current_allocation: torch.Tensor

    # Cumulative attention given to each claim
    cumulative_attention: torch.Tensor

    # Attention preference (learned from past engagement)
    attention_preferences: torch.Tensor

    @classmethod
    def initialize(
        cls,
        n_agents: int,
        n_claims: int,
        device: torch.device,
    ) -> "AttentionState":
        return cls(
            n_agents=n_agents,
            n_claims=n_claims,
            device=device,
            attention_budget=torch.ones(n_agents, device=device),
            fatigue=torch.zeros(n_agents, device=device),
            current_allocation=torch.zeros(n_agents, n_claims, device=device),
            cumulative_attention=torch.zeros(n_agents, n_claims, device=device),
            attention_preferences=torch.full(
                (n_agents, n_claims), 0.5, device=device
            ),
        )


def compute_salience(
    exposure: torch.Tensor,
    emotional_intensity: torch.Tensor,
    familiarity: torch.Tensor,
    social_signals: torch.Tensor,
    personal_relevance: torch.Tensor,
    cfg: AttentionConfig,
) -> torch.Tensor:
    """
    Compute salience of each claim for each agent.

    Salience determines how much attention a claim captures.
    High salience = likely to be noticed and processed.

    Shape: (n_agents, n_claims)
    """
    # Novelty is inverse of familiarity (novel things grab attention)
    novelty = 1.0 - familiarity

    # Emotional content captures attention
    emotional_salience = cfg.emotional_salience_weight * emotional_intensity

    # Novel content captures attention
    novelty_salience = cfg.novelty_salience_weight * novelty

    # Social validation increases salience
    social_salience = cfg.social_salience_weight * social_signals

    # Personally relevant content gets attention
    relevance_salience = cfg.personal_relevance_weight * personal_relevance

    # Combine with exposure as baseline
    salience = (
        exposure * (
            0.3  # Base exposure effect
            + emotional_salience
            + novelty_salience
            + social_salience
            + relevance_salience
        )
    )

    return torch.clamp(salience, 0.0, 1.0)


def allocate_attention(
    salience: torch.Tensor,
    state: AttentionState,
    cfg: AttentionConfig,
) -> torch.Tensor:
    """
    Allocate limited attention budget across claims.

    Uses a softmax-like allocation where high-salience items
    receive more attention, but total is constrained by budget.

    Returns: attention allocation (n_agents, n_claims)
    """
    # Effective budget reduced by fatigue
    effective_budget = state.attention_budget * (
        1.0 - cfg.fatigue_attention_penalty * state.fatigue
    )

    # Softmax to allocate proportionally to salience
    # Temperature controls how sharply attention focuses on high-salience items
    temperature = 0.5
    attention_weights = torch.softmax(salience / temperature, dim=1)

    # Scale by effective budget
    attention_allocation = attention_weights * effective_budget.unsqueeze(1)

    # Enforce min/max per item
    attention_allocation = torch.clamp(
        attention_allocation,
        cfg.min_attention_per_item,
        cfg.max_attention_per_item,
    )

    # Renormalize to not exceed budget
    total_allocated = attention_allocation.sum(dim=1, keepdim=True)
    scale_factor = effective_budget.unsqueeze(1) / (total_allocated + 1e-6)
    scale_factor = torch.clamp(scale_factor, 0.0, 1.0)
    attention_allocation = attention_allocation * scale_factor

    # Update state
    state.current_allocation = attention_allocation
    state.cumulative_attention = state.cumulative_attention + attention_allocation

    return attention_allocation


def compute_cognitive_load(
    attention_allocation: torch.Tensor,
    processing_intensity: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cognitive load from attention and processing demands.

    Returns per-agent cognitive load (n_agents,)
    """
    # Load is product of attention given and intensity of processing
    load_per_claim = attention_allocation * processing_intensity
    total_load = load_per_claim.sum(dim=1)

    return torch.clamp(total_load, 0.0, 1.0)


def filter_by_attention(
    exposure: torch.Tensor,
    attention_allocation: torch.Tensor,
    cfg: AttentionConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter exposure by attention - unattended items not processed.

    Key insight: Attention affects PROCESSING DEPTH, not whether exposure occurs.
    People are exposed whether or not they pay full attention - attention
    determines how deeply they process/evaluate the claim.

    Returns:
        attended_exposure: Exposure modulated by attention (not multiplied!)
        processing_depth: How deeply each item is processed (affects S1 vs S2)
    """
    # Items below threshold get shallow processing but still process
    below_threshold = attention_allocation < cfg.processing_threshold
    processing_mask = (~below_threshold).float()

    # Attended exposure: use attention as a soft gate, not a multiplier
    # This preserves most of the exposure signal while attention modulates depth
    attention_factor = 0.3 + 0.7 * attention_allocation  # Range: [0.3, 1.0]
    attended_exposure = exposure * attention_factor * (0.5 + 0.5 * processing_mask)

    # Processing depth: determines S1 vs S2 balance and learning rate
    depth = torch.clamp(
        (attention_allocation - cfg.processing_threshold)
        / (cfg.deep_processing_threshold - cfg.processing_threshold + 1e-6),
        0.0, 1.0
    )
    # Even unattended items get some processing (System 1 still works)
    processing_depth = 0.2 + 0.8 * depth * processing_mask

    return attended_exposure, processing_depth


def update_attention_fatigue(
    state: AttentionState,
    attention_used: torch.Tensor,
    cfg: AttentionConfig,
) -> None:
    """
    Update fatigue and attention budget after processing.

    Fatigue builds up with attention use and recovers slowly.
    """
    # Total attention used this timestep
    total_used = attention_used.sum(dim=1)

    # Fatigue increases with use
    fatigue_increase = cfg.fatigue_rate * total_used

    # Fatigue recovers
    fatigue_decrease = cfg.recovery_rate * state.fatigue

    # Update fatigue
    state.fatigue = torch.clamp(
        state.fatigue + fatigue_increase - fatigue_decrease,
        0.0, 1.0
    )

    # Attention budget replenishes but is reduced by fatigue
    base_recovery = 0.3  # Recover 30% per timestep
    fatigue_penalty = 0.2 * state.fatigue

    state.attention_budget = torch.clamp(
        state.attention_budget - total_used + base_recovery - fatigue_penalty,
        0.1, 1.0
    )


def apply_selective_exposure(
    exposure_opportunities: torch.Tensor,
    beliefs: torch.Tensor,
    identity_threat: torch.Tensor,
    state: AttentionState,
    cfg: AttentionConfig,
) -> torch.Tensor:
    """
    Apply selective exposure - agents avoid threatening content.

    Agents preferentially engage with belief-consistent content
    and avoid identity-threatening content (within their control).

    This models voluntary exposure decisions, not forced exposure.
    """
    # Preference for belief-consistent content
    preference_boost = cfg.preference_alignment_bonus * beliefs

    # Avoidance of identity-threatening content
    high_threat = identity_threat > cfg.avoidance_threshold
    avoidance_penalty = high_threat.float() * cfg.avoidance_strength

    # Combine with base exposure opportunities
    selective_exposure = exposure_opportunities * (
        1.0 + preference_boost - avoidance_penalty
    )

    # Update attention preferences based on engagement
    preference_update = 0.1 * (selective_exposure - state.attention_preferences)
    state.attention_preferences = torch.clamp(
        state.attention_preferences + preference_update,
        0.1, 0.9
    )

    return torch.clamp(selective_exposure, 0.0, exposure_opportunities.max())


def compute_attention_capture(
    exposure: torch.Tensor,
    emotional_profile: torch.Tensor,
    state: AttentionState,
) -> torch.Tensor:
    """
    Compute involuntary attention capture by stimuli.

    Some content (highly emotional, surprising) captures attention
    regardless of agent preferences. This is distinct from
    voluntary attention allocation.

    Returns: attention capture magnitude (n_agents, n_claims)
    """
    # Emotional intensity drives capture
    emotion_magnitude = emotional_profile.abs().mean(dim=-1) if emotional_profile.dim() > 2 else emotional_profile

    # Surprise (low cumulative attention = novel = surprising)
    surprise = 1.0 - torch.tanh(state.cumulative_attention)

    # Capture probability
    capture_prob = 0.3 * emotion_magnitude + 0.2 * surprise

    # Actual capture (probabilistic)
    capture = exposure * capture_prob

    return torch.clamp(capture, 0.0, 0.5)
