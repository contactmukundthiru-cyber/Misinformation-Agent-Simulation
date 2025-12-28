"""
Dual-Process Cognitive Architecture
====================================
Implements Kahneman's System 1 / System 2 theory for belief formation.

System 1 (Fast/Intuitive):
- Operates automatically with little effort
- Driven by emotional resonance, familiarity, narrative coherence
- Prone to cognitive biases but efficient
- Dominant when cognitive load is high

System 2 (Slow/Analytical):
- Requires conscious effort and attention
- Evaluates evidence quality, logical consistency, source credibility
- More accurate but resource-intensive
- Dominant when stakes are high and resources available

The model determines which system processes each piece of information
based on agent traits and situational factors.

References:
- Kahneman, D. (2011). Thinking, Fast and Slow
- Pennycook, G., & Rand, D. G. (2019). Lazy, not biased
- Bago, B., & De Neys, W. (2019). The Smart System 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
from pydantic import BaseModel


class DualProcessConfig(BaseModel):
    """Configuration for dual-process cognitive model."""

    # System 1 parameters
    s1_emotional_weight: float = 0.4
    s1_familiarity_weight: float = 0.3
    s1_narrative_weight: float = 0.3
    s1_novelty_penalty: float = 0.2
    s1_fluency_boost: float = 0.15

    # System 2 parameters
    s2_evidence_weight: float = 0.35
    s2_source_weight: float = 0.3
    s2_consistency_weight: float = 0.25
    s2_complexity_penalty: float = 0.1

    # Integration parameters
    base_s1_tendency: float = 0.6
    cognitive_load_s1_boost: float = 0.3
    stakes_s2_boost: float = 0.25
    need_for_cognition_s2_boost: float = 0.2

    # Thresholds
    deliberation_threshold: float = 0.4
    override_threshold: float = 0.7
    conflict_detection_threshold: float = 0.3

    # Temporal dynamics
    familiarity_decay: float = 0.95
    fluency_learning_rate: float = 0.1


@dataclass
class CognitiveState:
    """Tracks cognitive state for each agent across time."""

    n_agents: int
    n_claims: int
    device: torch.device

    # Processing tendency (0 = pure S2, 1 = pure S1)
    s1_tendency: torch.Tensor = field(init=False)

    # Current cognitive load (0 = fresh, 1 = exhausted)
    cognitive_load: torch.Tensor = field(init=False)

    # Need for cognition trait (stable personality factor)
    need_for_cognition: torch.Tensor = field(init=False)

    # Familiarity with each claim (builds with exposure)
    claim_familiarity: torch.Tensor = field(init=False)

    # Processing fluency for each claim (ease of comprehension)
    claim_fluency: torch.Tensor = field(init=False)

    # Perceived stakes for each claim (importance/relevance)
    perceived_stakes: torch.Tensor = field(init=False)

    # Conflict detection history (S1/S2 disagreements)
    conflict_history: torch.Tensor = field(init=False)

    # Override history (times S2 overrode S1)
    override_count: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.s1_tendency = torch.zeros(self.n_agents, device=self.device)
        self.cognitive_load = torch.zeros(self.n_agents, device=self.device)
        self.need_for_cognition = torch.zeros(self.n_agents, device=self.device)
        self.claim_familiarity = torch.zeros(
            (self.n_agents, self.n_claims), device=self.device
        )
        self.claim_fluency = torch.full(
            (self.n_agents, self.n_claims), 0.5, device=self.device
        )
        self.perceived_stakes = torch.full(
            (self.n_agents, self.n_claims), 0.3, device=self.device
        )
        self.conflict_history = torch.zeros(
            (self.n_agents, self.n_claims), device=self.device
        )
        self.override_count = torch.zeros(self.n_agents, device=self.device)


def initialize_cognitive_states(
    n_agents: int,
    n_claims: int,
    traits: Dict[str, torch.Tensor],
    cfg: DualProcessConfig,
    device: torch.device,
) -> CognitiveState:
    """
    Initialize cognitive states from agent traits.

    Maps personality and cognitive traits to dual-process parameters:
    - Need for cognition derived from numeracy and skepticism
    - S1 tendency from need_for_closure and time pressure
    """
    state = CognitiveState(n_agents=n_agents, n_claims=n_claims, device=device)

    # Need for cognition: composite of analytical traits
    # Higher numeracy and skepticism -> higher need for cognition
    numeracy = traits.get("numeracy", torch.full((n_agents,), 0.5, device=device))
    skepticism = traits.get("skepticism", torch.full((n_agents,), 0.5, device=device))

    state.need_for_cognition = torch.clamp(
        0.4 * numeracy + 0.4 * skepticism + 0.2 * torch.rand(n_agents, device=device),
        0.0, 1.0
    )

    # Base S1 tendency: influenced by need_for_closure
    need_for_closure = traits.get(
        "need_for_closure", torch.full((n_agents,), 0.5, device=device)
    )

    state.s1_tendency = torch.clamp(
        cfg.base_s1_tendency
        + 0.2 * need_for_closure
        - 0.15 * state.need_for_cognition,
        0.2, 0.9
    )

    # Initialize cognitive load to low random values
    state.cognitive_load = 0.1 + 0.2 * torch.rand(n_agents, device=device)

    return state


def compute_processing_mode(
    state: CognitiveState,
    exposure_intensity: torch.Tensor,
    identity_threat: torch.Tensor,
    cfg: DualProcessConfig,
) -> torch.Tensor:
    """
    Compute the effective S1/S2 balance for current processing.

    Returns tensor of shape (n_agents, n_claims) with values in [0, 1]
    where 0 = pure System 2 and 1 = pure System 1.

    Factors increasing S1 dominance:
    - High cognitive load
    - Low stakes
    - High familiarity (fluent processing)

    Factors increasing S2 dominance:
    - High stakes (identity threat)
    - Low familiarity (disfluent processing triggers deliberation)
    - High need for cognition
    - Detected conflict between intuition and analysis
    """
    # Base tendency expanded to (n_agents, n_claims)
    s1_weight = state.s1_tendency.unsqueeze(1).expand(-1, state.n_claims).clone()

    # Cognitive load increases S1 reliance
    load_effect = cfg.cognitive_load_s1_boost * state.cognitive_load.unsqueeze(1)
    s1_weight = s1_weight + load_effect

    # High stakes (identity threat) triggers S2
    stakes_effect = cfg.stakes_s2_boost * identity_threat
    s1_weight = s1_weight - stakes_effect

    # Need for cognition promotes S2
    nfc_effect = cfg.need_for_cognition_s2_boost * state.need_for_cognition.unsqueeze(1)
    s1_weight = s1_weight - nfc_effect

    # Low familiarity (novelty) triggers deliberation
    novelty = 1.0 - state.claim_familiarity
    novelty_trigger = (novelty > cfg.deliberation_threshold).float()
    s1_weight = s1_weight - 0.15 * novelty_trigger * novelty

    # Prior conflict detection increases S2 vigilance
    conflict_vigilance = 0.1 * state.conflict_history
    s1_weight = s1_weight - conflict_vigilance

    return torch.clamp(s1_weight, 0.1, 0.95)


def system1_evaluation(
    exposure: torch.Tensor,
    emotional_resonance: torch.Tensor,
    familiarity: torch.Tensor,
    narrative_fit: torch.Tensor,
    cfg: DualProcessConfig,
) -> torch.Tensor:
    """
    System 1 evaluation: fast, intuitive, emotion-driven.

    Inputs shape: (n_agents, n_claims)

    S1 accepts information that:
    - Resonates emotionally (fear, anger, hope alignment)
    - Feels familiar (prior exposure builds fluency)
    - Fits existing narrative schemas

    S1 rejects information that:
    - Feels unfamiliar/novel (processing disfluency)
    - Contradicts strong existing beliefs
    """
    # Processing fluency from familiarity
    fluency_boost = cfg.s1_fluency_boost * familiarity

    # Novelty penalty for unfamiliar claims
    novelty_penalty = cfg.s1_novelty_penalty * (1.0 - familiarity)

    # Combine S1 signals
    s1_signal = (
        cfg.s1_emotional_weight * emotional_resonance
        + cfg.s1_familiarity_weight * familiarity
        + cfg.s1_narrative_weight * narrative_fit
        + fluency_boost
        - novelty_penalty
    )

    # S1 outputs intuitive acceptance probability
    return torch.sigmoid(s1_signal)


def system2_evaluation(
    exposure: torch.Tensor,
    evidence_quality: torch.Tensor,
    source_credibility: torch.Tensor,
    logical_consistency: torch.Tensor,
    complexity: torch.Tensor,
    cfg: DualProcessConfig,
) -> torch.Tensor:
    """
    System 2 evaluation: slow, analytical, evidence-driven.

    Inputs shape: (n_agents, n_claims)

    S2 accepts information based on:
    - Quality of evidence presented
    - Credibility of source
    - Logical consistency with known facts

    S2 is hindered by:
    - High complexity (limited working memory)
    - Time pressure (not modeled here, affects processing mode)
    """
    # Complexity penalty for difficult-to-evaluate claims
    complexity_penalty = cfg.s2_complexity_penalty * complexity

    # Combine S2 signals
    s2_signal = (
        cfg.s2_evidence_weight * evidence_quality
        + cfg.s2_source_weight * source_credibility
        + cfg.s2_consistency_weight * logical_consistency
        - complexity_penalty
    )

    # S2 outputs analytical acceptance probability
    return torch.sigmoid(s2_signal)


def integrate_dual_process(
    s1_output: torch.Tensor,
    s2_output: torch.Tensor,
    processing_mode: torch.Tensor,
    state: CognitiveState,
    cfg: DualProcessConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate System 1 and System 2 outputs.

    The integration follows the "default-interventionist" model:
    1. S1 generates intuitive response
    2. S2 may override if conflict is detected and resources available

    Returns:
        integrated_output: Final acceptance probability (n_agents, n_claims)
        conflict_detected: Boolean mask of S1/S2 conflicts (n_agents, n_claims)
    """
    # Detect conflict between S1 and S2
    s1_s2_diff = torch.abs(s1_output - s2_output)
    conflict_detected = s1_s2_diff > cfg.conflict_detection_threshold

    # Basic weighted integration
    s1_weight = processing_mode
    integrated = s1_weight * s1_output + (1.0 - s1_weight) * s2_output

    # S2 override when conflict detected and S2 is strong
    override_mask = conflict_detected & (s2_output > cfg.override_threshold)
    s2_override_mask = override_mask & (processing_mode < 0.7)

    # Where S2 overrides, use S2 output directly
    integrated = torch.where(s2_override_mask, s2_output, integrated)

    # Update conflict history (exponential moving average)
    state.conflict_history = (
        0.9 * state.conflict_history + 0.1 * conflict_detected.float()
    )

    # Update override count
    override_count_delta = s2_override_mask.float().sum(dim=1)
    state.override_count = state.override_count + override_count_delta

    return integrated, conflict_detected


def update_familiarity(
    state: CognitiveState,
    exposure: torch.Tensor,
    cfg: DualProcessConfig,
) -> None:
    """
    Update claim familiarity based on exposure.

    Familiarity increases with exposure but decays over time.
    High familiarity leads to more fluent processing (more S1).
    """
    # Exposure increases familiarity
    exposure_normalized = torch.clamp(exposure / (exposure.max() + 1e-6), 0, 1)
    familiarity_gain = cfg.fluency_learning_rate * exposure_normalized

    # Apply decay and add new familiarity
    state.claim_familiarity = (
        cfg.familiarity_decay * state.claim_familiarity + familiarity_gain
    )
    state.claim_familiarity = torch.clamp(state.claim_familiarity, 0.0, 1.0)


def update_cognitive_load(
    state: CognitiveState,
    processing_intensity: torch.Tensor,
    rest_factor: float = 0.1,
) -> None:
    """
    Update cognitive load based on processing demands.

    High processing intensity increases load, which decays with rest.
    High load shifts processing toward System 1.
    """
    # Processing intensity is mean across claims for each agent
    intensity_per_agent = processing_intensity.mean(dim=1)

    # Load increases with intensity, decays with rest
    load_increase = 0.1 * intensity_per_agent
    load_decay = rest_factor * state.cognitive_load

    state.cognitive_load = torch.clamp(
        state.cognitive_load + load_increase - load_decay,
        0.0, 1.0
    )
