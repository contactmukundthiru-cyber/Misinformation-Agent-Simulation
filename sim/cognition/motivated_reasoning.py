"""
Motivated Reasoning and Identity-Protective Cognition
======================================================
Implements psychological mechanisms that bias information processing
based on identity and prior beliefs.

Key mechanisms:
1. Identity-Protective Cognition: Reject information threatening self-concept
2. Confirmation Bias: Preferentially accept belief-consistent information
3. Disconfirmation Bias: Scrutinize belief-inconsistent information more
4. Defensive Processing: Active counter-arguing against threatening info
5. Reactance: Backlash against perceived manipulation

References:
- Kahan, D. M. (2013). Ideology, motivated reasoning, and cognitive reflection
- Kunda, Z. (1990). The case for motivated reasoning
- Nyhan, B., & Reifler, J. (2010). When corrections fail (backfire effect)
- Taber, C. S., & Lodge, M. (2006). Motivated skepticism
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from pydantic import BaseModel


class IdentityConfig(BaseModel):
    """Configuration for identity and motivated reasoning."""

    # Identity dimensions and their weights
    n_identity_dimensions: int = 5
    identity_importance_mean: float = 0.5
    identity_importance_std: float = 0.2

    # Threat response parameters (reduced to allow realistic spread)
    threat_sensitivity: float = 0.4  # Was 0.6 - too sensitive
    defensive_threshold: float = 0.5  # Was 0.4 - raised to reduce defensive processing
    defensive_strength: float = 0.3  # Was 0.5 - reduced

    # Confirmation bias (these help spread, so keep moderate)
    confirmation_strength: float = 0.4
    disconfirmation_scrutiny: float = 0.4  # Was 0.6 - reduced

    # Reactance (reduced to allow debunking to work)
    reactance_threshold: float = 0.7  # Was 0.6 - raised
    reactance_strength: float = 0.25  # Was 0.35 - reduced
    reactance_decay: float = 0.9

    # Identity updating
    identity_plasticity: float = 0.02
    identity_crystallization_rate: float = 0.01


class IdentityState:
    """Tracks identity-related state for agents."""

    def __init__(
        self,
        n_agents: int,
        n_claims: int,
        n_identity_dims: int,
        device: torch.device,
    ):
        self.n_agents = n_agents
        self.n_claims = n_claims
        self.n_identity_dims = n_identity_dims
        self.device = device

        # Agent identity positions on each dimension (political, religious, etc.)
        # Shape: (n_agents, n_identity_dims)
        self.identity_positions = torch.rand(
            n_agents, n_identity_dims, device=device
        )

        # Importance of each identity dimension to each agent
        # Shape: (n_agents, n_identity_dims)
        self.identity_importance = torch.rand(
            n_agents, n_identity_dims, device=device
        )

        # How crystallized (stable) each identity dimension is
        # Higher = more resistant to change
        # Shape: (n_agents, n_identity_dims)
        self.identity_crystallization = 0.3 + 0.4 * torch.rand(
            n_agents, n_identity_dims, device=device
        )

        # Current identity salience (activated by threats)
        # Shape: (n_agents, n_identity_dims)
        self.identity_salience = torch.zeros(
            n_agents, n_identity_dims, device=device
        )

        # Accumulated reactance state
        # Shape: (n_agents,)
        self.reactance_level = torch.zeros(n_agents, device=device)

        # Claim-identity mapping: which claims threaten which identities
        # Shape: (n_claims, n_identity_dims)
        self.claim_identity_relevance = torch.zeros(
            n_claims, n_identity_dims, device=device
        )


def initialize_identity_state(
    n_agents: int,
    n_claims: int,
    traits: Dict[str, torch.Tensor],
    claim_topics: list,
    cfg: IdentityConfig,
    device: torch.device,
) -> IdentityState:
    """
    Initialize identity states from agent traits and claim properties.

    Identity dimensions:
    0: Political ideology (liberal/conservative)
    1: Religious identity
    2: Scientific worldview
    3: National/cultural identity
    4: Social group identity
    """
    state = IdentityState(
        n_agents=n_agents,
        n_claims=n_claims,
        n_identity_dims=cfg.n_identity_dimensions,
        device=device,
    )

    # Political identity from ideology proxy
    ideology = traits.get("ideology", torch.rand(n_agents, device=device))
    state.identity_positions[:, 0] = ideology

    # Religious identity from church trust
    trust_church = traits.get(
        "trust_church", torch.rand(n_agents, device=device)
    )
    state.identity_positions[:, 1] = trust_church

    # Scientific worldview from numeracy and skepticism
    numeracy = traits.get("numeracy", torch.rand(n_agents, device=device))
    skepticism = traits.get("skepticism", torch.rand(n_agents, device=device))
    state.identity_positions[:, 2] = 0.5 * numeracy + 0.5 * skepticism

    # National identity from outgroup trust (inverse)
    trust_outgroups = traits.get(
        "trust_outgroups", torch.rand(n_agents, device=device)
    )
    state.identity_positions[:, 3] = 1.0 - trust_outgroups

    # Social conformity as identity dimension
    conformity = traits.get("conformity", torch.rand(n_agents, device=device))
    state.identity_positions[:, 4] = conformity

    # Set identity importance based on personality
    state.identity_importance = torch.clamp(
        torch.randn(n_agents, cfg.n_identity_dimensions, device=device)
        * cfg.identity_importance_std + cfg.identity_importance_mean,
        0.1, 0.9
    )

    # Map claims to identity dimensions based on topics
    topic_identity_map = {
        "health_rumor": torch.tensor([0.2, 0.3, 0.8, 0.2, 0.3], device=device),
        "economic_panic": torch.tensor([0.7, 0.2, 0.4, 0.5, 0.3], device=device),
        "moral_spiral": torch.tensor([0.5, 0.9, 0.3, 0.4, 0.6], device=device),
        "tech_conspiracy": torch.tensor([0.4, 0.3, 0.9, 0.3, 0.4], device=device),
        "outsider_threat": torch.tensor([0.8, 0.5, 0.3, 0.9, 0.5], device=device),
    }

    for k, topic in enumerate(claim_topics):
        if topic in topic_identity_map:
            state.claim_identity_relevance[k] = topic_identity_map[topic]
        else:
            state.claim_identity_relevance[k] = torch.full(
                (cfg.n_identity_dimensions,), 0.3, device=device
            )

    return state


def compute_identity_threat(
    beliefs: torch.Tensor,
    claim_positions: torch.Tensor,
    state: IdentityState,
    cfg: IdentityConfig,
) -> torch.Tensor:
    """
    Compute identity threat level for each agent-claim pair.

    A claim threatens identity when:
    1. The claim is relevant to an identity dimension
    2. Accepting the claim would move agent away from identity position
    3. The identity dimension is important and crystallized

    Returns shape: (n_agents, n_claims)
    """
    # claim_positions: shape (n_claims,) - where claim "sits" on identity scale

    # Compute distance between claim position and agent identity
    # For each identity dimension, how far is the claim from agent's position?
    # Shape: (n_agents, n_claims, n_identity_dims)
    claim_pos_expanded = claim_positions.unsqueeze(0).unsqueeze(2)
    identity_expanded = state.identity_positions.unsqueeze(1)

    # Distance on each identity dimension
    distance = torch.abs(claim_pos_expanded - identity_expanded)

    # Weight by identity importance, crystallization, and claim relevance
    importance = state.identity_importance.unsqueeze(1)
    crystallization = state.identity_crystallization.unsqueeze(1)
    relevance = state.claim_identity_relevance.unsqueeze(0)

    # Threat is high when: distance is high, importance is high,
    # crystallization is high, and claim is relevant
    weighted_threat = (
        distance
        * importance
        * crystallization
        * relevance
        * cfg.threat_sensitivity
    )

    # Aggregate across identity dimensions (max threat matters most)
    identity_threat = weighted_threat.max(dim=2).values

    return torch.clamp(identity_threat, 0.0, 1.0)


def apply_confirmation_bias(
    acceptance_prob: torch.Tensor,
    beliefs: torch.Tensor,
    cfg: IdentityConfig,
) -> torch.Tensor:
    """
    Apply confirmation bias to acceptance probabilities.

    Agents more readily accept information consistent with existing beliefs
    and more readily reject information inconsistent with beliefs.

    For misinformation: if agent already believes, they're MORE likely
    to accept reinforcing information.

    Key insight: Confirmation bias should amplify acceptance for believers
    without penalizing non-believers too much. The "truth default" means
    people initially accept information unless they have reason to doubt.
    """
    # Confirmation effect: higher belief -> higher acceptance of confirming info
    confirmation_boost = cfg.confirmation_strength * beliefs

    # Disconfirmation effect: apply only for MODERATE believers (0.3-0.7)
    # who might scrutinize conflicting info more
    # Low believers (<0.3) don't have strong enough beliefs to scrutinize
    # High believers (>0.7) are too confirmed to scrutinize
    moderate_belief = (beliefs > 0.3) & (beliefs < 0.7)
    scrutiny_zone = moderate_belief.float() * torch.abs(0.5 - beliefs)
    disconfirmation_penalty = cfg.disconfirmation_scrutiny * scrutiny_zone

    # Net effect: minimal effect for very low/high believers
    # Moderate scrutiny only in the middle zone
    biased_prob = acceptance_prob * (1.0 + confirmation_boost - 0.3 * disconfirmation_penalty)

    return torch.clamp(biased_prob, 0.0, 1.0)


def compute_defensive_processing(
    identity_threat: torch.Tensor,
    exposure_intensity: torch.Tensor,
    state: IdentityState,
    cfg: IdentityConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute defensive processing response to identity threats.

    When identity is threatened, agents engage in:
    1. Counter-arguing: generating reasons to reject threatening info
    2. Source derogation: discounting source credibility
    3. Selective attention: avoiding threatening info

    Returns:
        rejection_boost: Added rejection probability (n_agents, n_claims)
        source_discount: Multiplier for source credibility (n_agents, n_claims)
    """
    # Defensive processing activates above threshold
    defensive_activation = torch.relu(
        identity_threat - cfg.defensive_threshold
    ) / (1.0 - cfg.defensive_threshold + 1e-6)

    # Counter-arguing increases rejection
    rejection_boost = cfg.defensive_strength * defensive_activation

    # Source derogation: discount credibility of threatening sources
    # 1.0 = full credibility, 0.0 = completely discounted
    source_discount = 1.0 - 0.5 * defensive_activation

    return rejection_boost, source_discount


def compute_reactance(
    correction_pressure: torch.Tensor,
    state: IdentityState,
    cfg: IdentityConfig,
) -> torch.Tensor:
    """
    Compute psychological reactance (backlash against correction).

    When agents feel their freedom to believe is threatened,
    they may reactively strengthen the threatened belief.

    This models the "backfire effect" observed in some contexts.
    """
    # Reactance triggers when correction pressure is high
    reactance_trigger = correction_pressure > cfg.reactance_threshold

    # Reactance magnitude depends on personality and current level
    reactance_response = (
        reactance_trigger.float()
        * cfg.reactance_strength
        * (1.0 + state.reactance_level.unsqueeze(1))
    )

    # Update reactance state (builds up, decays slowly)
    state.reactance_level = cfg.reactance_decay * state.reactance_level + \
        0.1 * reactance_response.mean(dim=1)

    return reactance_response


def update_identity_salience(
    state: IdentityState,
    identity_threat: torch.Tensor,
    decay_rate: float = 0.8,
) -> None:
    """
    Update identity salience based on recent threats.

    Threatened identity dimensions become more salient,
    increasing their influence on subsequent processing.
    """
    # Compute max threat to each identity dimension across claims
    # claim_identity_relevance: (n_claims, n_identity_dims)
    # identity_threat: (n_agents, n_claims)

    # For each agent, weight threat by relevance to each dimension
    relevance = state.claim_identity_relevance.unsqueeze(0)
    threat_expanded = identity_threat.unsqueeze(2)

    dimension_threat = (threat_expanded * relevance).max(dim=1).values

    # Update salience with decay
    state.identity_salience = (
        decay_rate * state.identity_salience
        + (1.0 - decay_rate) * dimension_threat
    )


def update_identity_positions(
    state: IdentityState,
    beliefs: torch.Tensor,
    claim_positions: torch.Tensor,
    cfg: IdentityConfig,
) -> None:
    """
    Slowly update identity positions based on accepted beliefs.

    This models gradual identity shift as people internalize
    beliefs that initially threatened their identity.

    Only high beliefs (accepted claims) can shift identity.
    Less crystallized identities shift more easily.
    """
    # Only strongly held beliefs shift identity
    strong_belief_mask = beliefs > 0.7

    # Compute desired identity shift for each dimension
    claim_pos_expanded = claim_positions.unsqueeze(0).unsqueeze(2)
    current_pos = state.identity_positions.unsqueeze(1)

    # Direction of shift
    shift_direction = claim_pos_expanded - current_pos

    # Weight by belief strength, claim relevance, and inverse crystallization
    plasticity = cfg.identity_plasticity * (1.0 - state.identity_crystallization)
    relevance = state.claim_identity_relevance.unsqueeze(0)

    weighted_shift = (
        strong_belief_mask.unsqueeze(2).float()
        * beliefs.unsqueeze(2)
        * shift_direction
        * relevance
        * plasticity.unsqueeze(1)
    )

    # Aggregate across claims and apply
    identity_shift = weighted_shift.mean(dim=1)
    state.identity_positions = torch.clamp(
        state.identity_positions + identity_shift,
        0.0, 1.0
    )

    # Increase crystallization slightly over time
    state.identity_crystallization = torch.clamp(
        state.identity_crystallization + cfg.identity_crystallization_rate,
        0.0, 0.95
    )
