"""
Truth Default Theory Implementation
=====================================
Models the default human tendency to accept information as true.

Based on:
- Truth Default Theory (Levine, 2014)
- Accuracy nudging (Pennycook & Rand, 2021)
- Plausibility heuristics

Key insight: Humans default to accepting information as true
unless they have specific reason to doubt it.
"""

from __future__ import annotations

from typing import Dict

import torch
from pydantic import BaseModel


class TruthDefaultConfig(BaseModel):
    """Configuration for truth default behavior."""

    # Default acceptance
    truth_default_strength: float = 0.6
    skepticism_override_threshold: float = 0.7

    # Accuracy nudging
    accuracy_nudge_strength: float = 0.25
    accuracy_nudge_duration: int = 10  # Timesteps

    # Plausibility factors
    familiarity_boost: float = 0.2
    source_quality_weight: float = 0.3
    consistency_weight: float = 0.25

    # Trigger conditions
    trigger_suspect_probability: float = 0.15


def compute_truth_default_acceptance(
    exposure: torch.Tensor,
    familiarity: torch.Tensor,
    source_credibility: torch.Tensor,
    skepticism: torch.Tensor,
    cfg: TruthDefaultConfig,
) -> torch.Tensor:
    """
    Compute truth-default acceptance probability.

    Truth Default Theory: People accept information by default
    unless they have explicit reason to doubt.

    Returns: base acceptance probability before other factors
    """
    n_agents, n_claims = exposure.shape

    # Base truth default (high acceptance)
    base_acceptance = cfg.truth_default_strength * torch.ones_like(exposure)

    # Familiarity increases acceptance (fluency heuristic)
    familiarity_boost = cfg.familiarity_boost * familiarity

    # Source credibility matters
    source_boost = cfg.source_quality_weight * source_credibility

    # Skepticism can override truth default
    skepticism_override = skepticism.unsqueeze(1) > cfg.skepticism_override_threshold
    skepticism_penalty = skepticism_override.float() * 0.3

    acceptance = base_acceptance + familiarity_boost + source_boost - skepticism_penalty

    return torch.clamp(acceptance, 0.1, 0.95)


def trigger_suspicion(
    exposure: torch.Tensor,
    emotional_intensity: torch.Tensor,
    source_mismatch: torch.Tensor,
    cfg: TruthDefaultConfig,
) -> torch.Tensor:
    """
    Determine which exposures trigger suspicion (break truth default).

    Suspicion is triggered by:
    - Very high emotional intensity (seems like manipulation)
    - Source mismatch (unexpected source for content)
    - Inconsistency with prior knowledge
    """
    # Base trigger probability
    trigger_prob = cfg.trigger_suspect_probability * torch.ones_like(exposure)

    # High emotion triggers suspicion
    emotion_trigger = 0.2 * emotional_intensity

    # Source mismatch triggers suspicion
    mismatch_trigger = 0.3 * source_mismatch

    trigger_prob = trigger_prob + emotion_trigger + mismatch_trigger

    # Sample trigger events
    triggered = torch.rand_like(exposure) < trigger_prob

    return triggered


def apply_accuracy_nudge(
    beliefs: torch.Tensor,
    acceptance_prob: torch.Tensor,
    nudge_active: torch.Tensor,
    ground_truth: torch.Tensor,
    cfg: TruthDefaultConfig,
) -> torch.Tensor:
    """
    Apply accuracy nudge intervention.

    Accuracy nudging (Pennycook & Rand) works by making people
    think about accuracy before evaluating content.

    This increases scrutiny and reduces false-positive acceptance.
    """
    # Nudge effect: increased scrutiny
    nudge_expanded = nudge_active.unsqueeze(1).expand(-1, beliefs.shape[1])

    # Nudged agents apply more scrutiny
    scrutiny_boost = cfg.accuracy_nudge_strength * nudge_expanded.float()

    # Scrutiny reduces acceptance of false claims more
    # (Assumes agents have some latent ability to detect falsehood)
    false_claims = ground_truth < 0.5
    false_claim_expanded = false_claims.unsqueeze(0).expand(beliefs.shape[0], -1)

    # Reduce acceptance of false claims for nudged agents
    nudged_acceptance = torch.where(
        nudge_expanded & false_claim_expanded,
        acceptance_prob * (1 - scrutiny_boost),
        acceptance_prob
    )

    return nudged_acceptance


def compute_source_plausibility(
    source_type: torch.Tensor,
    claim_characteristics: torch.Tensor,
    cfg: TruthDefaultConfig,
) -> torch.Tensor:
    """
    Compute how plausible the source is for this type of claim.

    Some claims are more plausible from certain sources.
    E.g., health claims from doctors, political claims from news.
    """
    # Source-claim compatibility matrix
    # This would ideally be learned or configured
    # For now, use a simple heuristic

    # All combinations have baseline plausibility
    plausibility = 0.5 * torch.ones_like(claim_characteristics)

    return plausibility


def compute_consistency_with_prior(
    beliefs: torch.Tensor,
    exposure: torch.Tensor,
    exposure_direction: torch.Tensor,
) -> torch.Tensor:
    """
    Compute how consistent new exposure is with prior beliefs.

    Consistent exposure (matches beliefs) is more readily accepted.
    Inconsistent exposure (contradicts beliefs) triggers scrutiny.
    """
    # exposure_direction: 1 = supports claim, -1 = opposes claim

    # Consistency = belief × direction
    # High belief + supporting = consistent
    # Low belief + opposing = consistent
    # High belief + opposing = inconsistent

    belief_direction = 2 * beliefs - 1  # Map [0,1] to [-1,1]
    consistency = belief_direction * exposure_direction

    # High consistency = easy acceptance
    # Low consistency = scrutiny

    return torch.clamp(consistency, -1, 1)
