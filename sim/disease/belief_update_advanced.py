"""
Advanced Belief Update with Cognitive Architecture
====================================================
Integrates all cognitive components into a unified belief update.

This replaces the simple belief update with:
1. Dual-process (System 1/System 2) evaluation
2. Motivated reasoning and identity protection
3. Attention allocation and cognitive load
4. Source memory and credibility
5. Narrative competition and inoculation
6. Truth default with accuracy nudging

The result is a psychologically-grounded belief dynamics model
that should produce realistic adoption patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from sim.cognition.dual_process import (
    CognitiveState,
    DualProcessConfig,
    compute_processing_mode,
    system1_evaluation,
    system2_evaluation,
    integrate_dual_process,
    update_familiarity,
    update_cognitive_load,
)
from sim.cognition.motivated_reasoning import (
    IdentityConfig,
    IdentityState,
    compute_identity_threat,
    apply_confirmation_bias,
    compute_defensive_processing,
    compute_reactance,
)
from sim.cognition.attention import (
    AttentionConfig,
    AttentionState,
    compute_salience,
    allocate_attention,
    filter_by_attention,
    update_attention_fatigue,
)
from sim.cognition.source_memory import (
    SourceMemory,
    SourceCredibility,
    update_source_memory,
    compute_source_weighted_exposure,
)
from sim.narratives.competition import (
    NarrativeConfig,
    NarrativeState,
    compute_claim_competition,
    apply_belief_budget_constraint,
    compute_consistency_pressure,
)
from sim.narratives.inoculation import (
    InoculationConfig,
    InoculationState,
    compute_inoculation_resistance,
    update_inoculation_decay,
    build_natural_resistance,
)


@dataclass
class AdvancedBeliefConfig:
    """Configuration for advanced belief update."""

    # Core learning parameters
    # Calibrated for realistic adoption: ~25-35% final adoption in 30 days
    base_learning_rate: float = 0.15  # Moderate learning
    decay_rate: float = 0.008  # Low decay for stability
    baseline_belief: float = 0.05

    # Integration weights
    s1_emotional_weight: float = 0.4
    s2_evidence_weight: float = 0.5
    social_proof_weight: float = 0.22  # Tuned for 25-35% adoption

    # Dampening factors (create heterogeneous response)
    skepticism_dampening: float = 0.4  # Moderate skeptic resistance
    resistance_dampening: float = 0.4  # Moderate inoculation

    # Thresholds
    exposure_threshold: float = 0.01
    update_threshold: float = 0.001


@dataclass
class CognitiveArchitecture:
    """Container for all cognitive state components."""

    cognitive_state: CognitiveState
    identity_state: IdentityState
    attention_state: AttentionState
    source_memory: SourceMemory
    source_credibility: SourceCredibility
    narrative_state: NarrativeState
    inoculation_state: InoculationState


def advanced_belief_update(
    beliefs: torch.Tensor,
    exposure: torch.Tensor,
    social_proof: torch.Tensor,
    emotional_resonance: torch.Tensor,
    evidence_quality: torch.Tensor,
    source_credibility_signal: torch.Tensor,
    debunk_pressure: torch.Tensor,
    traits: Dict[str, torch.Tensor],
    claim_positions: torch.Tensor,
    arch: CognitiveArchitecture,
    cfg: AdvancedBeliefConfig,
    dual_cfg: DualProcessConfig,
    identity_cfg: IdentityConfig,
    attention_cfg: AttentionConfig,
    narrative_cfg: NarrativeConfig,
    inoculation_cfg: InoculationConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Perform advanced belief update with full cognitive architecture.

    This is the core function that integrates all psychological mechanisms.

    Returns:
        updated_beliefs: New belief values
        diagnostics: Dict of diagnostic tensors for analysis
    """
    n_agents, n_claims = beliefs.shape
    device = beliefs.device

    diagnostics = {}

    # =========== STEP 1: ATTENTION ALLOCATION ===========
    # Compute salience to determine what gets attended to
    salience = compute_salience(
        exposure=exposure,
        emotional_intensity=emotional_resonance,
        familiarity=arch.cognitive_state.claim_familiarity,
        social_signals=social_proof,
        personal_relevance=arch.identity_state.identity_salience.mean(dim=1, keepdim=True).expand(-1, n_claims),
        cfg=attention_cfg,
    )

    # Allocate limited attention
    attention_allocation = allocate_attention(
        salience=salience,
        state=arch.attention_state,
        cfg=attention_cfg,
    )

    # Filter exposure by attention
    attended_exposure, processing_depth = filter_by_attention(
        exposure=exposure,
        attention_allocation=attention_allocation,
        cfg=attention_cfg,
    )

    diagnostics["attention_allocation"] = attention_allocation
    diagnostics["processing_depth"] = processing_depth

    # =========== STEP 2: IDENTITY THREAT ASSESSMENT ===========
    identity_threat = compute_identity_threat(
        beliefs=beliefs,
        claim_positions=claim_positions,
        state=arch.identity_state,
        cfg=identity_cfg,
    )

    diagnostics["identity_threat"] = identity_threat

    # =========== STEP 3: DETERMINE PROCESSING MODE ===========
    processing_mode = compute_processing_mode(
        state=arch.cognitive_state,
        exposure_intensity=attended_exposure,
        identity_threat=identity_threat,
        cfg=dual_cfg,
    )

    diagnostics["processing_mode"] = processing_mode  # Higher = more System 1

    # =========== STEP 4: DUAL-PROCESS EVALUATION ===========
    # System 1: Fast, intuitive, emotion-driven
    narrative_fit = 1.0 - identity_threat  # Threats don't fit narrative

    s1_output = system1_evaluation(
        exposure=attended_exposure,
        emotional_resonance=emotional_resonance,
        familiarity=arch.cognitive_state.claim_familiarity,
        narrative_fit=narrative_fit,
        cfg=dual_cfg,
    )

    # System 2: Slow, analytical, evidence-driven
    # Logical consistency: lower if beliefs already high (confirmation) or
    # if claim contradicts other beliefs
    consistency = 1.0 - compute_consistency_pressure(beliefs, arch.narrative_state, narrative_cfg).abs()
    claim_complexity = 1.0 - arch.narrative_state.ground_truth  # False claims are "complex"

    s2_output = system2_evaluation(
        exposure=attended_exposure,
        evidence_quality=evidence_quality,
        source_credibility=source_credibility_signal,
        logical_consistency=consistency,
        complexity=claim_complexity.unsqueeze(0).expand(n_agents, -1),
        cfg=dual_cfg,
    )

    # Integrate S1 and S2 outputs
    integrated_acceptance, conflict_detected = integrate_dual_process(
        s1_output=s1_output,
        s2_output=s2_output,
        processing_mode=processing_mode,
        state=arch.cognitive_state,
        cfg=dual_cfg,
    )

    diagnostics["s1_output"] = s1_output
    diagnostics["s2_output"] = s2_output
    diagnostics["conflict_detected"] = conflict_detected

    # =========== STEP 5: MOTIVATED REASONING ADJUSTMENTS ===========
    # Apply confirmation bias (helps with spread, so keep as-is)
    biased_acceptance = apply_confirmation_bias(
        acceptance_prob=integrated_acceptance,
        beliefs=beliefs,
        cfg=identity_cfg,
    )

    # Compute defensive processing response to threats
    rejection_boost, source_discount = compute_defensive_processing(
        identity_threat=identity_threat,
        exposure_intensity=attended_exposure,
        state=arch.identity_state,
        cfg=identity_cfg,
    )

    # =========== STEP 6: NARRATIVE COMPETITION ===========
    competition_effect = compute_claim_competition(
        beliefs=beliefs,
        state=arch.narrative_state,
        cfg=narrative_cfg,
    )

    # =========== STEP 7: INOCULATION RESISTANCE ===========
    resistance = compute_inoculation_resistance(
        state=arch.inoculation_state,
        exposure=attended_exposure,
        cfg=inoculation_cfg,
    )

    diagnostics["resistance"] = resistance

    # =========== STEP 8: COMPUTE FINAL ACCEPTANCE ===========
    # Key insight: Social proof should BOOST acceptance when neighbors believe
    # This creates the cascade dynamics needed for realistic spread
    skepticism = traits["skepticism"].unsqueeze(1)

    # Total penalty is capped sum of all resistance factors
    total_penalty = (
        0.2 * rejection_boost  # Defensive rejection (reduced weight)
        + cfg.resistance_dampening * resistance  # Inoculation
        + cfg.skepticism_dampening * skepticism  # Trait skepticism
        - 0.1 * competition_effect  # Competition can help (if reinforcing)
    )
    total_penalty = torch.clamp(total_penalty, 0.0, 0.5)  # Cap at 50% reduction

    # Apply penalty first
    penalized_acceptance = biased_acceptance * (1.0 - total_penalty)

    # Social proof boost: ADDITIVE boost when neighbors believe
    # This is more powerful and creates cascade dynamics
    # When 30% of neighbors believe, add 0.2*0.3 = 0.06 to acceptance
    social_proof_additive = cfg.social_proof_weight * social_proof

    # Also add emotional resonance boost for viral content
    emotional_boost = 0.15 * emotional_resonance * (1.0 - beliefs)

    # Final acceptance with social and emotional boosts
    final_acceptance = penalized_acceptance + social_proof_additive + emotional_boost

    # Clamp to valid range
    final_acceptance = torch.clamp(final_acceptance, 0.0, 0.95)

    # =========== STEP 9: COMPUTE BELIEF UPDATE ===========
    # Exposure mask - but use soft threshold, not hard cutoff
    exposure_strength = torch.clamp(attended_exposure / (cfg.exposure_threshold + 1e-6), 0.0, 1.0)

    # Update toward acceptance probability
    # Key insight: beliefs should move toward final_acceptance, scaled by exposure
    acceptance_delta = final_acceptance - beliefs

    # Learning rate: base + processing depth bonus (additive, not multiplicative)
    # This ensures even shallow processing produces meaningful updates
    effective_lr = cfg.base_learning_rate * (0.5 + 0.5 * processing_depth)

    # Positive update: when acceptance > current belief, increase belief
    # The magnitude is proportional to:
    # 1. How much higher acceptance is than belief
    # 2. How much exposure occurred
    # 3. Processing depth (deeper = larger update)
    positive_mask = (acceptance_delta > 0).float()
    update_positive = (
        positive_mask
        * effective_lr
        * acceptance_delta
        * exposure_strength
    )

    # Decay toward baseline - only for agents NOT being reinforced
    # Exposure protects against decay (reinforcement learning)
    exposure_protection = torch.clamp(exposure_strength, 0.0, 0.9)
    decay_strength = cfg.decay_rate * (1.0 - exposure_protection)
    update_decay = decay_strength * (cfg.baseline_belief - beliefs)

    # Debunking pressure
    # Apply reactance first
    reactance = compute_reactance(
        correction_pressure=debunk_pressure,
        state=arch.identity_state,
        cfg=identity_cfg,
    )
    effective_debunk = debunk_pressure * (1.0 - reactance)
    update_debunk = -0.15 * effective_debunk * beliefs  # Slightly stronger debunking

    # Total update
    total_update = update_positive + update_decay + update_debunk

    # Apply belief budget constraint (soft constraint)
    constrained_update = apply_belief_budget_constraint(
        beliefs=beliefs,
        belief_update=total_update,
        cfg=narrative_cfg,
    )

    # Apply all updates above minimal threshold
    significant = constrained_update.abs() > cfg.update_threshold
    final_update = torch.where(significant, constrained_update, torch.zeros_like(constrained_update))

    # Apply update
    new_beliefs = beliefs + final_update
    new_beliefs = torch.clamp(new_beliefs, 0.0, 1.0)

    diagnostics["belief_update"] = final_update
    diagnostics["final_acceptance"] = final_acceptance

    # =========== STEP 10: UPDATE COGNITIVE STATE ===========
    update_familiarity(
        state=arch.cognitive_state,
        exposure=attended_exposure,
        cfg=dual_cfg,
    )

    update_cognitive_load(
        state=arch.cognitive_state,
        processing_intensity=processing_depth,
    )

    update_attention_fatigue(
        state=arch.attention_state,
        attention_used=attention_allocation,
        cfg=attention_cfg,
    )

    update_inoculation_decay(
        state=arch.inoculation_state,
        cfg=inoculation_cfg,
    )

    build_natural_resistance(
        state=arch.inoculation_state,
        beliefs=new_beliefs,
        prev_beliefs=beliefs,
        exposure=attended_exposure,
    )

    return new_beliefs, diagnostics


def compute_evidence_quality(
    source_credibility: torch.Tensor,
    claim_falsifiability: torch.Tensor,
    social_proof: torch.Tensor,
) -> torch.Tensor:
    """
    Compute perceived evidence quality for System 2 evaluation.

    Evidence quality is based on:
    - Source credibility
    - Claim falsifiability (more falsifiable = higher quality)
    - Social proof (many believers = social evidence)
    """
    # Falsifiability as quality indicator (falsifiable claims are "honest")
    falsifiability_quality = claim_falsifiability.unsqueeze(0)

    # Social proof as evidence
    social_evidence = 0.3 * social_proof

    # Source credibility
    source_quality = 0.4 * source_credibility

    evidence = 0.3 * falsifiability_quality + source_quality + social_evidence

    return torch.clamp(evidence, 0.0, 1.0)


def compute_emotional_resonance(
    agent_emotions: Dict[str, torch.Tensor],
    claim_emotions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute emotional resonance between agents and claims.

    High resonance when agent's emotional state matches claim's emotional profile.
    """
    n_agents = agent_emotions["fear"].shape[0]
    n_claims = claim_emotions.shape[0]

    # claim_emotions: (n_claims, 3) for fear, anger, hope

    # Agent emotional state
    agent_fear = agent_emotions["fear"].unsqueeze(1)
    agent_anger = agent_emotions["anger"].unsqueeze(1)
    agent_hope = agent_emotions["hope"].unsqueeze(1)

    # Claim emotional profile
    claim_fear = claim_emotions[:, 0].unsqueeze(0)
    claim_anger = claim_emotions[:, 1].unsqueeze(0)
    claim_hope = claim_emotions[:, 2].unsqueeze(0)

    # Resonance is product of agent emotion and claim emotion weight
    resonance = (
        agent_fear * claim_fear
        + agent_anger * claim_anger
        + agent_hope * claim_hope
    )

    return resonance
