"""
Cognitive Architecture Module
=============================
Implements psychologically-grounded cognitive processing based on:
- Kahneman's Dual-Process Theory (System 1 / System 2)
- Motivated Reasoning and Identity-Protective Cognition
- Bounded Rationality and Attention Economics
- Source Memory and Credibility Tracking

This module represents a fundamental advance over simple belief update models
by incorporating decades of cognitive psychology research.
"""

from sim.cognition.dual_process import (
    CognitiveState,
    DualProcessConfig,
    initialize_cognitive_states,
    compute_processing_mode,
    system1_evaluation,
    system2_evaluation,
    integrate_dual_process,
)

from sim.cognition.motivated_reasoning import (
    IdentityConfig,
    compute_identity_threat,
    apply_confirmation_bias,
    compute_defensive_processing,
    update_identity_salience,
)

from sim.cognition.attention import (
    AttentionConfig,
    AttentionState,
    allocate_attention,
    compute_cognitive_load,
    filter_by_attention,
    update_attention_fatigue,
)

from sim.cognition.source_memory import (
    SourceMemory,
    SourceCredibility,
    initialize_source_memory,
    update_source_credibility,
    retrieve_source_history,
    compute_source_weighted_exposure,
)

__all__ = [
    "CognitiveState",
    "DualProcessConfig",
    "initialize_cognitive_states",
    "compute_processing_mode",
    "system1_evaluation",
    "system2_evaluation",
    "integrate_dual_process",
    "IdentityConfig",
    "compute_identity_threat",
    "apply_confirmation_bias",
    "compute_defensive_processing",
    "update_identity_salience",
    "AttentionConfig",
    "AttentionState",
    "allocate_attention",
    "compute_cognitive_load",
    "filter_by_attention",
    "update_attention_fatigue",
    "SourceMemory",
    "SourceCredibility",
    "initialize_source_memory",
    "update_source_credibility",
    "retrieve_source_history",
    "compute_source_weighted_exposure",
]
