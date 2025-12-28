"""
Competing Narratives and Inoculation Theory
=============================================
Models how multiple claims interact and how prebunking builds resistance.

Key concepts:
1. Narrative Competition: Claims compete for limited belief capacity
2. Belief Budgets: Agents can't believe contradictory things equally
3. Inoculation: Pre-exposure to weakened misinformation builds resistance
4. Prebunking: Proactive inoculation before exposure
5. Truth Default: Initial tendency to accept information

References:
- van der Linden, S., et al. (2017). Inoculating the public against misinformation
- Lewandowsky, S., et al. (2020). The Debunking Handbook
- Roozenbeek, J., & van der Linden, S. (2019). Fake news game confers resistance
"""

from sim.narratives.competition import (
    NarrativeConfig,
    NarrativeState,
    initialize_narrative_state,
    compute_claim_competition,
    apply_belief_budget_constraint,
    update_narrative_bundles,
)

from sim.narratives.inoculation import (
    InoculationConfig,
    InoculationState,
    initialize_inoculation_state,
    apply_prebunking,
    compute_inoculation_resistance,
    update_inoculation_decay,
)

from sim.narratives.truth_default import (
    TruthDefaultConfig,
    compute_truth_default_acceptance,
    apply_accuracy_nudge,
    compute_source_plausibility,
)

__all__ = [
    "NarrativeConfig",
    "NarrativeState",
    "initialize_narrative_state",
    "compute_claim_competition",
    "apply_belief_budget_constraint",
    "update_narrative_bundles",
    "InoculationConfig",
    "InoculationState",
    "initialize_inoculation_state",
    "apply_prebunking",
    "compute_inoculation_resistance",
    "update_inoculation_decay",
    "TruthDefaultConfig",
    "compute_truth_default_acceptance",
    "apply_accuracy_nudge",
    "compute_source_plausibility",
]
