"""
Information Cascade Tracking and Analysis
==========================================
Implements explicit cascade structure tracking and analysis.

Key features:
1. Cascade Recording: Track who exposed whom for each belief adoption
2. Cascade Trees: Build tree structure of information flow
3. Structural Virality: Measure cascade structure (viral vs broadcast)
4. Cascade Size Distribution: Should follow power law
5. Generation Tracking: True epidemiological generations

References:
- Goel, S., et al. (2016). The structural virality of online diffusion
- Vosoughi, S., et al. (2018). The spread of true and false news online
- Juul, J. L., & Ugander, J. (2022). Comparing information diffusion mechanisms
"""

from sim.cascades.tracker import (
    CascadeConfig,
    CascadeState,
    CascadeEvent,
    initialize_cascade_tracker,
    record_adoption_event,
    get_cascade_for_claim,
    prune_old_events,
)

from sim.cascades.analysis import (
    compute_structural_virality,
    compute_cascade_depth,
    compute_cascade_breadth,
    build_cascade_tree,
    analyze_cascade_statistics,
    compute_generation_time_distribution,
    compute_cascade_size_distribution,
    fit_power_law_exponent,
)

from sim.cascades.r_effective import (
    REffectiveTracker,
    compute_true_r_effective,
    compute_generation_interval,
    compute_growth_rate,
)

__all__ = [
    "CascadeConfig",
    "CascadeState",
    "CascadeEvent",
    "initialize_cascade_tracker",
    "record_adoption_event",
    "get_cascade_for_claim",
    "prune_old_events",
    "compute_structural_virality",
    "compute_cascade_depth",
    "compute_cascade_breadth",
    "build_cascade_tree",
    "analyze_cascade_statistics",
    "compute_generation_time_distribution",
    "compute_cascade_size_distribution",
    "fit_power_law_exponent",
    "REffectiveTracker",
    "compute_true_r_effective",
    "compute_generation_interval",
    "compute_growth_rate",
]
