"""
Bayesian Calibration Framework
===============================
Implements Approximate Bayesian Computation (ABC) for parameter calibration.

Calibrates simulation parameters against empirical stylized facts
from misinformation research literature.

Key components:
1. Empirical Targets: Stylized facts from literature
2. Prior Distributions: Reasonable parameter ranges
3. ABC-SMC: Sequential Monte Carlo for efficient calibration
4. Validation Metrics: Compare simulation to real-world data

References:
- Beaumont, M. A., et al. (2002). Approximate Bayesian computation
- Sisson, S. A., et al. (2007). Sequential Monte Carlo without likelihoods
- Vosoughi, S., et al. (2018). The spread of true and false news online
"""

from sim.calibration.empirical_targets import (
    EmpiricalTargets,
    COVID_MISINFORMATION_TARGETS,
    ELECTION_MISINFORMATION_TARGETS,
    GENERAL_MISINFORMATION_TARGETS,
    get_default_targets,
)

from sim.calibration.priors import (
    ParameterPrior,
    define_parameter_priors,
    sample_from_priors,
    compute_prior_probability,
)

from sim.calibration.abc import (
    ABCConfig,
    ABCResult,
    run_abc_rejection,
    run_abc_smc,
    compute_summary_statistics,
    compute_distance,
)

from sim.calibration.validation import (
    ValidationResult,
    validate_against_targets,
    compute_coverage,
    posterior_predictive_check,
)

__all__ = [
    "EmpiricalTargets",
    "COVID_MISINFORMATION_TARGETS",
    "ELECTION_MISINFORMATION_TARGETS",
    "GENERAL_MISINFORMATION_TARGETS",
    "get_default_targets",
    "ParameterPrior",
    "define_parameter_priors",
    "sample_from_priors",
    "compute_prior_probability",
    "ABCConfig",
    "ABCResult",
    "run_abc_rejection",
    "run_abc_smc",
    "compute_summary_statistics",
    "compute_distance",
    "ValidationResult",
    "validate_against_targets",
    "compute_coverage",
    "posterior_predictive_check",
]
