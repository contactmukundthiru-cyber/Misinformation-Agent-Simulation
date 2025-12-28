"""
Validation Module
==================
Validates calibrated model against held-out data and computes diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from sim.calibration.abc import ABCResult, compute_distance
from sim.calibration.empirical_targets import EmpiricalTargets


@dataclass
class ValidationResult:
    """Results from model validation."""

    target_name: str
    observed_value: float
    predicted_mean: float
    predicted_std: float
    prediction_interval: Tuple[float, float]
    in_interval: bool
    z_score: float
    distance: float


def validate_against_targets(
    result: ABCResult,
    simulate_fn: Callable[[Dict[str, float]], Dict[str, float]],
    targets: EmpiricalTargets,
    n_simulations: int = 100,
    rng: np.random.Generator = None,
) -> List[ValidationResult]:
    """
    Validate calibrated model against targets.

    Runs simulations from posterior and compares to target values.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(result.accepted_params) == 0:
        return []

    # Sample from posterior
    weights = np.array(result.weights)
    weights = weights / weights.sum()

    simulated_stats = []

    for _ in range(n_simulations):
        # Sample parameter set
        idx = rng.choice(len(result.accepted_params), p=weights)
        params = result.accepted_params[idx]

        try:
            stats = simulate_fn(params)
            simulated_stats.append(stats)
        except Exception:
            continue

    if len(simulated_stats) == 0:
        return []

    # Compute validation metrics
    validation_results = []

    for name, target in targets.targets.items():
        observed = target.value
        tolerance = target.tolerance

        # Get simulated values for this statistic
        sim_values = [s.get(name, np.nan) for s in simulated_stats]
        sim_values = [v for v in sim_values if not np.isnan(v)]

        if len(sim_values) == 0:
            continue

        sim_values = np.array(sim_values)

        predicted_mean = float(np.mean(sim_values))
        predicted_std = float(np.std(sim_values))

        # 95% prediction interval
        lower = float(np.percentile(sim_values, 2.5))
        upper = float(np.percentile(sim_values, 97.5))

        in_interval = lower <= observed <= upper

        # Z-score
        z_score = (observed - predicted_mean) / (predicted_std + 1e-10)

        # Distance (normalized)
        distance = abs(observed - predicted_mean) / (tolerance + 1e-10)

        validation_results.append(ValidationResult(
            target_name=name,
            observed_value=observed,
            predicted_mean=predicted_mean,
            predicted_std=predicted_std,
            prediction_interval=(lower, upper),
            in_interval=in_interval,
            z_score=float(z_score),
            distance=float(distance),
        ))

    return validation_results


def compute_coverage(
    validation_results: List[ValidationResult],
) -> Dict[str, float]:
    """
    Compute coverage statistics.

    Coverage = fraction of targets where observed value is in prediction interval.
    Good calibration should give ~95% coverage for 95% intervals.
    """
    if len(validation_results) == 0:
        return {"coverage": 0.0, "mean_z_score": 0.0, "mean_distance": 0.0}

    in_interval = [r.in_interval for r in validation_results]
    z_scores = [r.z_score for r in validation_results]
    distances = [r.distance for r in validation_results]

    return {
        "coverage": float(np.mean(in_interval)),
        "mean_z_score": float(np.mean(np.abs(z_scores))),
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "n_targets": len(validation_results),
    }


def posterior_predictive_check(
    result: ABCResult,
    simulate_fn: Callable[[Dict[str, float]], Dict[str, float]],
    statistic_fn: Callable[[Dict[str, float]], float],
    observed_statistic: float,
    n_simulations: int = 100,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """
    Perform posterior predictive check for a specific statistic.

    Computes p-value: fraction of simulations with statistic more extreme
    than observed.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(result.accepted_params) == 0:
        return {"p_value": np.nan, "n_simulations": 0}

    weights = np.array(result.weights)
    weights = weights / weights.sum()

    simulated_statistics = []

    for _ in range(n_simulations):
        idx = rng.choice(len(result.accepted_params), p=weights)
        params = result.accepted_params[idx]

        try:
            stats = simulate_fn(params)
            stat_value = statistic_fn(stats)
            simulated_statistics.append(stat_value)
        except Exception:
            continue

    if len(simulated_statistics) == 0:
        return {"p_value": np.nan, "n_simulations": 0}

    simulated_statistics = np.array(simulated_statistics)

    # Two-sided p-value
    more_extreme = np.abs(simulated_statistics - np.mean(simulated_statistics)) >= \
                   np.abs(observed_statistic - np.mean(simulated_statistics))
    p_value = float(np.mean(more_extreme))

    return {
        "p_value": p_value,
        "n_simulations": len(simulated_statistics),
        "simulated_mean": float(np.mean(simulated_statistics)),
        "simulated_std": float(np.std(simulated_statistics)),
        "observed": observed_statistic,
    }


def compute_calibration_score(
    validation_results: List[ValidationResult],
) -> float:
    """
    Compute overall calibration score (0 = poor, 1 = excellent).

    Based on:
    - Coverage (should be ~95%)
    - Mean z-score (should be ~1)
    - Mean distance (should be <1)
    """
    if len(validation_results) == 0:
        return 0.0

    coverage = compute_coverage(validation_results)

    # Score components
    coverage_score = 1.0 - abs(coverage["coverage"] - 0.95) / 0.95
    z_score = 1.0 - min(abs(coverage["mean_z_score"] - 1.0) / 2.0, 1.0)
    distance_score = 1.0 - min(coverage["mean_distance"], 1.0)

    # Weighted average
    total_score = 0.4 * coverage_score + 0.3 * z_score + 0.3 * distance_score

    return float(max(0.0, min(1.0, total_score)))
