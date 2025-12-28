"""
Approximate Bayesian Computation
=================================
Implements ABC for parameter calibration without likelihood.

ABC works by:
1. Sampling parameters from prior
2. Simulating with those parameters
3. Computing summary statistics
4. Accepting if summary statistics close to observed

This avoids computing intractable likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from sim.calibration.priors import (
    ParameterPrior,
    sample_from_priors,
    perturb_params,
    compute_prior_probability,
)


class ABCConfig(BaseModel):
    """Configuration for ABC algorithm."""

    # Basic ABC parameters
    n_samples: int = 1000
    acceptance_threshold: float = 0.1

    # ABC-SMC parameters
    n_particles: int = 100
    n_generations: int = 10
    threshold_schedule: str = "adaptive"  # "adaptive" or "linear"
    ess_threshold: float = 0.5  # Effective sample size threshold

    # Simulation parameters
    n_sim_per_param: int = 3  # Simulations per parameter set for robustness
    max_sim_attempts: int = 10000

    # Distance computation
    distance_metric: str = "euclidean"  # "euclidean", "manhattan", "mahalanobis"
    summary_stats: List[str] = [
        "final_adoption_fraction",
        "mean_belief",
        "belief_variance",
        "days_to_peak",
        "structural_virality",
    ]


@dataclass
class ABCResult:
    """Results from ABC calibration."""

    accepted_params: List[Dict[str, float]]
    distances: List[float]
    weights: List[float]
    n_attempted: int
    acceptance_rate: float
    final_threshold: float
    summary_stats_observed: Dict[str, float]
    summary_stats_simulated: List[Dict[str, float]]

    def posterior_mean(self) -> Dict[str, float]:
        """Compute weighted posterior mean."""
        means = {}
        total_weight = sum(self.weights)

        if len(self.accepted_params) == 0:
            return {}

        for param_name in self.accepted_params[0].keys():
            weighted_sum = sum(
                p[param_name] * w
                for p, w in zip(self.accepted_params, self.weights)
            )
            means[param_name] = weighted_sum / total_weight

        return means

    def posterior_std(self) -> Dict[str, float]:
        """Compute weighted posterior standard deviation."""
        means = self.posterior_mean()
        stds = {}
        total_weight = sum(self.weights)

        for param_name in self.accepted_params[0].keys():
            weighted_sq_diff = sum(
                ((p[param_name] - means[param_name]) ** 2) * w
                for p, w in zip(self.accepted_params, self.weights)
            )
            stds[param_name] = np.sqrt(weighted_sq_diff / total_weight)

        return stds

    def credible_interval(
        self, param_name: str, level: float = 0.95
    ) -> Tuple[float, float]:
        """Compute credible interval for a parameter."""
        values = [p[param_name] for p in self.accepted_params]
        weights = np.array(self.weights)
        weights = weights / weights.sum()

        # Sort by value
        sorted_indices = np.argsort(values)
        sorted_values = np.array(values)[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Cumulative weights
        cumsum = np.cumsum(sorted_weights)

        # Find quantiles
        alpha = (1 - level) / 2
        lower_idx = np.searchsorted(cumsum, alpha)
        upper_idx = np.searchsorted(cumsum, 1 - alpha)

        lower = sorted_values[max(0, lower_idx - 1)]
        upper = sorted_values[min(len(sorted_values) - 1, upper_idx)]

        return float(lower), float(upper)


def compute_summary_statistics(
    beliefs: np.ndarray,
    adoption_threshold: float,
    belief_history: Optional[List[np.ndarray]] = None,
    cascade_stats: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Compute summary statistics from simulation output.

    These statistics are compared to empirical targets.
    """
    n_agents, n_claims = beliefs.shape

    # Adoption fraction
    adopted = beliefs >= adoption_threshold
    adoption_fraction = adopted.mean()

    # Mean belief
    mean_belief = beliefs.mean()

    # Belief variance
    belief_variance = beliefs.var()

    # Bimodality (fraction in tails)
    tail_fraction = ((beliefs < 0.2) | (beliefs > 0.8)).mean()

    stats = {
        "final_adoption_fraction": float(adoption_fraction),
        "mean_belief": float(mean_belief),
        "belief_variance": float(belief_variance),
        "tail_fraction": float(tail_fraction),
    }

    # Time-series statistics if history available
    if belief_history and len(belief_history) > 1:
        adoption_series = [
            (h >= adoption_threshold).mean() for h in belief_history
        ]
        peak_idx = np.argmax(adoption_series)
        stats["days_to_peak"] = float(peak_idx)

        # Growth rate at peak
        if peak_idx > 0:
            growth_rate = (adoption_series[peak_idx] - adoption_series[peak_idx - 1])
            stats["peak_growth_rate"] = float(growth_rate)

    # Cascade statistics if available
    if cascade_stats:
        stats.update(cascade_stats)

    return stats


def compute_distance(
    observed: Dict[str, float],
    simulated: Dict[str, float],
    targets: Dict[str, Tuple[float, float]],
    metric: str = "euclidean",
) -> float:
    """
    Compute distance between observed and simulated statistics.

    Normalizes by tolerance from targets.
    """
    distances = []

    for stat_name, (target_value, tolerance) in targets.items():
        if stat_name in simulated:
            sim_value = simulated[stat_name]
            normalized_diff = abs(sim_value - target_value) / (tolerance + 1e-10)
            distances.append(normalized_diff)

    if len(distances) == 0:
        return float('inf')

    if metric == "euclidean":
        return float(np.sqrt(np.sum(np.array(distances) ** 2)))
    elif metric == "manhattan":
        return float(np.sum(distances))
    elif metric == "max":
        return float(np.max(distances))
    else:
        return float(np.sqrt(np.sum(np.array(distances) ** 2)))


def run_abc_rejection(
    simulate_fn: Callable[[Dict[str, float]], Dict[str, float]],
    priors: Dict[str, ParameterPrior],
    targets: Dict[str, Tuple[float, float]],
    cfg: ABCConfig,
    rng: np.random.Generator,
) -> ABCResult:
    """
    Run basic ABC rejection sampling.

    Args:
        simulate_fn: Function that takes params and returns summary stats
        priors: Prior distributions for parameters
        targets: Target statistics as (value, tolerance) tuples
        cfg: ABC configuration
        rng: Random number generator

    Returns:
        ABCResult with accepted parameters and distances
    """
    accepted_params = []
    distances = []
    summary_stats_simulated = []
    n_attempted = 0

    while len(accepted_params) < cfg.n_samples and n_attempted < cfg.max_sim_attempts:
        # Sample from prior
        param_set = sample_from_priors(priors, rng, n_samples=1)[0]
        n_attempted += 1

        # Simulate
        try:
            sim_stats = simulate_fn(param_set)
        except Exception:
            continue

        # Compute distance
        distance = compute_distance({}, sim_stats, targets, cfg.distance_metric)

        # Accept if below threshold
        if distance < cfg.acceptance_threshold:
            accepted_params.append(param_set)
            distances.append(distance)
            summary_stats_simulated.append(sim_stats)

    # Uniform weights for rejection sampling
    weights = [1.0 / len(accepted_params)] * len(accepted_params) if accepted_params else []

    return ABCResult(
        accepted_params=accepted_params,
        distances=distances,
        weights=weights,
        n_attempted=n_attempted,
        acceptance_rate=len(accepted_params) / n_attempted if n_attempted > 0 else 0,
        final_threshold=cfg.acceptance_threshold,
        summary_stats_observed={k: v for k, (v, _) in targets.items()},
        summary_stats_simulated=summary_stats_simulated,
    )


def run_abc_smc(
    simulate_fn: Callable[[Dict[str, float]], Dict[str, float]],
    priors: Dict[str, ParameterPrior],
    targets: Dict[str, Tuple[float, float]],
    cfg: ABCConfig,
    rng: np.random.Generator,
) -> ABCResult:
    """
    Run ABC-SMC (Sequential Monte Carlo) for more efficient calibration.

    Uses adaptive thresholding and importance sampling.
    """
    n_particles = cfg.n_particles
    n_generations = cfg.n_generations

    # Initialize with rejection sampling at loose threshold
    initial_threshold = 10.0 * cfg.acceptance_threshold

    particles = []
    distances = []
    weights = []

    # Initial population from prior
    n_attempts = 0
    while len(particles) < n_particles and n_attempts < cfg.max_sim_attempts:
        param_set = sample_from_priors(priors, rng, n_samples=1)[0]
        n_attempts += 1

        try:
            sim_stats = simulate_fn(param_set)
            distance = compute_distance({}, sim_stats, targets, cfg.distance_metric)

            if distance < initial_threshold:
                particles.append(param_set)
                distances.append(distance)
                weights.append(1.0)
        except Exception:
            continue

    if len(particles) < n_particles // 2:
        # Not enough particles, return what we have
        return ABCResult(
            accepted_params=particles,
            distances=distances,
            weights=weights,
            n_attempted=n_attempts,
            acceptance_rate=len(particles) / n_attempts if n_attempts > 0 else 0,
            final_threshold=initial_threshold,
            summary_stats_observed={k: v for k, (v, _) in targets.items()},
            summary_stats_simulated=[],
        )

    # SMC generations
    current_threshold = initial_threshold

    for gen in range(n_generations):
        # Adapt threshold based on current distances
        sorted_distances = np.sort(distances)
        new_threshold = sorted_distances[int(len(sorted_distances) * 0.5)]
        new_threshold = max(new_threshold, cfg.acceptance_threshold)

        if new_threshold >= current_threshold:
            break  # Can't improve further

        current_threshold = new_threshold

        # Resample and perturb
        new_particles = []
        new_distances = []
        new_weights = []

        # Normalize weights
        weight_arr = np.array(weights)
        weight_arr = weight_arr / weight_arr.sum()

        # Compute kernel scale (twice weighted covariance)
        param_names = list(particles[0].keys())
        cov_matrix = np.zeros((len(param_names), len(param_names)))

        for i, name_i in enumerate(param_names):
            for j, name_j in enumerate(param_names):
                mean_i = sum(p[name_i] * w for p, w in zip(particles, weight_arr))
                mean_j = sum(p[name_j] * w for p, w in zip(particles, weight_arr))
                cov = sum(
                    (p[name_i] - mean_i) * (p[name_j] - mean_j) * w
                    for p, w in zip(particles, weight_arr)
                )
                cov_matrix[i, j] = cov

        scale = 2.0 * np.sqrt(np.diag(cov_matrix).mean() + 1e-10)

        n_attempts_gen = 0
        while len(new_particles) < n_particles and n_attempts_gen < cfg.max_sim_attempts:
            # Sample parent
            parent_idx = rng.choice(len(particles), p=weight_arr)
            parent = particles[parent_idx]

            # Perturb
            candidate = perturb_params(parent, priors, rng, scale=scale)
            n_attempts_gen += 1

            # Check prior support
            prior_prob = compute_prior_probability(candidate, priors)
            if prior_prob == -np.inf:
                continue

            # Simulate
            try:
                sim_stats = simulate_fn(candidate)
                distance = compute_distance({}, sim_stats, targets, cfg.distance_metric)

                if distance < current_threshold:
                    new_particles.append(candidate)
                    new_distances.append(distance)

                    # Compute importance weight
                    # w_new = prior / sum(w_old * kernel)
                    # Simplified: uniform weights within threshold
                    new_weights.append(1.0)
            except Exception:
                continue

        if len(new_particles) >= n_particles // 2:
            particles = new_particles
            distances = new_distances
            weights = new_weights
        else:
            break  # Can't maintain population

    # Normalize final weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    return ABCResult(
        accepted_params=particles,
        distances=distances,
        weights=weights,
        n_attempted=n_attempts,
        acceptance_rate=len(particles) / n_attempts if n_attempts > 0 else 0,
        final_threshold=current_threshold,
        summary_stats_observed={k: v for k, (v, _) in targets.items()},
        summary_stats_simulated=[],
    )
