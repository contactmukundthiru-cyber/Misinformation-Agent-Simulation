"""
True Epidemiological R_effective Computation
==============================================
Computes proper reproduction number using generation intervals.

Unlike the naive R0-like metric, this tracks actual generation times
and computes R_eff using proper epidemiological methods.

References:
- Cori, A., et al. (2013). A new framework for real-time estimation of R
- Wallinga, J., & Teunis, P. (2004). Different epidemic curves
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sim.cascades.tracker import CascadeEvent, CascadeState


@dataclass
class REffectiveTracker:
    """Tracks data needed for R_effective computation."""

    n_claims: int
    window_size: int = 7

    # Daily new adoptions per claim
    daily_adoptions: Dict[int, List[int]] = field(default_factory=dict)

    # Daily adoptions by generation
    daily_by_generation: Dict[int, Dict[int, List[int]]] = field(default_factory=dict)

    # Computed R_eff history
    r_eff_history: Dict[int, List[float]] = field(default_factory=dict)

    # Generation interval estimates
    generation_intervals: Dict[int, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        for claim in range(self.n_claims):
            self.daily_adoptions[claim] = []
            self.daily_by_generation[claim] = {}
            self.r_eff_history[claim] = []
            self.generation_intervals[claim] = []


def compute_true_r_effective(
    tracker: REffectiveTracker,
    cascade_state: CascadeState,
    current_day: int,
    adoption_threshold: float = 0.7,
) -> Dict[int, float]:
    """
    Compute true R_effective for each claim using generation method.

    R_eff = (cases in generation g) / (cases in generation g-1)

    Averaged over recent generations with proper weighting.
    """
    r_eff = {}

    for claim in range(cascade_state.n_claims):
        events = cascade_state.events.get(claim, [])

        if len(events) < 10:
            r_eff[claim] = 0.0
            continue

        # Group events by generation
        gen_counts: Dict[int, int] = {}
        for event in events:
            gen = event.generation
            gen_counts[gen] = gen_counts.get(gen, 0) + 1

        if len(gen_counts) < 2:
            r_eff[claim] = 0.0
            continue

        # Compute R_eff for each generation pair
        generations = sorted(gen_counts.keys())
        r_values = []
        weights = []

        for i in range(1, len(generations)):
            prev_gen = generations[i - 1]
            curr_gen = generations[i]

            if gen_counts[prev_gen] > 0:
                r_gen = gen_counts[curr_gen] / gen_counts[prev_gen]
                r_values.append(r_gen)
                # Weight by recency and sample size
                weight = gen_counts[prev_gen] * (0.9 ** (max(generations) - curr_gen))
                weights.append(weight)

        if len(r_values) == 0:
            r_eff[claim] = 0.0
        else:
            # Weighted average
            weights_np = np.array(weights)
            r_values_np = np.array(r_values)
            r_eff[claim] = float(np.average(r_values_np, weights=weights_np))

        tracker.r_eff_history[claim].append(r_eff[claim])

    return r_eff


def compute_generation_interval(
    cascade_state: CascadeState,
    claim: int,
) -> Dict[str, float]:
    """
    Compute generation interval statistics for a claim.

    Generation interval = time from infector adoption to infectee adoption.
    """
    events = cascade_state.events.get(claim, [])

    if len(events) < 2:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
        }

    # Build adoption time lookup
    adoption_times = {e.adopter: e.time for e in events}

    intervals = []
    for event in events:
        if event.source >= 0 and event.source in adoption_times:
            interval = event.time - adoption_times[event.source]
            if interval > 0:
                intervals.append(interval)

    if len(intervals) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
        }

    intervals_np = np.array(intervals)

    return {
        "mean": float(np.mean(intervals_np)),
        "std": float(np.std(intervals_np)),
        "median": float(np.median(intervals_np)),
    }


def compute_growth_rate(
    tracker: REffectiveTracker,
    cascade_state: CascadeState,
    claim: int,
    window: int = 7,
) -> float:
    """
    Compute instantaneous growth rate from recent R_eff.

    Growth rate r = log(R_eff) / mean_generation_interval
    """
    if len(tracker.r_eff_history.get(claim, [])) == 0:
        return 0.0

    recent_r = tracker.r_eff_history[claim][-window:]
    mean_r = np.mean(recent_r)

    if mean_r <= 0:
        return -1.0  # Declining

    gen_interval = compute_generation_interval(cascade_state, claim)
    mean_gi = gen_interval["mean"]

    if mean_gi <= 0:
        mean_gi = 1.0  # Default to 1 day

    growth_rate = np.log(mean_r) / mean_gi

    return float(growth_rate)


def compute_time_varying_r(
    cascade_state: CascadeState,
    claim: int,
    window: int = 7,
) -> List[Tuple[int, float]]:
    """
    Compute time-varying R_eff over the epidemic curve.

    Uses a sliding window approach.

    Returns list of (day, R_eff) tuples.
    """
    events = cascade_state.events.get(claim, [])

    if len(events) < 5:
        return []

    # Group by day
    daily_counts: Dict[int, int] = {}
    for event in events:
        daily_counts[event.time] = daily_counts.get(event.time, 0) + 1

    if len(daily_counts) < 2:
        return []

    days = sorted(daily_counts.keys())
    r_series = []

    for i in range(window, len(days)):
        # Cases in window vs cases in previous window
        window_days = days[i - window + 1:i + 1]
        prev_window_days = days[max(0, i - 2 * window + 1):i - window + 1]

        if len(prev_window_days) == 0:
            continue

        window_cases = sum(daily_counts.get(d, 0) for d in window_days)
        prev_cases = sum(daily_counts.get(d, 0) for d in prev_window_days)

        if prev_cases > 0:
            r_t = window_cases / prev_cases
            r_series.append((days[i], r_t))

    return r_series


def estimate_herd_immunity_threshold(
    r_eff: float,
) -> float:
    """
    Estimate fraction needed for herd immunity.

    HIT = 1 - 1/R_0

    This assumes R_eff approximates R_0 (early epidemic).
    """
    if r_eff <= 1:
        return 0.0  # Already below threshold

    hit = 1 - 1 / r_eff
    return float(hit)


def compute_doubling_time(
    growth_rate: float,
) -> float:
    """
    Compute epidemic doubling time from growth rate.

    Doubling time = ln(2) / growth_rate
    """
    if growth_rate <= 0:
        return float('inf')  # Not growing

    return float(np.log(2) / growth_rate)


def project_final_size(
    r_eff: float,
    current_fraction: float,
    n_agents: int,
) -> int:
    """
    Project final epidemic size using simple SIR model.

    Assumes homogeneous mixing and constant R.
    """
    if r_eff <= 1:
        # Epidemic dying out, current is approximately final
        return int(current_fraction * n_agents)

    # Final size equation: R = 1 - exp(-R * z) / z
    # where z = final fraction infected
    # Use numerical approximation

    z = current_fraction
    for _ in range(100):  # Newton-Raphson iterations
        f = z - 1 + np.exp(-r_eff * z)
        f_prime = 1 + r_eff * np.exp(-r_eff * z)
        z_new = z - f / f_prime
        z_new = np.clip(z_new, current_fraction, 1.0)
        if abs(z_new - z) < 1e-6:
            break
        z = z_new

    return int(z * n_agents)
