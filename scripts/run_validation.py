#!/usr/bin/env python3
"""
Validation Script for Advanced Misinformation Simulation
==========================================================
Demonstrates the improvements over the baseline simulation.

This script:
1. Runs both baseline and advanced simulations
2. Compares outputs against empirical targets
3. Generates comparison plots
4. Computes calibration scores
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.config import load_config, SimulationConfig
from sim.simulation import run_simulation
from sim.simulation_advanced import (
    AdvancedSimulationConfig,
    run_advanced_simulation,
)
from sim.calibration.empirical_targets import (
    GENERAL_MISINFORMATION_TARGETS,
    get_default_targets,
)
from sim.calibration.abc import compute_distance
from sim.cognition.dual_process import DualProcessConfig
from sim.cognition.motivated_reasoning import IdentityConfig
from sim.cognition.attention import AttentionConfig
from sim.dynamics.network_evolution import NetworkEvolutionConfig
from sim.dynamics.influence import InfluenceConfig
from sim.cascades.tracker import CascadeConfig
from sim.narratives.competition import NarrativeConfig
from sim.narratives.inoculation import InoculationConfig
from sim.disease.belief_update_advanced import AdvancedBeliefConfig


logging.basicConfig(level=logging.INFO)


def create_default_config() -> SimulationConfig:
    """Create default simulation config."""
    return SimulationConfig()


def create_advanced_config(base: SimulationConfig) -> AdvancedSimulationConfig:
    """Create advanced simulation config with tuned parameters."""
    return AdvancedSimulationConfig(
        base=base,
        dual_process=DualProcessConfig(
            base_s1_tendency=0.55,
            cognitive_load_s1_boost=0.25,
            stakes_s2_boost=0.3,
        ),
        identity=IdentityConfig(
            threat_sensitivity=0.5,
            confirmation_strength=0.35,
            reactance_strength=0.25,
        ),
        attention=AttentionConfig(
            emotional_salience_weight=0.3,
            fatigue_rate=0.12,
        ),
        network_evolution=NetworkEvolutionConfig(
            rewiring_rate=0.015,
            heterophily_dissolution_rate=0.08,
        ),
        influence=InfluenceConfig(
            accuracy_influence_boost=0.08,
        ),
        cascade=CascadeConfig(),
        narrative=NarrativeConfig(
            competition_strength=0.4,
            enable_belief_budget=True,
        ),
        inoculation=InoculationConfig(
            inoculation_strength=0.5,
        ),
        advanced_belief=AdvancedBeliefConfig(
            base_learning_rate=0.06,
            decay_rate=0.02,
            skepticism_dampening=0.5,
        ),
    )


def compare_to_targets(metrics: pd.DataFrame, summary: dict) -> dict:
    """Compare simulation results to empirical targets."""
    targets = get_default_targets()
    target_dict = targets.get_target_dict()

    # Extract relevant statistics from simulation
    final_metrics = metrics[metrics["day"] == metrics["day"].max()]

    observed = {
        "final_adoption_fraction": float(final_metrics["adoption_fraction"].mean()),
        "mean_belief": float(final_metrics["mean_belief"].mean()),
        "belief_variance": float(final_metrics["variance"].mean()),
    }

    # Add from summary
    if "structural_virality" in summary:
        observed["structural_virality"] = summary.get("mean_structural_virality", 3.0)

    # Compute distance
    available_targets = {
        k: v for k, v in target_dict.items() if k in observed
    }
    distance = compute_distance({}, observed, available_targets)

    return {
        "observed": observed,
        "targets": {k: v[0] for k, v in available_targets.items()},
        "distance": distance,
    }


def generate_comparison_plots(
    baseline_metrics: pd.DataFrame,
    advanced_metrics: pd.DataFrame,
    output_dir: Path,
):
    """Generate comparison plots between baseline and advanced."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Adoption curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Baseline
    ax = axes[0]
    for claim in baseline_metrics["claim"].unique():
        subset = baseline_metrics[baseline_metrics["claim"] == claim]
        ax.plot(subset["day"], subset["adoption_fraction"], label=f"Claim {claim}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Adoption Fraction")
    ax.set_title("Baseline Model")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.25, color='red', linestyle='--', label='Empirical target')

    # Advanced
    ax = axes[1]
    for claim in advanced_metrics["claim"].unique():
        subset = advanced_metrics[advanced_metrics["claim"] == claim]
        ax.plot(subset["day"], subset["adoption_fraction"], label=f"Claim {claim}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Adoption Fraction")
    ax.set_title("Advanced Cognitive Model")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.25, color='red', linestyle='--', label='Empirical target')

    plt.tight_layout()
    plt.savefig(output_dir / "adoption_comparison.png", dpi=150)
    plt.close()

    # Mean belief comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    mean_baseline = baseline_metrics.groupby("day")["mean_belief"].mean()
    ax.plot(mean_baseline.index, mean_baseline.values, 'b-', linewidth=2)
    ax.fill_between(mean_baseline.index, 0, mean_baseline.values, alpha=0.3)
    ax.set_xlabel("Day")
    ax.set_ylabel("Mean Belief")
    ax.set_title("Baseline: Mean Belief Over Time")
    ax.set_ylim(0, 1)

    ax = axes[1]
    mean_advanced = advanced_metrics.groupby("day")["mean_belief"].mean()
    ax.plot(mean_advanced.index, mean_advanced.values, 'g-', linewidth=2)
    ax.fill_between(mean_advanced.index, 0, mean_advanced.values, alpha=0.3)
    ax.set_xlabel("Day")
    ax.set_ylabel("Mean Belief")
    ax.set_title("Advanced: Mean Belief Over Time")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "belief_comparison.png", dpi=150)
    plt.close()

    logging.info(f"Comparison plots saved to {output_dir}")


def main():
    """Run validation comparison."""
    output_dir = Path("runs/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create configs
    base_config = create_default_config()
    base_config.sim.n_agents = 2000
    base_config.sim.steps = 180  # 6 months

    advanced_config = create_advanced_config(base_config)

    print("=" * 60)
    print("MISINFORMATION SIMULATION VALIDATION")
    print("=" * 60)

    # Run baseline
    print("\n[1/2] Running baseline simulation...")
    baseline_output = run_simulation(base_config, output_dir / "baseline")
    print(f"  Final adoption: {baseline_output.summary.get('claim_0_final_adoption', 'N/A'):.3f}")

    # Run advanced
    print("\n[2/2] Running advanced cognitive simulation...")
    advanced_output = run_advanced_simulation(advanced_config, output_dir / "advanced")
    print(f"  Final adoption: {advanced_output.summary['final_adoption_mean']:.3f}")

    # Compare to targets
    print("\n" + "=" * 60)
    print("COMPARISON TO EMPIRICAL TARGETS")
    print("=" * 60)

    baseline_comparison = compare_to_targets(baseline_output.metrics, baseline_output.summary)
    advanced_comparison = compare_to_targets(advanced_output.metrics, advanced_output.summary)

    print("\nEmpirical Targets (from literature):")
    for key, value in baseline_comparison["targets"].items():
        print(f"  {key}: {value:.3f}")

    print("\nBaseline Model:")
    for key, value in baseline_comparison["observed"].items():
        target = baseline_comparison["targets"].get(key, "N/A")
        print(f"  {key}: {value:.3f} (target: {target})")
    print(f"  Distance from targets: {baseline_comparison['distance']:.3f}")

    print("\nAdvanced Model:")
    for key, value in advanced_comparison["observed"].items():
        target = advanced_comparison["targets"].get(key, "N/A")
        print(f"  {key}: {value:.3f} (target: {target})")
    print(f"  Distance from targets: {advanced_comparison['distance']:.3f}")

    # Improvement
    improvement = (baseline_comparison["distance"] - advanced_comparison["distance"]) / baseline_comparison["distance"] * 100
    print(f"\nImprovement: {improvement:.1f}% closer to empirical targets")

    # Additional advanced metrics
    print("\n" + "=" * 60)
    print("ADVANCED MODEL UNIQUE METRICS")
    print("=" * 60)
    print(f"  Cascade structural virality: {advanced_output.cascade_stats.get('mean_structural_virality', 'N/A')}")
    print(f"  Max echo chamber modularity: {max(advanced_output.echo_chamber_history) if advanced_output.echo_chamber_history else 'N/A':.3f}")
    print(f"  Power law exponent: {advanced_output.cascade_stats.get('power_law_exponent', 'N/A')}")

    # Generate plots
    print("\nGenerating comparison plots...")
    generate_comparison_plots(
        baseline_output.metrics,
        advanced_output.metrics,
        output_dir / "plots"
    )

    # Save comparison summary
    comparison_summary = {
        "baseline": {
            "final_adoption": baseline_comparison["observed"]["final_adoption_fraction"],
            "distance_to_targets": baseline_comparison["distance"],
        },
        "advanced": {
            "final_adoption": advanced_comparison["observed"]["final_adoption_fraction"],
            "distance_to_targets": advanced_comparison["distance"],
            "cascade_stats": advanced_output.cascade_stats,
        },
        "improvement_percent": improvement,
    }

    with (output_dir / "comparison_summary.json").open("w") as f:
        json.dump(comparison_summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
