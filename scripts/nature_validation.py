#!/usr/bin/env python3
"""
Comprehensive Validation Suite for Nature-Ready Submission
============================================================
Runs extensive validation across seeds, world configurations,
and intervention scenarios to produce publication-quality results.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

torch.use_deterministic_algorithms(False)

from sim.config import load_config
from sim.simulation_advanced import AdvancedSimulationConfig, run_advanced_simulation


def run_seed_sweep(
    world_config: str,
    seeds: List[int],
    n_agents: int = 1000,
    n_steps: int = 30,
) -> pd.DataFrame:
    """Run simulation across multiple seeds for a single world config."""
    results = []

    for seed in seeds:
        base_cfg = load_config(f'configs/{world_config}.yaml')
        base_cfg.sim.n_agents = n_agents
        base_cfg.sim.steps = n_steps
        base_cfg.sim.seed = seed
        base_cfg.sim.deterministic = False

        advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_advanced_simulation(advanced_cfg, tmpdir)

        # Extract key metrics
        results.append({
            'world': world_config,
            'seed': seed,
            'final_adoption': outputs.summary['adoption_fraction'],
            'mean_belief': outputs.summary['mean_belief'],
            'belief_variance': outputs.summary['belief_variance'],
            'total_cascades': outputs.summary.get('total_cascades', 0),
            'max_cascade_size': outputs.summary.get('max_cascade_size', 0),
        })

    return pd.DataFrame(results)


def run_intervention_test(
    prebunking_day: int,
    prebunking_fraction: float,
    seeds: List[int],
    n_agents: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare baseline vs intervention across seeds."""
    baseline_results = []
    intervention_results = []

    for seed in seeds:
        # Baseline (no intervention)
        base_cfg = load_config('configs/world_baseline.yaml')
        base_cfg.sim.n_agents = n_agents
        base_cfg.sim.steps = 30
        base_cfg.sim.seed = seed
        base_cfg.sim.deterministic = False

        advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_advanced_simulation(advanced_cfg, tmpdir)
        baseline_results.append({
            'seed': seed,
            'adoption': outputs.summary['adoption_fraction'],
            'mean_belief': outputs.summary['mean_belief'],
        })

        # With prebunking
        advanced_cfg.prebunking_day = prebunking_day
        advanced_cfg.prebunking_fraction = prebunking_fraction

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_advanced_simulation(advanced_cfg, tmpdir)
        intervention_results.append({
            'seed': seed,
            'adoption': outputs.summary['adoption_fraction'],
            'mean_belief': outputs.summary['mean_belief'],
        })

    return pd.DataFrame(baseline_results), pd.DataFrame(intervention_results)


def compute_statistics(df: pd.DataFrame, metric: str) -> Dict:
    """Compute publication-quality statistics."""
    values = df[metric].values
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75),
        'min': np.min(values),
        'max': np.max(values),
        'n': len(values),
        'ci95_low': np.mean(values) - 1.96 * np.std(values) / np.sqrt(len(values)),
        'ci95_high': np.mean(values) + 1.96 * np.std(values) / np.sqrt(len(values)),
    }


def run_sensitivity_analysis(
    param_name: str,
    param_values: List[float],
    seeds: List[int],
) -> pd.DataFrame:
    """Run sensitivity analysis on a single parameter."""
    results = []

    for value in param_values:
        for seed in seeds:
            base_cfg = load_config('configs/world_baseline.yaml')
            base_cfg.sim.n_agents = 1000
            base_cfg.sim.steps = 30
            base_cfg.sim.seed = seed
            base_cfg.sim.deterministic = False

            advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

            # Set parameter value
            if param_name == 'social_proof_weight':
                advanced_cfg.advanced_belief.social_proof_weight = value
            elif param_name == 'skepticism_dampening':
                advanced_cfg.advanced_belief.skepticism_dampening = value
            elif param_name == 'base_learning_rate':
                advanced_cfg.advanced_belief.base_learning_rate = value

            with tempfile.TemporaryDirectory() as tmpdir:
                outputs = run_advanced_simulation(advanced_cfg, tmpdir)

            results.append({
                'param': param_name,
                'value': value,
                'seed': seed,
                'adoption': outputs.summary['adoption_fraction'],
                'mean_belief': outputs.summary['mean_belief'],
            })

    return pd.DataFrame(results)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=50)
    parser.add_argument('--output', type=str, default='validation_results')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.n_seeds))

    print(f"Running Nature validation suite with {args.n_seeds} seeds...")
    print("=" * 60)

    # 1. Baseline world - extensive seed sweep
    print("\n[1/5] Running baseline seed sweep...")
    baseline_df = run_seed_sweep('world_baseline', seeds)
    baseline_df.to_csv(output_dir / 'baseline_seeds.csv', index=False)

    stats = compute_statistics(baseline_df, 'final_adoption')
    print(f"  Baseline adoption: {stats['mean']:.1%} ± {stats['std']:.1%}")
    print(f"  95% CI: [{stats['ci95_low']:.1%}, {stats['ci95_high']:.1%}]")

    # 2. All world configurations
    print("\n[2/5] Running all world configurations...")
    worlds = [
        'world_baseline',
        'world_high_trust_gov',
        'world_low_trust_gov',
        'world_strong_moderation',
        'world_collapsed_local_media',
        'world_high_religion_hub',
        'world_outrage_algorithm',
    ]

    world_results = []
    for world in worlds:
        try:
            df = run_seed_sweep(world, seeds[:20])  # 20 seeds per world
            stats = compute_statistics(df, 'final_adoption')
            world_results.append({
                'world': world,
                **stats
            })
            print(f"  {world}: {stats['mean']:.1%} ± {stats['std']:.1%}")
        except Exception as e:
            print(f"  {world}: ERROR - {e}")

    world_df = pd.DataFrame(world_results)
    world_df.to_csv(output_dir / 'world_comparison.csv', index=False)

    # 3. Intervention effects
    print("\n[3/5] Testing intervention effects...")
    baseline_int, prebunk_int = run_intervention_test(
        prebunking_day=5,
        prebunking_fraction=0.3,
        seeds=seeds[:30],
    )

    baseline_stats = compute_statistics(baseline_int, 'adoption')
    prebunk_stats = compute_statistics(prebunk_int, 'adoption')
    reduction = 1 - prebunk_stats['mean'] / baseline_stats['mean']

    print(f"  Baseline: {baseline_stats['mean']:.1%}")
    print(f"  Prebunking: {prebunk_stats['mean']:.1%}")
    print(f"  Reduction: {reduction:.1%}")

    intervention_summary = pd.DataFrame([
        {'condition': 'baseline', **baseline_stats},
        {'condition': 'prebunking', **prebunk_stats},
    ])
    intervention_summary.to_csv(output_dir / 'intervention_effects.csv', index=False)

    # 4. Sensitivity analysis
    print("\n[4/5] Running sensitivity analysis...")
    sensitivity_results = []

    # Social proof weight
    sp_df = run_sensitivity_analysis(
        'social_proof_weight',
        [0.10, 0.15, 0.20, 0.25, 0.30],
        seeds[:10],
    )
    sensitivity_results.append(sp_df)

    # Skepticism dampening
    sk_df = run_sensitivity_analysis(
        'skepticism_dampening',
        [0.2, 0.3, 0.4, 0.5, 0.6],
        seeds[:10],
    )
    sensitivity_results.append(sk_df)

    sensitivity_df = pd.concat(sensitivity_results)
    sensitivity_df.to_csv(output_dir / 'sensitivity_analysis.csv', index=False)

    # Compute sensitivity summary
    for param in ['social_proof_weight', 'skepticism_dampening']:
        param_df = sensitivity_df[sensitivity_df['param'] == param]
        grouped = param_df.groupby('value')['adoption'].agg(['mean', 'std'])
        print(f"  {param}:")
        for val, row in grouped.iterrows():
            print(f"    {val}: {row['mean']:.1%} ± {row['std']:.1%}")

    # 5. Summary statistics
    print("\n[5/5] Generating summary...")

    summary = {
        'baseline_adoption_mean': baseline_df['final_adoption'].mean(),
        'baseline_adoption_std': baseline_df['final_adoption'].std(),
        'baseline_adoption_ci95': [
            baseline_df['final_adoption'].mean() - 1.96 * baseline_df['final_adoption'].std() / np.sqrt(len(baseline_df)),
            baseline_df['final_adoption'].mean() + 1.96 * baseline_df['final_adoption'].std() / np.sqrt(len(baseline_df)),
        ],
        'intervention_reduction': reduction,
        'n_seeds_baseline': len(baseline_df),
        'n_worlds_tested': len(world_results),
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"\nKey Results:")
    print(f"  Baseline adoption: {summary['baseline_adoption_mean']:.1%} ± {summary['baseline_adoption_std']:.1%}")
    print(f"  95% CI: [{summary['baseline_adoption_ci95'][0]:.1%}, {summary['baseline_adoption_ci95'][1]:.1%}]")
    print(f"  Prebunking reduces adoption by: {summary['intervention_reduction']:.1%}")
    print(f"\nResults saved to: {output_dir}")
