"""
Comprehensive Validation Suite for Nature Submission
=====================================================
Runs extensive validation with maximum data collection.
"""

import os
# Force CPU only BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import tempfile
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy import stats

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.config import load_config
from sim.simulation_advanced import AdvancedSimulationConfig, run_advanced_simulation

# Disable torch deterministic for speed
torch.use_deterministic_algorithms(False)

# Setup logging
logging.basicConfig(level=logging.WARNING)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_single_simulation(world, seed, n_agents, n_steps, param_overrides):
    """Run a single simulation with given parameters."""
    try:
        base_cfg = load_config(f'configs/{world}.yaml')
        base_cfg.sim.n_agents = n_agents
        base_cfg.sim.steps = n_steps
        base_cfg.sim.seed = seed
        base_cfg.sim.device = 'cpu'
        base_cfg.sim.deterministic = False

        advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

        # Apply parameter overrides
        for key, value in param_overrides.items():
            if key == 'scenario':
                continue  # Skip metadata
            if '.' in key:
                obj_name, attr_name = key.split('.', 1)
                obj = getattr(advanced_cfg, obj_name)
                setattr(obj, attr_name, value)
            else:
                setattr(advanced_cfg, key, value)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_advanced_simulation(advanced_cfg, tmpdir)

        return {
            'world': world,
            'seed': seed,
            'final_adoption': outputs.summary['adoption_fraction'],
            'mean_belief': outputs.summary['mean_belief'],
            'belief_variance': outputs.summary.get('belief_variance', 0),
            'total_cascades': outputs.cascade_stats.get('total_cascades', 0),
            'max_cascade_size': outputs.cascade_stats.get('max_cascade_size', 0),
            'r_effective_mean': np.mean(outputs.r_effective_history.get(0, [0])),
            'echo_chamber_final': outputs.echo_chamber_history[-1] if outputs.echo_chamber_history else 0,
            **param_overrides,
            'success': True,
            'error': None,
        }
    except Exception as e:
        return {
            'world': world,
            'seed': seed,
            'success': False,
            'error': str(e),
            **param_overrides,
        }


def run_baseline_validation(n_seeds=50, n_agents=1000, n_steps=30):
    """Run extensive baseline validation."""
    print(f"\n{'='*60}")
    print(f"BASELINE VALIDATION ({n_seeds} seeds)")
    print(f"{'='*60}")

    results = []
    for seed in range(n_seeds):
        result = run_single_simulation('world_baseline', seed, n_agents, n_steps, {})
        results.append(result)
        if result['success']:
            print(f"  Seed {seed}: {result['final_adoption']*100:.1f}%")
        else:
            print(f"  Seed {seed}: ERROR - {result['error']}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "baseline_50seeds.csv", index=False)

    # Statistics
    successful = df[df['success'] == True]
    if len(successful) > 0:
        adoptions = successful['final_adoption'].values

        mean_a = np.mean(adoptions)
        std_a = np.std(adoptions, ddof=1)
        sem = std_a / np.sqrt(len(adoptions))
        ci_95 = stats.t.interval(0.95, len(adoptions)-1, loc=mean_a, scale=sem)

        print(f"\nResults:")
        print(f"  Mean: {mean_a*100:.2f}%")
        print(f"  Std: {std_a*100:.2f}%")
        print(f"  95% CI: [{ci_95[0]*100:.2f}%, {ci_95[1]*100:.2f}%]")
        print(f"  In target [20-40%]: {np.mean((adoptions >= 0.2) & (adoptions <= 0.4))*100:.0f}%")
    else:
        print("\nNo successful runs!")

    return df


def run_world_comparison(n_seeds=20, n_agents=1000, n_steps=30):
    """Run validation across all world configurations."""
    print(f"\n{'='*60}")
    print(f"WORLD COMPARISON ({n_seeds} seeds each)")
    print(f"{'='*60}")

    worlds = [
        'world_baseline',
        'world_high_trust_gov',
        'world_low_trust_gov',
        'world_strong_moderation',
        'world_collapsed_local_media',
        'world_high_religion_hub',
        'world_outrage_algorithm',
    ]

    results = []
    for world in worlds:
        print(f"\n  {world}:")
        for seed in range(n_seeds):
            result = run_single_simulation(world, seed, n_agents, n_steps, {})
            results.append(result)
            if result['success']:
                print(f"    Seed {seed}: {result['final_adoption']*100:.1f}%")
            else:
                print(f"    Seed {seed}: ERROR")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "world_comparison.csv", index=False)

    # Summary by world
    print("\n\nResults by world:")
    summary = []
    for world in worlds:
        world_df = df[(df['world'] == world) & (df['success'] == True)]
        if len(world_df) > 0:
            adoptions = world_df['final_adoption'].values
            mean_a = np.mean(adoptions)
            std_a = np.std(adoptions, ddof=1)
            sem = std_a / np.sqrt(len(adoptions))
            ci_95 = stats.t.interval(0.95, len(adoptions)-1, loc=mean_a, scale=sem)

            print(f"  {world}: {mean_a*100:.1f}% ± {std_a*100:.1f}% (95% CI: [{ci_95[0]*100:.1f}%, {ci_95[1]*100:.1f}%])")
            summary.append({
                'world': world,
                'mean': mean_a,
                'std': std_a,
                'ci_lower': ci_95[0],
                'ci_upper': ci_95[1],
                'n': len(adoptions),
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "world_summary.csv", index=False)

    return df


def run_parameter_sweep(param_name, values, n_seeds=5, n_agents=1000, n_steps=30):
    """Run parameter sweep with fine granularity."""
    print(f"\n{'='*60}")
    print(f"PARAMETER SWEEP: {param_name}")
    print(f"{'='*60}")

    results = []
    for value in values:
        print(f"\n  {param_name}={value:.3f}:")
        for seed in range(n_seeds):
            result = run_single_simulation('world_baseline', seed, n_agents, n_steps, {param_name: value})
            results.append(result)
            if result['success']:
                print(f"    Seed {seed}: {result['final_adoption']*100:.1f}%")

    df = pd.DataFrame(results)
    safe_name = param_name.replace('.', '_')
    df.to_csv(OUTPUT_DIR / f"sweep_{safe_name}.csv", index=False)

    # Summary
    print("\n\nSummary:")
    for value in values:
        value_df = df[(df[param_name] == value) & (df['success'] == True)]
        if len(value_df) > 0:
            adoptions = value_df['final_adoption'].values
            mean_a = np.mean(adoptions)
            std_a = np.std(adoptions, ddof=1)
            print(f"  {param_name}={value:.3f}: {mean_a*100:.1f}% ± {std_a*100:.1f}%")

    return df


def run_intervention_comparison(n_seeds=15, n_agents=1000, n_steps=30):
    """Compare multiple intervention scenarios."""
    print(f"\n{'='*60}")
    print(f"INTERVENTION COMPARISON ({n_seeds} seeds each)")
    print(f"{'='*60}")

    scenarios = [
        ('No intervention', {}),
        ('Prebunking 10%', {'prebunking_day': 0, 'prebunking_fraction': 0.1}),
        ('Prebunking 30%', {'prebunking_day': 0, 'prebunking_fraction': 0.3}),
        ('Prebunking 50%', {'prebunking_day': 0, 'prebunking_fraction': 0.5}),
        ('Late prebunking (day 10)', {'prebunking_day': 10, 'prebunking_fraction': 0.3}),
        ('Late prebunking (day 20)', {'prebunking_day': 20, 'prebunking_fraction': 0.3}),
    ]

    results = []
    for scenario_name, overrides in scenarios:
        print(f"\n  {scenario_name}:")
        for seed in range(n_seeds):
            result = run_single_simulation('world_baseline', seed, n_agents, n_steps,
                                          {**overrides, 'scenario': scenario_name})
            results.append(result)
            if result['success']:
                print(f"    Seed {seed}: {result['final_adoption']*100:.1f}%")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "intervention_comparison.csv", index=False)

    # Summary
    print("\n\nSummary:")
    baseline_adoptions = None
    baseline_mean = None
    for scenario_name, _ in scenarios:
        scenario_df = df[(df['scenario'] == scenario_name) & (df['success'] == True)]
        if len(scenario_df) > 0:
            adoptions = scenario_df['final_adoption'].values
            mean_a = np.mean(adoptions)
            std_a = np.std(adoptions, ddof=1)

            if baseline_adoptions is None:
                baseline_adoptions = adoptions
                baseline_mean = mean_a
                print(f"  {scenario_name}: {mean_a*100:.1f}% ± {std_a*100:.1f}%")
            else:
                reduction = (baseline_mean - mean_a) / baseline_mean * 100
                # Effect size
                pooled_std = np.sqrt((np.std(baseline_adoptions)**2 + std_a**2) / 2)
                cohens_d = (baseline_mean - mean_a) / pooled_std if pooled_std > 0 else 0
                print(f"  {scenario_name}: {mean_a*100:.1f}% ± {std_a*100:.1f}% (↓{reduction:.0f}%, d={cohens_d:.2f})")

    return df


def main():
    """Run all validation suites."""
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Baseline validation (50 seeds)
    baseline_df = run_baseline_validation(n_seeds=50)

    # 2. World comparison (20 seeds each)
    world_df = run_world_comparison(n_seeds=20)

    # 3. Parameter sweeps (fine-grained)
    sweep_results = {}

    # Social proof weight
    sweep_results['social_proof'] = run_parameter_sweep(
        'advanced_belief.social_proof_weight',
        np.linspace(0.10, 0.35, 10).tolist(),
        n_seeds=5
    )

    # Learning rate
    sweep_results['learning_rate'] = run_parameter_sweep(
        'advanced_belief.base_learning_rate',
        np.linspace(0.05, 0.25, 10).tolist(),
        n_seeds=5
    )

    # Skepticism dampening
    sweep_results['skepticism'] = run_parameter_sweep(
        'advanced_belief.skepticism_dampening',
        np.linspace(0.1, 0.7, 10).tolist(),
        n_seeds=5
    )

    # 4. Intervention comparison
    intervention_df = run_intervention_comparison(n_seeds=15)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
