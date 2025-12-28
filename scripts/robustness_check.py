"""
Robustness Checks
=================
Tests model stability across different configurations.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.config import load_config
from sim.simulation_advanced import AdvancedSimulationConfig, run_advanced_simulation

torch.use_deterministic_algorithms(False)

OUTPUT_DIR = Path(__file__).parent.parent / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_simulation(n_agents, seed, n_steps=30):
    """Run a single simulation with specified parameters."""
    base_cfg = load_config('configs/world_baseline.yaml')
    base_cfg.sim.n_agents = n_agents
    base_cfg.sim.steps = n_steps
    base_cfg.sim.seed = seed
    base_cfg.sim.device = 'cpu'
    base_cfg.sim.deterministic = False

    advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = run_advanced_simulation(advanced_cfg, tmpdir)

    return outputs.summary


def main():
    print("=" * 60)
    print("ROBUSTNESS CHECKS")
    print("=" * 60)

    results = []

    # 1. Network size robustness
    print("\n1. Network Size Robustness:")
    sizes = [500, 1000, 2000, 5000]
    n_seeds = 5

    for size in sizes:
        print(f"\n  Testing n_agents={size}...")
        adoptions = []
        for seed in range(n_seeds):
            try:
                summary = run_simulation(size, seed)
                adoptions.append(summary['adoption_fraction'])
                results.append({
                    'test': 'network_size',
                    'parameter': size,
                    'seed': seed,
                    'adoption': summary['adoption_fraction'],
                    'mean_belief': summary['mean_belief'],
                })
                print(f"    Seed {seed}: {summary['adoption_fraction']*100:.1f}%")
            except Exception as e:
                print(f"    Seed {seed}: ERROR - {e}")

        if adoptions:
            print(f"  n={size}: {np.mean(adoptions)*100:.1f}% ± {np.std(adoptions)*100:.1f}%")

    # 2. Seed range robustness (test high seed numbers)
    print("\n2. Seed Range Robustness:")
    seed_ranges = [(0, 5), (100, 105), (1000, 1005)]

    for start, end in seed_ranges:
        print(f"\n  Testing seeds {start}-{end}...")
        adoptions = []
        for seed in range(start, end):
            try:
                summary = run_simulation(1000, seed)
                adoptions.append(summary['adoption_fraction'])
                results.append({
                    'test': 'seed_range',
                    'parameter': f'{start}-{end}',
                    'seed': seed,
                    'adoption': summary['adoption_fraction'],
                    'mean_belief': summary['mean_belief'],
                })
            except Exception as e:
                print(f"    Seed {seed}: ERROR - {e}")

        if adoptions:
            print(f"  Seeds {start}-{end}: {np.mean(adoptions)*100:.1f}% ± {np.std(adoptions)*100:.1f}%")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "robustness_checks.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)

    # Network size summary
    print("\nNetwork Size Effect:")
    for size in sizes:
        size_data = results_df[(results_df['test'] == 'network_size') & (results_df['parameter'] == size)]
        if len(size_data) > 0:
            print(f"  n={size}: {size_data['adoption'].mean()*100:.1f}% ± {size_data['adoption'].std()*100:.1f}%")

    # Check if results are stable across sizes
    size_means = [results_df[(results_df['test'] == 'network_size') & (results_df['parameter'] == s)]['adoption'].mean()
                  for s in sizes]
    size_range = max(size_means) - min(size_means) if size_means else 0
    print(f"\n  Max variation across sizes: {size_range*100:.1f}%")
    if size_range < 0.05:
        print("  [OK] Model is robust to network size")
    else:
        print("  [WARNING] Model shows sensitivity to network size")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
