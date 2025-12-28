"""
Long-term Dynamics Validation (365 days)
=========================================
Tests whether the model produces stable long-term behavior.
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


def main():
    print("=" * 60)
    print("LONG-TERM DYNAMICS VALIDATION (365 days)")
    print("=" * 60)

    n_seeds = 10
    results = []

    for seed in range(n_seeds):
        print(f"Running seed {seed} (365 days)...")
        try:
            base_cfg = load_config('configs/world_baseline.yaml')
            base_cfg.sim.n_agents = 1000
            base_cfg.sim.steps = 365
            base_cfg.sim.seed = seed
            base_cfg.sim.device = 'cpu'
            base_cfg.sim.deterministic = False
            base_cfg.sim.snapshot_interval = 7  # Weekly snapshots

            advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

            with tempfile.TemporaryDirectory() as tmpdir:
                outputs = run_advanced_simulation(advanced_cfg, tmpdir)

            # Extract key metrics at different timepoints
            metrics_df = outputs.metrics
            claim0 = metrics_df[metrics_df['claim'] == 0]

            # Get adoption at key timepoints
            timepoints = [30, 60, 90, 180, 365]
            for tp in timepoints:
                tp_data = claim0[claim0['day'] == tp - 1]  # 0-indexed
                if len(tp_data) > 0:
                    results.append({
                        'seed': seed,
                        'day': tp,
                        'adoption': tp_data['adoption_fraction'].values[0],
                        'mean_belief': tp_data['mean_belief'].values[0],
                        'r_effective': tp_data['r_effective'].values[0],
                    })

            print(f"  Final adoption: {outputs.summary['adoption_fraction']*100:.1f}%")

        except Exception as e:
            print(f"  Seed {seed} failed: {e}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "longterm_dynamics.csv", index=False)

    # Summary
    print("\nLong-term Adoption Summary:")
    for tp in [30, 60, 90, 180, 365]:
        tp_data = results_df[results_df['day'] == tp]['adoption']
        if len(tp_data) > 0:
            print(f"  Day {tp:3d}: {tp_data.mean()*100:.1f}% ± {tp_data.std()*100:.1f}%")

    # Check for stability (does adoption plateau?)
    print("\nStability Check:")
    day90 = results_df[results_df['day'] == 90]['adoption'].mean()
    day365 = results_df[results_df['day'] == 365]['adoption'].mean()
    change = abs(day365 - day90) / day90 * 100 if day90 > 0 else 0
    print(f"  Change from day 90 to 365: {change:.1f}%")
    if change < 10:
        print("  [OK] Model shows stable equilibrium")
    else:
        print("  [WARNING] Model still evolving at day 365")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
