"""
Network and Echo Chamber Analysis
=================================
Analyzes network structure evolution and echo chamber formation.
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


def run_network_analysis(world, seed, n_agents=1000, n_steps=30):
    """Run simulation and extract network evolution data."""
    base_cfg = load_config(f'configs/{world}.yaml')
    base_cfg.sim.n_agents = n_agents
    base_cfg.sim.steps = n_steps
    base_cfg.sim.seed = seed
    base_cfg.sim.device = 'cpu'
    base_cfg.sim.deterministic = False

    advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = run_advanced_simulation(advanced_cfg, tmpdir)

    return outputs.network_evolution, outputs.echo_chamber_history, outputs.summary


def main():
    print("=" * 60)
    print("NETWORK & ECHO CHAMBER ANALYSIS")
    print("=" * 60)

    n_seeds = 15
    all_echo = []
    all_network = []
    summary_data = []

    for seed in range(n_seeds):
        print(f"Running seed {seed}...")
        try:
            network_df, echo_history, summary = run_network_analysis('world_baseline', seed)

            # Echo chamber history
            for day, echo_val in enumerate(echo_history):
                all_echo.append({
                    'seed': seed,
                    'day': day,
                    'echo_chamber_index': echo_val,
                })

            # Network evolution
            for _, row in network_df.iterrows():
                all_network.append({
                    'seed': seed,
                    'day': row.get('day', 0),
                    'edges_added': row.get('edges_added', 0),
                    'edges_removed': row.get('edges_removed', 0),
                    'mean_clustering': row.get('mean_clustering', 0),
                    'belief_homophily': row.get('belief_homophily', 0),
                })

            summary_data.append({
                'seed': seed,
                'final_adoption': summary['adoption_fraction'],
                'final_echo': echo_history[-1] if echo_history else 0,
            })

        except Exception as e:
            print(f"  Seed {seed} failed: {e}")

    # Save data
    echo_df = pd.DataFrame(all_echo)
    echo_df.to_csv(OUTPUT_DIR / "echo_chamber_dynamics.csv", index=False)

    network_df = pd.DataFrame(all_network)
    network_df.to_csv(OUTPUT_DIR / "network_evolution.csv", index=False)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "network_summary.csv", index=False)

    # Analysis
    print("\nEcho Chamber Formation:")
    for day in [0, 5, 10, 15, 20, 25, 29]:
        day_data = echo_df[echo_df['day'] == day]['echo_chamber_index'].values
        if len(day_data) > 0:
            print(f"  Day {day:2d}: {np.mean(day_data):.3f} ± {np.std(day_data):.3f}")

    # Correlation between echo chamber and adoption
    if len(summary_df) > 0:
        corr = np.corrcoef(summary_df['final_adoption'], summary_df['final_echo'])[0, 1]
        print(f"\nCorrelation (adoption vs echo chamber): r = {corr:.3f}")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
