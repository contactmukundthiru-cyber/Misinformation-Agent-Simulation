"""
Temporal Dynamics Analysis
==========================
Captures day-by-day adoption curves and cascade patterns.
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


def run_with_daily_tracking(world, seed, n_agents=1000, n_steps=30):
    """Run simulation and extract daily metrics."""
    base_cfg = load_config(f'configs/{world}.yaml')
    base_cfg.sim.n_agents = n_agents
    base_cfg.sim.steps = n_steps
    base_cfg.sim.seed = seed
    base_cfg.sim.device = 'cpu'
    base_cfg.sim.deterministic = False
    base_cfg.sim.snapshot_interval = 1  # Daily snapshots

    advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = run_advanced_simulation(advanced_cfg, tmpdir)

    return outputs.metrics, outputs.r_effective_history, outputs.echo_chamber_history


def main():
    print("=" * 60)
    print("TEMPORAL DYNAMICS ANALYSIS")
    print("=" * 60)

    # Run multiple seeds and collect daily data
    n_seeds = 20
    all_daily = []

    for seed in range(n_seeds):
        print(f"Running seed {seed}...")
        try:
            metrics_df, r_eff, echo = run_with_daily_tracking('world_baseline', seed)

            # Extract daily metrics - average across all claims for consistency with summary
            days = metrics_df['day'].unique()

            for day in sorted(days):
                day_df = metrics_df[metrics_df['day'] == day]
                all_daily.append({
                    'seed': seed,
                    'day': int(day),
                    'adoption_mean': day_df['adoption_fraction'].mean(),  # Mean across claims
                    'adoption_claim0': day_df[day_df['claim'] == 0]['adoption_fraction'].values[0],  # Primary claim
                    'mean_belief': day_df['mean_belief'].mean(),
                    'variance': day_df['variance'].mean(),
                    'r_effective': day_df['r_effective'].mean(),
                    'echo_chamber': echo[int(day)] if int(day) < len(echo) else 0,
                })
        except Exception as e:
            print(f"  Seed {seed} failed: {e}")

    daily_df = pd.DataFrame(all_daily)
    daily_df.to_csv(OUTPUT_DIR / "temporal_dynamics.csv", index=False)

    # Compute mean curves with confidence bands
    print("\nDaily Adoption - Mean Across Claims (consistent with summary):")
    summary = []
    max_day = int(daily_df['day'].max()) + 1
    for day in range(max_day):
        day_data = daily_df[daily_df['day'] == day]
        if len(day_data) > 0:
            mean_a = day_data['adoption_mean'].mean()
            std_a = day_data['adoption_mean'].std()
            mean_c0 = day_data['adoption_claim0'].mean()
            ci_lower = mean_a - 1.96 * std_a / np.sqrt(len(day_data))
            ci_upper = mean_a + 1.96 * std_a / np.sqrt(len(day_data))
            print(f"  Day {day:2d}: {mean_a*100:5.1f}% ± {std_a*100:4.1f}% (claim0: {mean_c0*100:.1f}%)")
            summary.append({
                'day': day,
                'mean_adoption': mean_a,
                'std_adoption': std_a,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'claim0_adoption': mean_c0,
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "adoption_curve.csv", index=False)

    # Identify key temporal features
    print("\nKey Temporal Features (using mean adoption across claims):")

    # Time to 10% adoption
    time_to_10 = []
    for seed in daily_df['seed'].unique():
        seed_data = daily_df[daily_df['seed'] == seed].sort_values('day')
        crossed = seed_data[seed_data['adoption_mean'] >= 0.1]
        if len(crossed) > 0:
            time_to_10.append(crossed['day'].iloc[0])
    if time_to_10:
        print(f"  Time to 10% adoption: {np.mean(time_to_10):.1f} ± {np.std(time_to_10):.1f} days")
    else:
        print(f"  Time to 10% adoption: Not reached in 30 days")

    # Time to 20% adoption
    time_to_20 = []
    for seed in daily_df['seed'].unique():
        seed_data = daily_df[daily_df['seed'] == seed].sort_values('day')
        crossed = seed_data[seed_data['adoption_mean'] >= 0.2]
        if len(crossed) > 0:
            time_to_20.append(crossed['day'].iloc[0])
    if time_to_20:
        print(f"  Time to 20% adoption: {np.mean(time_to_20):.1f} ± {np.std(time_to_20):.1f} days")
    else:
        print(f"  Time to 20% adoption: Not reached in 30 days")

    # Peak growth rate (derivative of adoption)
    peak_growth = []
    for seed in daily_df['seed'].unique():
        seed_data = daily_df[daily_df['seed'] == seed].sort_values('day')
        adoptions = seed_data['adoption_mean'].values
        if len(adoptions) > 1:
            growth = np.diff(adoptions)
            peak_growth.append(np.max(growth))
    if peak_growth:
        print(f"  Peak daily growth: {np.mean(peak_growth)*100:.2f}% ± {np.std(peak_growth)*100:.2f}%")

    # Plateau timing (when growth drops below 0.5%)
    plateau_times = []
    for seed in daily_df['seed'].unique():
        seed_data = daily_df[daily_df['seed'] == seed].sort_values('day')
        adoptions = seed_data['adoption_mean'].values
        if len(adoptions) > 5:
            growth = np.diff(adoptions)
            # Find first day after day 5 where growth < 0.005
            for i in range(5, len(growth)):
                if growth[i] < 0.005:
                    plateau_times.append(i + 1)
                    break
    if plateau_times:
        print(f"  Plateau onset: {np.mean(plateau_times):.1f} ± {np.std(plateau_times):.1f} days")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
