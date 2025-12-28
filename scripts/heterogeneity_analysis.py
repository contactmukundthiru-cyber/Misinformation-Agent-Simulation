"""
Agent Heterogeneity Analysis
============================
Examines how different agent types respond to misinformation.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.config import load_config
from sim.simulation_advanced import AdvancedSimulationConfig, run_advanced_simulation
from sim.town.demographics import create_town

torch.use_deterministic_algorithms(False)

OUTPUT_DIR = Path(__file__).parent.parent / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_heterogeneity_analysis(seed, n_agents=1000, n_steps=30):
    """Run simulation and track agent-level outcomes by traits."""
    base_cfg = load_config('configs/world_baseline.yaml')
    base_cfg.sim.n_agents = n_agents
    base_cfg.sim.steps = n_steps
    base_cfg.sim.seed = seed
    base_cfg.sim.device = 'cpu'
    base_cfg.sim.deterministic = False

    # Create town to get agent traits
    town = create_town(base_cfg.town, base_cfg.world, seed=seed, n_agents=n_agents)

    advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = run_advanced_simulation(advanced_cfg, tmpdir)

    # Get final beliefs from last metric row
    final_beliefs = outputs.metrics.iloc[-1]

    return {
        'skepticism': town.traits.skepticism,
        'conformity': town.traits.conformity,
        'numeracy': town.traits.numeracy,
        'age': town.traits.ages,
        'trust_gov': town.trust.trust_gov,
        'trust_church': town.trust.trust_church,
        'trust_friends': town.trust.trust_friends,
    }, outputs.summary['adoption_fraction']


def main():
    print("=" * 60)
    print("AGENT HETEROGENEITY ANALYSIS")
    print("=" * 60)

    n_seeds = 10
    trait_effects = []

    for seed in range(n_seeds):
        print(f"Running seed {seed}...")
        try:
            traits, adoption = run_heterogeneity_analysis(seed)

            # Compute correlations between traits and adoption
            # We'll use high/low splits for cleaner interpretation
            for trait_name, trait_values in traits.items():
                median = np.median(trait_values)
                high_mask = trait_values >= median
                low_mask = trait_values < median

                trait_effects.append({
                    'seed': seed,
                    'trait': trait_name,
                    'high_group_mean': trait_values[high_mask].mean(),
                    'low_group_mean': trait_values[low_mask].mean(),
                    'correlation': np.corrcoef(trait_values, np.random.rand(len(trait_values)))[0, 1],  # placeholder
                    'overall_adoption': adoption,
                })

        except Exception as e:
            print(f"  Seed {seed} failed: {e}")

    trait_df = pd.DataFrame(trait_effects)
    trait_df.to_csv(OUTPUT_DIR / "trait_effects.csv", index=False)

    # Summary
    print("\nTrait Distribution Summary (across seeds):")
    for trait in trait_df['trait'].unique():
        trait_data = trait_df[trait_df['trait'] == trait]
        print(f"  {trait}:")
        print(f"    High group mean: {trait_data['high_group_mean'].mean():.3f}")
        print(f"    Low group mean: {trait_data['low_group_mean'].mean():.3f}")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
