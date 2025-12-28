"""
Reproducibility Testing
=======================
Tests deterministic execution and cross-run consistency.
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

OUTPUT_DIR = Path(__file__).parent.parent / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_deterministic_test():
    """Test that same seed produces identical results."""
    print("\n" + "=" * 60)
    print("DETERMINISTIC EXECUTION TEST")
    print("=" * 60)

    seed = 42
    results = []

    for run in range(3):
        # Force deterministic
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        base_cfg = load_config('configs/world_baseline.yaml')
        base_cfg.sim.n_agents = 500
        base_cfg.sim.steps = 15
        base_cfg.sim.seed = seed
        base_cfg.sim.device = 'cpu'
        base_cfg.sim.deterministic = True

        advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = run_advanced_simulation(advanced_cfg, tmpdir)

        adoption = outputs.summary['adoption_fraction']
        mean_belief = outputs.summary['mean_belief']
        results.append({
            'run': run,
            'adoption': adoption,
            'mean_belief': mean_belief,
        })
        print(f"  Run {run}: adoption={adoption:.6f}, mean_belief={mean_belief:.6f}")

    # Check consistency
    adoptions = [r['adoption'] for r in results]
    if len(set(adoptions)) == 1:
        print("\n  ✓ PASS: All runs produced identical results")
        status = "PASS"
    else:
        diff = max(adoptions) - min(adoptions)
        print(f"\n  ✗ FAIL: Runs differ by {diff:.6f}")
        status = "FAIL"

    # Disable deterministic for speed
    torch.use_deterministic_algorithms(False)

    return status, results


def run_cross_seed_consistency():
    """Test that nearby seeds produce similar distributions."""
    print("\n" + "=" * 60)
    print("CROSS-SEED CONSISTENCY TEST")
    print("=" * 60)

    # Run with seeds 0-9 and 100-109
    torch.use_deterministic_algorithms(False)

    groups = {
        'seeds_0_9': list(range(10)),
        'seeds_100_109': list(range(100, 110)),
    }

    results = {}
    for group_name, seeds in groups.items():
        adoptions = []
        for seed in seeds:
            base_cfg = load_config('configs/world_baseline.yaml')
            base_cfg.sim.n_agents = 1000
            base_cfg.sim.steps = 30
            base_cfg.sim.seed = seed
            base_cfg.sim.device = 'cpu'
            base_cfg.sim.deterministic = False

            advanced_cfg = AdvancedSimulationConfig(base=base_cfg)

            with tempfile.TemporaryDirectory() as tmpdir:
                outputs = run_advanced_simulation(advanced_cfg, tmpdir)

            adoption = outputs.summary['adoption_fraction']
            adoptions.append(adoption)
            print(f"  {group_name} seed {seed}: {adoption*100:.1f}%")

        results[group_name] = {
            'mean': np.mean(adoptions),
            'std': np.std(adoptions),
            'min': np.min(adoptions),
            'max': np.max(adoptions),
        }

    # Compare groups
    print("\nGroup comparison:")
    for group_name, stats in results.items():
        print(f"  {group_name}: {stats['mean']*100:.1f}% ± {stats['std']*100:.1f}%")

    # Check if means are similar (within 2 standard deviations)
    mean_diff = abs(results['seeds_0_9']['mean'] - results['seeds_100_109']['mean'])
    pooled_std = np.sqrt((results['seeds_0_9']['std']**2 + results['seeds_100_109']['std']**2) / 2)

    if mean_diff < 2 * pooled_std:
        print(f"\n  ✓ PASS: Group means within 2σ (diff={mean_diff*100:.2f}%, threshold={2*pooled_std*100:.2f}%)")
        status = "PASS"
    else:
        print(f"\n  ✗ FAIL: Group means differ significantly")
        status = "FAIL"

    return status, results


def run_version_info():
    """Capture version information for reproducibility."""
    print("\n" + "=" * 60)
    print("VERSION INFORMATION")
    print("=" * 60)

    import platform
    versions = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'platform': platform.platform(),
    }

    for k, v in versions.items():
        print(f"  {k}: {v}")

    return versions


def main():
    print("=" * 80)
    print("REPRODUCIBILITY TESTING")
    print("=" * 80)

    versions = run_version_info()

    det_status, det_results = run_deterministic_test()
    seed_status, seed_results = run_cross_seed_consistency()

    # Summary
    print("\n" + "=" * 80)
    print("REPRODUCIBILITY SUMMARY")
    print("=" * 80)
    print(f"  Deterministic execution: {det_status}")
    print(f"  Cross-seed consistency: {seed_status}")

    overall = "PASS" if det_status == "PASS" and seed_status == "PASS" else "FAIL"
    print(f"\n  OVERALL: {overall}")

    # Save results
    summary = {
        'deterministic_test': det_status,
        'cross_seed_test': seed_status,
        'overall': overall,
        **versions,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_DIR / "reproducibility_test.csv", index=False)

    print(f"\nResults saved to {OUTPUT_DIR}/reproducibility_test.csv")


if __name__ == "__main__":
    main()
