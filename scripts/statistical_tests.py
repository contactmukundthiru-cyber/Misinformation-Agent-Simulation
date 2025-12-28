"""
Statistical Tests for Publication
=================================
Comprehensive statistical analysis for Nature submission.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent.parent / "validation_results"
STATS_DIR = OUTPUT_DIR / "statistics"
STATS_DIR.mkdir(exist_ok=True)


def welch_ttest(group1, group2):
    """Perform Welch's t-test (unequal variances)."""
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    # Cohen's d
    pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
    return t_stat, p_val, cohens_d


def run_world_anova():
    """Run one-way ANOVA across world configurations."""
    print("\n" + "=" * 60)
    print("WORLD COMPARISON: One-Way ANOVA")
    print("=" * 60)

    world = pd.read_csv(OUTPUT_DIR / "world_comparison.csv")
    successful = world[world['success'] == True]

    groups = []
    world_names = []
    for w in successful['world'].unique():
        adoptions = successful[successful['world'] == w]['final_adoption'].values
        groups.append(adoptions)
        world_names.append(w)

    # One-way ANOVA
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nOne-Way ANOVA:")
    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  p-value: {p_val:.2e}")

    # Effect size (eta-squared)
    ss_between = sum(len(g) * (np.mean(g) - np.mean(np.concatenate(groups)))**2 for g in groups)
    ss_total = sum((x - np.mean(np.concatenate(groups)))**2 for g in groups for x in g)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    print(f"  Eta-squared (η²): {eta_squared:.3f}")

    # Post-hoc: Tukey HSD-style pairwise comparisons
    print("\nPost-hoc Pairwise Comparisons (Bonferroni-corrected):")
    results = []
    n_comparisons = len(world_names) * (len(world_names) - 1) // 2

    for i, w1 in enumerate(world_names):
        for j, w2 in enumerate(world_names):
            if i < j:
                g1 = successful[successful['world'] == w1]['final_adoption'].values
                g2 = successful[successful['world'] == w2]['final_adoption'].values
                t_stat, p_val, d = welch_ttest(g1, g2)
                p_adj = min(p_val * n_comparisons, 1.0)  # Bonferroni
                sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""

                results.append({
                    'comparison': f"{w1} vs {w2}",
                    'mean1': np.mean(g1),
                    'mean2': np.mean(g2),
                    'difference': np.mean(g1) - np.mean(g2),
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'p_adjusted': p_adj,
                    'cohens_d': d,
                    'significant': sig
                })

                if p_adj < 0.05:
                    print(f"  {w1.replace('world_', '')} vs {w2.replace('world_', '')}: "
                          f"Δ={np.mean(g1)-np.mean(g2):.3f}, d={d:.2f}, p={p_adj:.2e}{sig}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(STATS_DIR / "world_pairwise_tests.csv", index=False)

    return f_stat, p_val, eta_squared


def run_intervention_tests():
    """Run statistical tests for intervention effectiveness."""
    print("\n" + "=" * 60)
    print("INTERVENTION EFFECTIVENESS: Statistical Tests")
    print("=" * 60)

    interv = pd.read_csv(OUTPUT_DIR / "intervention_comparison.csv")
    successful = interv[interv['success'] == True]

    baseline = successful[successful['scenario'] == 'No intervention']['final_adoption'].values

    results = []
    for scenario in successful['scenario'].unique():
        if scenario == 'No intervention':
            continue

        treatment = successful[successful['scenario'] == scenario]['final_adoption'].values
        t_stat, p_val, d = welch_ttest(baseline, treatment)

        # Reduction percentage
        reduction = (np.mean(baseline) - np.mean(treatment)) / np.mean(baseline) * 100

        results.append({
            'scenario': scenario,
            'baseline_mean': np.mean(baseline),
            'treatment_mean': np.mean(treatment),
            'reduction_pct': reduction,
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': d,
            'n_baseline': len(baseline),
            'n_treatment': len(treatment),
        })

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"\n  {scenario}:")
        print(f"    Mean: {np.mean(treatment)*100:.1f}% (baseline: {np.mean(baseline)*100:.1f}%)")
        print(f"    Reduction: {reduction:.1f}%")
        print(f"    t = {t_stat:.3f}, p = {p_val:.2e}{sig}")
        print(f"    Cohen's d = {d:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(STATS_DIR / "intervention_tests.csv", index=False)

    return results_df


def run_sensitivity_regression():
    """Run regression analysis on parameter sensitivities."""
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY: Regression Analysis")
    print("=" * 60)

    results = []

    # Social proof weight
    try:
        spw = pd.read_csv(OUTPUT_DIR / "sweep_advanced_belief_social_proof_weight.csv")
        spw_success = spw[spw['success'] == True]
        x = spw_success['advanced_belief.social_proof_weight'].values
        y = spw_success['final_adoption'].values
        slope, intercept, r, p, se = stats.linregress(x, y)

        results.append({
            'parameter': 'social_proof_weight',
            'slope': slope,
            'intercept': intercept,
            'r_squared': r**2,
            'p_value': p,
            'std_error': se,
        })

        print(f"\n  Social Proof Weight:")
        print(f"    Slope: {slope:.3f} (adoption per unit increase)")
        print(f"    R²: {r**2:.3f}")
        print(f"    p-value: {p:.2e}")
    except Exception as e:
        print(f"  Social proof sweep not available: {e}")

    # Learning rate
    try:
        lr = pd.read_csv(OUTPUT_DIR / "sweep_advanced_belief_base_learning_rate.csv")
        lr_success = lr[lr['success'] == True]
        x = lr_success['advanced_belief.base_learning_rate'].values
        y = lr_success['final_adoption'].values
        slope, intercept, r, p, se = stats.linregress(x, y)

        results.append({
            'parameter': 'base_learning_rate',
            'slope': slope,
            'intercept': intercept,
            'r_squared': r**2,
            'p_value': p,
            'std_error': se,
        })

        print(f"\n  Base Learning Rate:")
        print(f"    Slope: {slope:.3f}")
        print(f"    R²: {r**2:.3f}")
        print(f"    p-value: {p:.2e}")
    except Exception as e:
        print(f"  Learning rate sweep not available: {e}")

    # Skepticism dampening
    try:
        sk = pd.read_csv(OUTPUT_DIR / "sweep_advanced_belief_skepticism_dampening.csv")
        sk_success = sk[sk['success'] == True]
        x = sk_success['advanced_belief.skepticism_dampening'].values
        y = sk_success['final_adoption'].values
        slope, intercept, r, p, se = stats.linregress(x, y)

        results.append({
            'parameter': 'skepticism_dampening',
            'slope': slope,
            'intercept': intercept,
            'r_squared': r**2,
            'p_value': p,
            'std_error': se,
        })

        print(f"\n  Skepticism Dampening:")
        print(f"    Slope: {slope:.3f}")
        print(f"    R²: {r**2:.3f}")
        print(f"    p-value: {p:.2e}")
    except Exception as e:
        print(f"  Skepticism sweep not available: {e}")

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(STATS_DIR / "sensitivity_regression.csv", index=False)

    return results


def run_power_analysis():
    """Estimate statistical power of comparisons."""
    print("\n" + "=" * 60)
    print("POWER ANALYSIS")
    print("=" * 60)

    # For baseline validation
    baseline = pd.read_csv(OUTPUT_DIR / "baseline_50seeds.csv")
    successful = baseline[baseline['success'] == True]
    adoptions = successful['final_adoption'].values

    n = len(adoptions)
    std = np.std(adoptions, ddof=1)

    # Power to detect various effect sizes (using approximation)
    print(f"\n  Sample size: n = {n}")
    print(f"  Standard deviation: σ = {std:.4f}")

    # Detectable effect at 80% power, α = 0.05
    # d = t_crit * sqrt(2/n) approximately
    t_crit = stats.t.ppf(0.975, n-1)
    min_detectable_d = t_crit * np.sqrt(2/n) * 1.4  # Approximate correction
    min_detectable_diff = min_detectable_d * std

    print(f"\n  Minimum detectable effect (80% power, α=0.05):")
    print(f"    Cohen's d = {min_detectable_d:.2f}")
    print(f"    Absolute difference = {min_detectable_diff*100:.2f}%")

    # World comparison power
    world = pd.read_csv(OUTPUT_DIR / "world_comparison.csv")
    world_success = world[world['success'] == True]
    n_per_world = world_success.groupby('world').size().min()

    print(f"\n  World comparison (n = {n_per_world} per world):")
    min_d_world = t_crit * np.sqrt(2/n_per_world) * 1.4
    print(f"    Minimum detectable d = {min_d_world:.2f}")


def run_normality_tests():
    """Test for normality of distributions."""
    print("\n" + "=" * 60)
    print("NORMALITY TESTS (Shapiro-Wilk)")
    print("=" * 60)

    results = []

    # Baseline
    baseline = pd.read_csv(OUTPUT_DIR / "baseline_50seeds.csv")
    successful = baseline[baseline['success'] == True]
    adoptions = successful['final_adoption'].values

    w_stat, p_val = stats.shapiro(adoptions)
    normal = "Yes" if p_val > 0.05 else "No"
    print(f"\n  Baseline adoption (n={len(adoptions)}):")
    print(f"    W = {w_stat:.4f}, p = {p_val:.3f}")
    print(f"    Normal distribution: {normal}")

    results.append({
        'distribution': 'baseline_adoption',
        'n': len(adoptions),
        'w_statistic': w_stat,
        'p_value': p_val,
        'normal': normal,
    })

    # World configurations
    world = pd.read_csv(OUTPUT_DIR / "world_comparison.csv")
    world_success = world[world['success'] == True]

    for w in world_success['world'].unique():
        adoptions = world_success[world_success['world'] == w]['final_adoption'].values
        if len(adoptions) >= 8:  # Shapiro-Wilk requires n >= 3, but more is better
            w_stat, p_val = stats.shapiro(adoptions)
            normal = "Yes" if p_val > 0.05 else "No"
            results.append({
                'distribution': w,
                'n': len(adoptions),
                'w_statistic': w_stat,
                'p_value': p_val,
                'normal': normal,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(STATS_DIR / "normality_tests.csv", index=False)

    return results_df


def generate_summary_stats():
    """Generate comprehensive summary statistics."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("=" * 60)

    summary = {}

    # Baseline
    baseline = pd.read_csv(OUTPUT_DIR / "baseline_50seeds.csv")
    successful = baseline[baseline['success'] == True]
    adoptions = successful['final_adoption'].values

    summary['baseline'] = {
        'n': len(adoptions),
        'mean': np.mean(adoptions),
        'std': np.std(adoptions, ddof=1),
        'sem': np.std(adoptions, ddof=1) / np.sqrt(len(adoptions)),
        'median': np.median(adoptions),
        'iqr': np.percentile(adoptions, 75) - np.percentile(adoptions, 25),
        'min': np.min(adoptions),
        'max': np.max(adoptions),
        'skewness': stats.skew(adoptions),
        'kurtosis': stats.kurtosis(adoptions),
    }

    print(f"\nBaseline (n={summary['baseline']['n']}):")
    print(f"  Mean ± SEM: {summary['baseline']['mean']*100:.2f}% ± {summary['baseline']['sem']*100:.2f}%")
    print(f"  Median (IQR): {summary['baseline']['median']*100:.2f}% ({summary['baseline']['iqr']*100:.2f}%)")
    print(f"  Range: [{summary['baseline']['min']*100:.2f}%, {summary['baseline']['max']*100:.2f}%]")
    print(f"  Skewness: {summary['baseline']['skewness']:.3f}")
    print(f"  Kurtosis: {summary['baseline']['kurtosis']:.3f}")

    # Export
    summary_df = pd.DataFrame([summary['baseline']])
    summary_df.to_csv(STATS_DIR / "baseline_summary_stats.csv", index=False)

    return summary


def main():
    print("=" * 80)
    print("STATISTICAL ANALYSIS FOR NATURE SUBMISSION")
    print("=" * 80)

    # Run all tests
    run_world_anova()
    run_intervention_tests()
    run_sensitivity_regression()
    run_power_analysis()
    run_normality_tests()
    generate_summary_stats()

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print(f"Results saved to: {STATS_DIR}")
    print("=" * 80)

    # List files
    print("\nGenerated files:")
    for f in sorted(STATS_DIR.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
