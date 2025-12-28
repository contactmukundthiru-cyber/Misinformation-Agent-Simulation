"""
Publication Figure Data Generation
===================================
Generates clean data files optimized for publication figures.
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
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def generate_adoption_curve_figure():
    """Generate data for adoption curve figure."""
    print("Generating adoption curve figure data...")

    temporal = pd.read_csv(OUTPUT_DIR / "temporal_dynamics.csv")

    # Aggregate by day
    curve_data = []
    for day in sorted(temporal['day'].unique()):
        day_data = temporal[temporal['day'] == day]
        mean_a = day_data['adoption_mean'].mean()
        std_a = day_data['adoption_mean'].std()
        n = len(day_data)
        sem = std_a / np.sqrt(n)
        ci_95 = 1.96 * sem

        curve_data.append({
            'day': int(day),
            'mean': mean_a,
            'std': std_a,
            'ci_lower': mean_a - ci_95,
            'ci_upper': mean_a + ci_95,
            'n': n,
        })

    curve_df = pd.DataFrame(curve_data)
    curve_df.to_csv(FIGURE_DIR / "fig1_adoption_curve.csv", index=False)
    print(f"  Saved fig1_adoption_curve.csv")


def generate_world_comparison_figure():
    """Generate data for world comparison bar chart."""
    print("Generating world comparison figure data...")

    world = pd.read_csv(OUTPUT_DIR / "world_comparison.csv")

    worlds = [
        'world_baseline',
        'world_high_trust_gov',
        'world_low_trust_gov',
        'world_strong_moderation',
        'world_collapsed_local_media',
        'world_high_religion_hub',
        'world_outrage_algorithm',
    ]

    world_labels = [
        'Baseline',
        'High Trust Gov',
        'Low Trust Gov',
        'Strong Moderation',
        'Collapsed Media',
        'Religion Hub',
        'Outrage Algorithm',
    ]

    bar_data = []
    baseline_mean = world[world['world'] == 'world_baseline']['final_adoption'].mean()

    for w, label in zip(worlds, world_labels):
        w_data = world[world['world'] == w]
        successful = w_data[w_data['success'] == True]
        if len(successful) > 0:
            mean_a = successful['final_adoption'].mean()
            std_a = successful['final_adoption'].std()
            n = len(successful)
            sem = std_a / np.sqrt(n)
            ci_95 = 1.96 * sem
            delta = (mean_a - baseline_mean) / baseline_mean * 100

            bar_data.append({
                'world': w,
                'label': label,
                'mean': mean_a,
                'std': std_a,
                'ci_lower': mean_a - ci_95,
                'ci_upper': mean_a + ci_95,
                'delta_pct': delta,
                'n': n,
            })

    bar_df = pd.DataFrame(bar_data)
    bar_df.to_csv(FIGURE_DIR / "fig2_world_comparison.csv", index=False)
    print(f"  Saved fig2_world_comparison.csv")


def generate_intervention_figure():
    """Generate data for intervention effectiveness figure."""
    print("Generating intervention figure data...")

    interv = pd.read_csv(OUTPUT_DIR / "intervention_comparison.csv")

    scenarios = [
        ('No intervention', 'Control'),
        ('Prebunking 10%', 'Prebunk 10%'),
        ('Prebunking 30%', 'Prebunk 30%'),
        ('Prebunking 50%', 'Prebunk 50%'),
        ('Late prebunking (day 10)', 'Late (day 10)'),
        ('Late prebunking (day 20)', 'Late (day 20)'),
    ]

    baseline = interv[interv['scenario'] == 'No intervention']['final_adoption']
    baseline_mean = baseline.mean()

    interv_data = []
    for scenario, label in scenarios:
        s_data = interv[(interv['scenario'] == scenario) & (interv['success'] == True)]
        if len(s_data) > 0:
            mean_a = s_data['final_adoption'].mean()
            std_a = s_data['final_adoption'].std()
            n = len(s_data)
            sem = std_a / np.sqrt(n)
            ci_95 = 1.96 * sem
            reduction = (baseline_mean - mean_a) / baseline_mean * 100 if baseline_mean > 0 else 0

            # Effect size
            pooled_std = np.sqrt((baseline.std()**2 + std_a**2) / 2)
            cohens_d = (baseline_mean - mean_a) / pooled_std if pooled_std > 0 else 0

            interv_data.append({
                'scenario': scenario,
                'label': label,
                'mean': mean_a,
                'std': std_a,
                'ci_lower': mean_a - ci_95,
                'ci_upper': mean_a + ci_95,
                'reduction_pct': reduction,
                'cohens_d': cohens_d,
                'n': n,
            })

    interv_df = pd.DataFrame(interv_data)
    interv_df.to_csv(FIGURE_DIR / "fig3_interventions.csv", index=False)
    print(f"  Saved fig3_interventions.csv")


def generate_sensitivity_figure():
    """Generate data for parameter sensitivity figure."""
    print("Generating sensitivity figure data...")

    # Social proof weight
    spw = pd.read_csv(OUTPUT_DIR / "sweep_advanced_belief_social_proof_weight.csv")
    spw_data = []
    for val in sorted(spw['advanced_belief.social_proof_weight'].unique()):
        v_data = spw[(spw['advanced_belief.social_proof_weight'] == val) & (spw['success'] == True)]
        if len(v_data) > 0:
            spw_data.append({
                'parameter': 'social_proof_weight',
                'value': val,
                'mean': v_data['final_adoption'].mean(),
                'std': v_data['final_adoption'].std(),
            })

    # Learning rate
    lr = pd.read_csv(OUTPUT_DIR / "sweep_advanced_belief_base_learning_rate.csv")
    lr_data = []
    for val in sorted(lr['advanced_belief.base_learning_rate'].unique()):
        v_data = lr[(lr['advanced_belief.base_learning_rate'] == val) & (lr['success'] == True)]
        if len(v_data) > 0:
            lr_data.append({
                'parameter': 'base_learning_rate',
                'value': val,
                'mean': v_data['final_adoption'].mean(),
                'std': v_data['final_adoption'].std(),
            })

    # Skepticism
    sk = pd.read_csv(OUTPUT_DIR / "sweep_advanced_belief_skepticism_dampening.csv")
    sk_data = []
    for val in sorted(sk['advanced_belief.skepticism_dampening'].unique()):
        v_data = sk[(sk['advanced_belief.skepticism_dampening'] == val) & (sk['success'] == True)]
        if len(v_data) > 0:
            sk_data.append({
                'parameter': 'skepticism_dampening',
                'value': val,
                'mean': v_data['final_adoption'].mean(),
                'std': v_data['final_adoption'].std(),
            })

    all_sensitivity = pd.DataFrame(spw_data + lr_data + sk_data)
    all_sensitivity.to_csv(FIGURE_DIR / "fig4_sensitivity.csv", index=False)
    print(f"  Saved fig4_sensitivity.csv")


def generate_summary_table():
    """Generate publication summary table."""
    print("Generating summary table...")

    # Load all data
    baseline = pd.read_csv(OUTPUT_DIR / "baseline_50seeds.csv")
    world = pd.read_csv(OUTPUT_DIR / "world_summary.csv")
    interv = pd.read_csv(OUTPUT_DIR / "intervention_comparison.csv")

    summary = {
        'Baseline Adoption': f"{baseline['final_adoption'].mean()*100:.1f}% ± {baseline['final_adoption'].std()*100:.1f}%",
        '95% CI': f"[{baseline['final_adoption'].mean()*100 - 1.96*baseline['final_adoption'].std()*100/np.sqrt(50):.1f}%, "
                  f"{baseline['final_adoption'].mean()*100 + 1.96*baseline['final_adoption'].std()*100/np.sqrt(50):.1f}%]",
        'N Seeds': 50,
        'Strong Moderation Effect': f"-{(1 - world[world['world']=='world_strong_moderation']['mean'].values[0] / world[world['world']=='world_baseline']['mean'].values[0])*100:.0f}%",
        'Prebunking 30% Effect': f"-{(1 - interv[interv['scenario']=='Prebunking 30%']['final_adoption'].mean() / interv[interv['scenario']=='No intervention']['final_adoption'].mean())*100:.0f}%",
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(FIGURE_DIR / "table1_summary.csv", index=False)
    print(f"  Saved table1_summary.csv")


def main():
    print("=" * 60)
    print("PUBLICATION FIGURE DATA GENERATION")
    print("=" * 60)

    generate_adoption_curve_figure()
    generate_world_comparison_figure()
    generate_intervention_figure()
    generate_sensitivity_figure()
    generate_summary_table()

    print("\n" + "=" * 60)
    print("FIGURE DATA COMPLETE")
    print(f"Files saved to: {FIGURE_DIR}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(FIGURE_DIR.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
