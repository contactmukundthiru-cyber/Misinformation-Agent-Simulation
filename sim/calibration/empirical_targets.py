"""
Empirical Calibration Targets
==============================
Defines target statistics from misinformation research literature.

These stylized facts come from empirical studies of misinformation spread
and should be matched by the simulation to claim validity.

Sources:
- Vosoughi et al. (2018): False news spread patterns on Twitter
- Guess et al. (2019): Misinformation sharing demographics
- Allen et al. (2020): Evaluating the fake news problem
- Pennycook & Rand (2021): Psychology of fake news
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class EmpiricalTarget:
    """A single empirical target for calibration."""

    name: str
    description: str
    value: float
    tolerance: float  # Acceptable deviation
    weight: float  # Importance in overall distance
    source: str  # Citation


@dataclass
class EmpiricalTargets:
    """Collection of empirical targets for a domain."""

    domain: str
    targets: Dict[str, EmpiricalTarget]

    def get_target_dict(self) -> Dict[str, Tuple[float, float]]:
        """Get targets as dict of (value, tolerance) tuples."""
        return {
            name: (t.value, t.tolerance)
            for name, t in self.targets.items()
        }

    def get_weights(self) -> Dict[str, float]:
        """Get importance weights for each target."""
        return {name: t.weight for name, t in self.targets.items()}


# COVID-19 misinformation targets (based on 2020-2021 studies)
COVID_MISINFORMATION_TARGETS = EmpiricalTargets(
    domain="covid_misinformation",
    targets={
        # Final adoption rate: 20-35% of population believed at least one
        # major COVID conspiracy (Roozenbeek et al., 2020)
        "final_adoption_fraction": EmpiricalTarget(
            name="final_adoption_fraction",
            description="Fraction of population adopting misinformation",
            value=0.28,
            tolerance=0.08,
            weight=1.0,
            source="Roozenbeek et al. (2020)",
        ),

        # Time to peak: misinformation peaked around 2-4 weeks after emergence
        "days_to_peak": EmpiricalTarget(
            name="days_to_peak",
            description="Days from emergence to peak adoption",
            value=21.0,
            tolerance=7.0,
            weight=0.8,
            source="Cinelli et al. (2020)",
        ),

        # Demographic heterogeneity: 2-3x difference between groups
        "demographic_heterogeneity": EmpiricalTarget(
            name="demographic_heterogeneity",
            description="Ratio of high to low adopting demographics",
            value=2.5,
            tolerance=0.7,
            weight=0.6,
            source="Guess et al. (2019)",
        ),

        # Cascade structure: most cascades small, few viral
        "power_law_exponent": EmpiricalTarget(
            name="power_law_exponent",
            description="Exponent of cascade size power law",
            value=2.5,
            tolerance=0.4,
            weight=0.7,
            source="Vosoughi et al. (2018)",
        ),

        # False news spreads faster than true news (6x)
        "spread_rate_ratio": EmpiricalTarget(
            name="spread_rate_ratio",
            description="Spread rate of false vs true information",
            value=6.0,
            tolerance=2.0,
            weight=0.5,
            source="Vosoughi et al. (2018)",
        ),

        # Polarization increase
        "polarization_increase": EmpiricalTarget(
            name="polarization_increase",
            description="Change in polarization over epidemic",
            value=0.15,
            tolerance=0.05,
            weight=0.6,
            source="Cinelli et al. (2021)",
        ),
    },
)


# Election misinformation targets (based on 2016/2020 US elections)
ELECTION_MISINFORMATION_TARGETS = EmpiricalTargets(
    domain="election_misinformation",
    targets={
        # Exposure rate: 25% of Americans saw fake news (Guess et al.)
        "exposure_fraction": EmpiricalTarget(
            name="exposure_fraction",
            description="Fraction exposed to misinformation",
            value=0.25,
            tolerance=0.05,
            weight=1.0,
            source="Guess et al. (2018)",
        ),

        # Sharing rate: 8.5% of those exposed shared
        "share_given_exposure": EmpiricalTarget(
            name="share_given_exposure",
            description="Probability of sharing given exposure",
            value=0.085,
            tolerance=0.02,
            weight=0.9,
            source="Guess et al. (2019)",
        ),

        # Age effect: 65+ share 7x more than young
        "age_sharing_ratio": EmpiricalTarget(
            name="age_sharing_ratio",
            description="Ratio of old to young sharing rate",
            value=7.0,
            tolerance=2.0,
            weight=0.7,
            source="Guess et al. (2019)",
        ),

        # Partisan asymmetry
        "partisan_asymmetry": EmpiricalTarget(
            name="partisan_asymmetry",
            description="Ratio of conservative to liberal sharing",
            value=1.5,
            tolerance=0.3,
            weight=0.5,
            source="Osmundsen et al. (2021)",
        ),

        # Correction effectiveness: 20-30% reduction
        "correction_effectiveness": EmpiricalTarget(
            name="correction_effectiveness",
            description="Belief reduction from corrections",
            value=0.25,
            tolerance=0.08,
            weight=0.8,
            source="Walter et al. (2020) meta-analysis",
        ),
    },
)


# General misinformation targets (cross-domain patterns)
GENERAL_MISINFORMATION_TARGETS = EmpiricalTargets(
    domain="general_misinformation",
    targets={
        # Final prevalence: 15-35% range for typical misinformation
        "final_adoption_fraction": EmpiricalTarget(
            name="final_adoption_fraction",
            description="Final fraction adopting misinformation",
            value=0.25,
            tolerance=0.10,
            weight=1.0,
            source="Meta-analysis of multiple studies",
        ),

        # S-curve shape: inflection around 20-30% population
        "inflection_point": EmpiricalTarget(
            name="inflection_point",
            description="Fraction at epidemic inflection",
            value=0.25,
            tolerance=0.08,
            weight=0.7,
            source="Epidemiological models",
        ),

        # Mean belief intensity among adopters
        "adopter_mean_belief": EmpiricalTarget(
            name="adopter_mean_belief",
            description="Mean belief among those who adopt",
            value=0.75,
            tolerance=0.08,
            weight=0.6,
            source="Survey data synthesis",
        ),

        # Variance in population beliefs
        "belief_variance": EmpiricalTarget(
            name="belief_variance",
            description="Variance of beliefs across population",
            value=0.08,
            tolerance=0.03,
            weight=0.8,
            source="Calculated from survey distributions",
        ),

        # Structural virality of cascades
        "structural_virality": EmpiricalTarget(
            name="structural_virality",
            description="Average structural virality of cascades",
            value=3.5,
            tolerance=1.0,
            weight=0.6,
            source="Goel et al. (2016)",
        ),

        # R_effective at peak
        "peak_r_effective": EmpiricalTarget(
            name="peak_r_effective",
            description="R_effective at epidemic peak",
            value=1.8,
            tolerance=0.4,
            weight=0.7,
            source="Epidemiological estimates",
        ),

        # Education effect: negative correlation
        "education_belief_correlation": EmpiricalTarget(
            name="education_belief_correlation",
            description="Correlation of education with belief",
            value=-0.25,
            tolerance=0.08,
            weight=0.6,
            source="Pennycook & Rand (2021)",
        ),

        # Trust erosion: 10-20% decline in institutional trust
        "trust_erosion": EmpiricalTarget(
            name="trust_erosion",
            description="Decline in institutional trust",
            value=0.15,
            tolerance=0.05,
            weight=0.5,
            source="Gallup/Pew longitudinal data",
        ),
    },
)


def get_default_targets() -> EmpiricalTargets:
    """Get the default calibration targets."""
    return GENERAL_MISINFORMATION_TARGETS


def combine_targets(*target_sets: EmpiricalTargets) -> EmpiricalTargets:
    """Combine multiple target sets, averaging overlapping targets."""
    combined = {}

    for ts in target_sets:
        for name, target in ts.targets.items():
            if name in combined:
                # Average values and tolerances
                existing = combined[name]
                combined[name] = EmpiricalTarget(
                    name=name,
                    description=target.description,
                    value=(existing.value + target.value) / 2,
                    tolerance=(existing.tolerance + target.tolerance) / 2,
                    weight=max(existing.weight, target.weight),
                    source=f"{existing.source}; {target.source}",
                )
            else:
                combined[name] = target

    return EmpiricalTargets(domain="combined", targets=combined)
