from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sim.config import StrainConfig


@dataclass
class Strain:
    name: str
    topic: str
    memeticity: float
    emotional_profile: Dict[str, float]
    falsifiability: float
    stealth: float
    mutation_rate: float
    violation_risk: float


def default_strains() -> List[Strain]:
    return [
        Strain(
            name="silver_river",
            topic="health_rumor",
            memeticity=0.55,
            emotional_profile={"fear": 0.6, "anger": 0.2, "hope": 0.2},
            falsifiability=0.7,
            stealth=0.4,
            mutation_rate=0.05,
            violation_risk=0.3,
        ),
        Strain(
            name="market_shiver",
            topic="economic_panic",
            memeticity=0.6,
            emotional_profile={"fear": 0.5, "anger": 0.3, "hope": 0.2},
            falsifiability=0.6,
            stealth=0.45,
            mutation_rate=0.04,
            violation_risk=0.35,
        ),
        Strain(
            name="temple_echo",
            topic="moral_spiral",
            memeticity=0.5,
            emotional_profile={"fear": 0.3, "anger": 0.4, "hope": 0.3},
            falsifiability=0.5,
            stealth=0.55,
            mutation_rate=0.06,
            violation_risk=0.4,
        ),
        Strain(
            name="signal_fog",
            topic="tech_conspiracy",
            memeticity=0.58,
            emotional_profile={"fear": 0.2, "anger": 0.5, "hope": 0.3},
            falsifiability=0.65,
            stealth=0.5,
            mutation_rate=0.05,
            violation_risk=0.45,
        ),
        Strain(
            name="border_whisper",
            topic="outsider_threat",
            memeticity=0.62,
            emotional_profile={"fear": 0.4, "anger": 0.4, "hope": 0.2},
            falsifiability=0.55,
            stealth=0.5,
            mutation_rate=0.04,
            violation_risk=0.5,
        ),
    ]


def load_strains(strain_cfgs: List[StrainConfig] | None) -> List[Strain]:
    if not strain_cfgs:
        return default_strains()
    strains = []
    for cfg in strain_cfgs:
        strains.append(
            Strain(
                name=cfg.name,
                topic=cfg.topic,
                memeticity=cfg.memeticity,
                emotional_profile=cfg.emotional_profile,
                falsifiability=cfg.falsifiability,
                stealth=cfg.stealth,
                mutation_rate=cfg.mutation_rate,
                violation_risk=cfg.violation_risk,
            )
        )
    return strains


def mutate_strains(strains: List[Strain], rng) -> List[Strain]:
    mutated = []
    for strain in strains:
        if rng.random() < strain.mutation_rate:
            stealth = min(1.0, max(0.0, strain.stealth + rng.normal(0, 0.05)))
            falsifiability = min(1.0, max(0.1, strain.falsifiability - rng.normal(0, 0.03)))
            mutated.append(
                Strain(
                    name=strain.name + "_m",
                    topic=strain.topic,
                    memeticity=strain.memeticity,
                    emotional_profile=strain.emotional_profile,
                    falsifiability=falsifiability,
                    stealth=stealth,
                    mutation_rate=strain.mutation_rate,
                    violation_risk=strain.violation_risk,
                )
            )
        else:
            mutated.append(strain)
    return mutated
