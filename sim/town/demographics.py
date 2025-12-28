from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from sim.config import TraitConfig, TownConfig, WorldConfig


@dataclass
class Demographics:
    age: np.ndarray
    education_level: np.ndarray
    occupation: np.ndarray


@dataclass
class Traits:
    personality: np.ndarray
    skepticism: np.ndarray
    need_for_closure: np.ndarray
    conspiratorial_tendency: np.ndarray
    numeracy: np.ndarray
    conformity: np.ndarray
    status_seeking: np.ndarray
    prosociality: np.ndarray
    conflict_tolerance: np.ndarray
    emotions: Dict[str, np.ndarray]


@dataclass
class Trust:
    trust_gov: np.ndarray
    trust_church: np.ndarray
    trust_local_news: np.ndarray
    trust_national_news: np.ndarray
    trust_friends: np.ndarray
    trust_outgroups: np.ndarray


@dataclass
class MediaDiet:
    channels: List[str]
    weights: np.ndarray


def _beta(rng: np.random.Generator, alpha: float, beta: float, size: int) -> np.ndarray:
    return rng.beta(alpha, beta, size=size).astype(np.float32)


def generate_demographics(
    rng: np.random.Generator,
    n_agents: int,
    town: TownConfig,
) -> Demographics:
    children = int(n_agents * town.children_fraction)
    seniors = int(n_agents * town.senior_fraction)
    adults = n_agents - children - seniors

    ages = np.concatenate([
        rng.integers(0, 18, size=children),
        rng.integers(18, 65, size=adults),
        rng.integers(65, town.max_age + 1, size=seniors),
    ])
    rng.shuffle(ages)

    education_levels = rng.integers(0, len(town.education_levels), size=n_agents)
    occupation = rng.integers(0, len(town.occupation_types), size=n_agents)
    return Demographics(age=ages.astype(np.int32), education_level=education_levels, occupation=occupation)


def generate_traits(
    rng: np.random.Generator,
    n_agents: int,
    traits: TraitConfig,
    emotions_enabled: bool,
) -> Traits:
    personality = np.stack([
        _beta(rng, traits.personality.alpha, traits.personality.beta, n_agents)
        for _ in range(5)
    ], axis=1)

    skepticism = _beta(rng, traits.cognitive.alpha, traits.cognitive.beta, n_agents)
    need_for_closure = _beta(rng, traits.cognitive.alpha, traits.cognitive.beta, n_agents)
    conspiratorial_tendency = _beta(rng, traits.cognitive.alpha, traits.cognitive.beta, n_agents)
    numeracy = _beta(rng, traits.cognitive.alpha, traits.cognitive.beta, n_agents)

    conformity = _beta(rng, traits.social.alpha, traits.social.beta, n_agents)
    status_seeking = _beta(rng, traits.social.alpha, traits.social.beta, n_agents)
    prosociality = _beta(rng, traits.social.alpha, traits.social.beta, n_agents)
    conflict_tolerance = _beta(rng, traits.social.alpha, traits.social.beta, n_agents)

    emotions = {}
    if emotions_enabled:
        emotions = {
            "fear": _beta(rng, traits.emotion.alpha, traits.emotion.beta, n_agents),
            "anger": _beta(rng, traits.emotion.alpha, traits.emotion.beta, n_agents),
            "hope": _beta(rng, traits.emotion.alpha, traits.emotion.beta, n_agents),
        }
    return Traits(
        personality=personality,
        skepticism=skepticism,
        need_for_closure=need_for_closure,
        conspiratorial_tendency=conspiratorial_tendency,
        numeracy=numeracy,
        conformity=conformity,
        status_seeking=status_seeking,
        prosociality=prosociality,
        conflict_tolerance=conflict_tolerance,
        emotions=emotions,
    )


def generate_trust(
    rng: np.random.Generator,
    n_agents: int,
    world: WorldConfig,
) -> Trust:
    def jitter(base: float) -> np.ndarray:
        vals = rng.normal(loc=base, scale=world.trust_variance, size=n_agents)
        return np.clip(vals, 0.0, 1.0).astype(np.float32)

    return Trust(
        trust_gov=jitter(world.trust_baselines["gov"]),
        trust_church=jitter(world.trust_baselines["church"]),
        trust_local_news=jitter(world.trust_baselines["local_news"]),
        trust_national_news=jitter(world.trust_baselines["national_news"]),
        trust_friends=jitter(world.trust_baselines["friends"]),
        trust_outgroups=jitter(world.trust_baselines["outgroups"]),
    )


def generate_media_diet(
    rng: np.random.Generator,
    n_agents: int,
) -> MediaDiet:
    channels = ["local_social", "national_social", "tv", "local_news", "church"]
    weights = rng.dirichlet([1.5, 1.2, 1.0, 1.3, 0.9], size=n_agents).astype(np.float32)
    return MediaDiet(channels=channels, weights=weights)


def ideology_proxy(traits: Traits, trust: Trust) -> np.ndarray:
    raw = (
        0.2 * traits.conspiratorial_tendency
        + 0.2 * (1 - traits.numeracy)
        + 0.15 * traits.conformity
        + 0.15 * (1 - trust.trust_outgroups)
        + 0.15 * trust.trust_church
        + 0.15 * (1 - trust.trust_national_news)
    )
    return np.clip(raw, 0.0, 1.0).astype(np.float32)
