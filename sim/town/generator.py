from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from sim.config import NetworkConfig, TownConfig, TraitConfig, WorldConfig
from sim.town.demographics import (
    Demographics,
    MediaDiet,
    Traits,
    Trust,
    generate_demographics,
    generate_media_diet,
    generate_traits,
    generate_trust,
    ideology_proxy,
)
from sim.town.networks import build_networks


@dataclass
class Town:
    n_agents: int
    neighborhoods: int
    neighborhood_coords: np.ndarray
    neighborhood_ids: np.ndarray
    household_ids: np.ndarray
    workplace_ids: np.ndarray
    school_ids: np.ndarray
    church_ids: np.ndarray
    demographics: Demographics
    traits: Traits
    trust: Trust
    media_diet: MediaDiet
    ideology: np.ndarray
    networks: Dict[str, np.ndarray]
    aggregate_edges: Tuple[np.ndarray, np.ndarray, np.ndarray]
    neighbor_weight_sum: np.ndarray


def assign_groups(
    rng: np.random.Generator, n_agents: int, mean_size: float, std_size: float | None = None
) -> np.ndarray:
    if std_size is None:
        std_size = mean_size * 0.3
    sizes = np.maximum(1, rng.normal(loc=mean_size, scale=std_size, size=n_agents).astype(int))
    group_ids = np.empty(n_agents, dtype=np.int32)
    gid = 0
    idx = 0
    while idx < n_agents:
        size = int(np.clip(sizes[gid % len(sizes)], 1, n_agents - idx))
        group_ids[idx : idx + size] = gid
        gid += 1
        idx += size
    rng.shuffle(group_ids)
    return group_ids


def generate_town(
    rng: np.random.Generator,
    n_agents: int,
    town_cfg: TownConfig,
    trait_cfg: TraitConfig,
    world_cfg: WorldConfig,
    network_cfg: NetworkConfig,
) -> Town:
    """Create synthetic town demographics, traits, and multilayer networks."""
    neighborhoods = town_cfg.n_neighborhoods
    grid_rows, grid_cols = town_cfg.neighborhood_grid
    coords = np.array(
        [(r, c) for r in range(grid_rows) for c in range(grid_cols)], dtype=np.float32
    )[:neighborhoods]

    neighborhood_ids = rng.integers(0, neighborhoods, size=n_agents)
    household_ids = assign_groups(
        rng, n_agents, town_cfg.household_size_mean, town_cfg.household_size_std
    )
    workplace_ids = assign_groups(rng, n_agents, town_cfg.workplace_size_mean)
    school_ids = assign_groups(rng, n_agents, town_cfg.school_size_mean)

    church_ids = np.full(n_agents, -1, dtype=np.int32)
    attendees = np.where(rng.random(n_agents) < town_cfg.church_attendance_rate)[0]
    if attendees.size:
        assigned = assign_groups(rng, attendees.size, town_cfg.church_size_mean)
        church_ids[attendees] = assigned

    demographics = generate_demographics(rng, n_agents, town_cfg)
    traits = generate_traits(rng, n_agents, trait_cfg, world_cfg.emotions_enabled)
    trust = generate_trust(rng, n_agents, world_cfg)
    media_diet = generate_media_diet(rng, n_agents)
    ideology = ideology_proxy(traits, trust)

    networks, aggregate_edges, neighbor_weight_sum = build_networks(
        rng,
        n_agents,
        neighborhood_ids,
        coords,
        household_ids,
        workplace_ids,
        school_ids,
        church_ids,
        ideology,
        network_cfg,
    )

    return Town(
        n_agents=n_agents,
        neighborhoods=neighborhoods,
        neighborhood_coords=coords,
        neighborhood_ids=neighborhood_ids,
        household_ids=household_ids,
        workplace_ids=workplace_ids,
        school_ids=school_ids,
        church_ids=church_ids,
        demographics=demographics,
        traits=traits,
        trust=trust,
        media_diet=media_diet,
        ideology=ideology,
        networks=networks,
        aggregate_edges=aggregate_edges,
        neighbor_weight_sum=neighbor_weight_sum,
    )
