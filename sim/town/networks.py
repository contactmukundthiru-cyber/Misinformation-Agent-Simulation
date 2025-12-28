from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from sim.config import NetworkConfig


def _add_edges(
    edges: List[Tuple[int, int, float, str]],
    members: np.ndarray,
    rng: np.random.Generator,
    layer: str,
    fully_connected: bool,
    avg_degree: int,
    ideology: np.ndarray,
    neighborhood_ids: np.ndarray,
    coords: np.ndarray,
    dist_matrix: np.ndarray,
    cfg: NetworkConfig,
) -> None:
    n = len(members)
    if n <= 1:
        return
    if fully_connected:
        for idx, src in enumerate(members):
            for dst in members[idx + 1 :]:
                weight = _edge_weight(src, dst, ideology, neighborhood_ids, coords, dist_matrix, cfg)
                edges.append((src, dst, weight, layer))
                edges.append((dst, src, weight, layer))
        return

    for src in members:
        k = min(n - 1, max(1, avg_degree))
        choices = rng.choice(members[members != src], size=k, replace=False)
        for dst in choices:
            weight = _edge_weight(src, dst, ideology, neighborhood_ids, coords, dist_matrix, cfg)
            edges.append((src, dst, weight, layer))


def _edge_weight(
    src: int,
    dst: int,
    ideology: np.ndarray,
    neighborhood_ids: np.ndarray,
    coords: np.ndarray,
    dist_matrix: np.ndarray,
    cfg: NetworkConfig,
) -> float:
    sim = 1.0 - abs(float(ideology[src]) - float(ideology[dst]))
    dist = float(dist_matrix[neighborhood_ids[src], neighborhood_ids[dst]])
    geo = float(np.exp(-dist / cfg.geo_scale))
    weight = 1.0 + cfg.homophily_strength * (sim - 0.5) + cfg.geography_strength * (geo - 0.5)
    return float(np.clip(weight, 0.05, 3.0))


def _group_members(group_ids: np.ndarray) -> Dict[int, np.ndarray]:
    groups: Dict[int, List[int]] = {}
    for idx, gid in enumerate(group_ids):
        groups.setdefault(int(gid), []).append(idx)
    return {gid: np.array(members, dtype=np.int32) for gid, members in groups.items()}


def _neighborhood_edges(
    rng: np.random.Generator,
    n_agents: int,
    neighborhood_ids: np.ndarray,
    coords: np.ndarray,
    ideology: np.ndarray,
    dist_matrix: np.ndarray,
    cfg: NetworkConfig,
) -> List[Tuple[int, int, float, str]]:
    edges: List[Tuple[int, int, float, str]] = []
    neighborhoods = int(neighborhood_ids.max()) + 1
    members_by_neighborhood = {nid: np.where(neighborhood_ids == nid)[0] for nid in range(neighborhoods)}
    for nid, members in members_by_neighborhood.items():
        if len(members) <= 1:
            continue
        k_intra = min(10, max(1, int(cfg.intra_neighborhood_p * len(members))))
        for src in members:
            choices = rng.choice(members[members != src], size=k_intra, replace=False)
            for dst in choices:
                weight = _edge_weight(src, dst, ideology, neighborhood_ids, coords, dist_matrix, cfg)
                edges.append((src, dst, weight, "neighborhood"))
    if neighborhoods <= 1:
        return edges
    for src in range(n_agents):
        nid = neighborhood_ids[src]
        neighbor_pool = []
        for other_nid in range(neighborhoods):
            if other_nid == nid:
                continue
            dist = dist_matrix[nid, other_nid]
            if dist <= 1.5:
                neighbor_pool.extend(members_by_neighborhood[other_nid].tolist())
        if not neighbor_pool:
            continue
        k_inter = min(3, max(1, int(cfg.inter_neighborhood_p * len(neighbor_pool))))
        choices = rng.choice(neighbor_pool, size=k_inter, replace=False)
        for dst in choices:
            weight = _edge_weight(src, dst, ideology, neighborhood_ids, coords, dist_matrix, cfg)
            edges.append((src, dst, weight, "neighborhood"))
    return edges


def build_networks(
    rng: np.random.Generator,
    n_agents: int,
    neighborhood_ids: np.ndarray,
    coords: np.ndarray,
    household_ids: np.ndarray,
    workplace_ids: np.ndarray,
    school_ids: np.ndarray,
    church_ids: np.ndarray,
    ideology: np.ndarray,
    cfg: NetworkConfig,
) -> Tuple[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    edges: List[Tuple[int, int, float, str]] = []
    neighborhoods = int(neighborhood_ids.max()) + 1
    dist_matrix = np.zeros((neighborhoods, neighborhoods), dtype=np.float32)
    for i in range(neighborhoods):
        for j in range(neighborhoods):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    for gid, members in _group_members(household_ids).items():
        _add_edges(edges, members, rng, "family", True, 0, ideology, neighborhood_ids, coords, dist_matrix, cfg)

    for gid, members in _group_members(workplace_ids).items():
        _add_edges(edges, members, rng, "work", False, 6, ideology, neighborhood_ids, coords, dist_matrix, cfg)

    for gid, members in _group_members(school_ids).items():
        _add_edges(edges, members, rng, "school", False, 5, ideology, neighborhood_ids, coords, dist_matrix, cfg)

    if np.any(church_ids >= 0):
        for gid in np.unique(church_ids[church_ids >= 0]):
            members = np.where(church_ids == gid)[0]
            _add_edges(edges, members, rng, "church", False, 5, ideology, neighborhood_ids, coords, dist_matrix, cfg)

    edges.extend(_neighborhood_edges(rng, n_agents, neighborhood_ids, coords, ideology, dist_matrix, cfg))

    layer_edges: Dict[str, List[Tuple[int, int, float]]] = {}
    for src, dst, weight, layer in edges:
        layer_edges.setdefault(layer, []).append((src, dst, weight))

    layer_arrays: Dict[str, np.ndarray] = {
        layer: np.array(items, dtype=np.float32) for layer, items in layer_edges.items()
    }

    aggregate: Dict[Tuple[int, int], float] = {}
    for layer, items in layer_edges.items():
        multiplier = cfg.layer_multipliers.get(layer, 1.0)
        for src, dst, weight in items:
            key = (int(src), int(dst))
            aggregate[key] = aggregate.get(key, 0.0) + float(weight) * multiplier

    src_idx = np.fromiter((k[0] for k in aggregate.keys()), dtype=np.int64)
    dst_idx = np.fromiter((k[1] for k in aggregate.keys()), dtype=np.int64)
    weights = np.fromiter(aggregate.values(), dtype=np.float32)

    neighbor_weight_sum = np.zeros(n_agents, dtype=np.float32)
    np.add.at(neighbor_weight_sum, dst_idx, weights)
    neighbor_weight_sum = np.clip(neighbor_weight_sum, 1e-6, None)

    return layer_arrays, (src_idx, dst_idx, weights), neighbor_weight_sum
