"""
Cascade Analysis
=================
Analyzes structure and properties of information cascades.

Implements:
- Structural virality (Goel et al. 2016)
- Cascade depth and breadth
- Power law fitting for cascade size distribution
- Generation time analysis
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sim.cascades.tracker import CascadeEvent, CascadeState


def build_cascade_tree(
    events: List[CascadeEvent],
) -> Dict[int, List[int]]:
    """
    Build tree structure from cascade events.

    Returns adjacency list: source -> list of targets
    """
    tree: Dict[int, List[int]] = defaultdict(list)
    roots: List[int] = []

    for event in events:
        if event.source == -1:
            roots.append(event.adopter)
        elif event.source >= 0:
            tree[event.source].append(event.adopter)

    # Add roots as children of virtual root
    tree[-1] = roots

    return dict(tree)


def compute_cascade_depth(
    tree: Dict[int, List[int]],
    root: int = -1,
) -> int:
    """
    Compute maximum depth of cascade tree.

    Depth = longest path from seed to any adopter.
    """
    def dfs_depth(node: int, visited: set) -> int:
        if node in visited:
            return 0
        visited.add(node)

        children = tree.get(node, [])
        if not children:
            return 1

        max_child_depth = 0
        for child in children:
            child_depth = dfs_depth(child, visited)
            max_child_depth = max(max_child_depth, child_depth)

        return 1 + max_child_depth

    return dfs_depth(root, set()) - 1  # -1 to not count virtual root


def compute_cascade_breadth(
    tree: Dict[int, List[int]],
) -> Dict[int, int]:
    """
    Compute breadth at each level of cascade.

    Returns dict mapping level -> number of nodes at that level.
    """
    breadth: Dict[int, int] = defaultdict(int)

    def bfs():
        queue = [(-1, 0)]  # (node, level)
        visited = set()

        while queue:
            node, level = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if node != -1:  # Don't count virtual root
                breadth[level] += 1

            for child in tree.get(node, []):
                if child not in visited:
                    queue.append((child, level + 1))

    bfs()
    return dict(breadth)


def compute_structural_virality(
    events: List[CascadeEvent],
) -> float:
    """
    Compute structural virality metric (Goel et al. 2016).

    Structural virality = average pairwise distance in cascade tree.

    High structural virality = viral spread (many generations)
    Low structural virality = broadcast spread (few generations, many seeds)

    Returns value in [1, n] where n is cascade size.
    """
    if len(events) < 2:
        return 1.0

    # Build adjacency list
    tree = build_cascade_tree(events)

    # Get all nodes
    nodes = list(set([e.adopter for e in events]))
    n = len(nodes)

    if n < 2:
        return 1.0

    # Build undirected adjacency for distance computation
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for parent, children in tree.items():
        if parent != -1:
            for child in children:
                adjacency[parent].append(child)
                adjacency[child].append(parent)

    # Compute all-pairs shortest paths (BFS from each node)
    total_distance = 0
    pair_count = 0

    for start_node in nodes:
        distances = {start_node: 0}
        queue = [start_node]

        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        # Sum distances to other nodes
        for target_node in nodes:
            if target_node != start_node and target_node in distances:
                total_distance += distances[target_node]
                pair_count += 1

    if pair_count == 0:
        return 1.0

    # Average pairwise distance (divided by 2 to avoid double counting)
    structural_virality = total_distance / pair_count

    return structural_virality


def compute_generation_time_distribution(
    events: List[CascadeEvent],
) -> Dict[str, float]:
    """
    Compute distribution of generation times.

    Generation time = time between source adoption and target adoption.
    """
    generation_times = []

    # Build time lookup
    adoption_times = {e.adopter: e.time for e in events}

    for event in events:
        if event.source >= 0 and event.source in adoption_times:
            gen_time = event.time - adoption_times[event.source]
            if gen_time > 0:
                generation_times.append(gen_time)

    if len(generation_times) == 0:
        return {
            "mean_generation_time": 0.0,
            "median_generation_time": 0.0,
            "std_generation_time": 0.0,
            "min_generation_time": 0.0,
            "max_generation_time": 0.0,
        }

    gen_times_np = np.array(generation_times)

    return {
        "mean_generation_time": float(np.mean(gen_times_np)),
        "median_generation_time": float(np.median(gen_times_np)),
        "std_generation_time": float(np.std(gen_times_np)),
        "min_generation_time": float(np.min(gen_times_np)),
        "max_generation_time": float(np.max(gen_times_np)),
    }


def compute_cascade_size_distribution(
    state: CascadeState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cascade size distribution across all claims.

    Returns:
        sizes: Array of cascade sizes
        counts: Histogram counts
    """
    sizes = []

    for claim in range(state.n_claims):
        events = state.events.get(claim, [])
        if len(events) > 0:
            sizes.append(len(events))

    if len(sizes) == 0:
        return np.array([]), np.array([])

    sizes_np = np.array(sizes)

    # Create log-spaced bins for power law
    max_size = sizes_np.max()
    if max_size > 1:
        bins = np.logspace(0, np.log10(max_size), 20)
        counts, edges = np.histogram(sizes_np, bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2
    else:
        bin_centers = np.array([1])
        counts = np.array([len(sizes)])

    return bin_centers, counts


def fit_power_law_exponent(
    sizes: np.ndarray,
    x_min: float = 1.0,
) -> Tuple[float, float]:
    """
    Fit power law exponent using MLE.

    For cascade sizes following P(x) ~ x^(-alpha)

    Returns:
        alpha: Power law exponent
        x_min: Minimum x value for power law regime
    """
    if len(sizes) == 0:
        return 2.5, x_min  # Default exponent

    # Filter to values >= x_min
    filtered = sizes[sizes >= x_min]

    if len(filtered) < 10:
        return 2.5, x_min  # Not enough data

    # MLE for power law: alpha = 1 + n / sum(ln(x/x_min))
    n = len(filtered)
    log_sum = np.sum(np.log(filtered / x_min))

    if log_sum <= 0:
        return 2.5, x_min

    alpha = 1 + n / log_sum

    return float(alpha), x_min


def analyze_cascade_statistics(
    state: CascadeState,
) -> Dict[str, float]:
    """
    Compute comprehensive cascade statistics.
    """
    all_stats = {
        "total_cascades": state.n_claims,
        "total_events": state.total_events,
    }

    # Per-claim statistics
    depths = []
    viralitys = []
    sizes = []
    gen_times_all = []

    for claim in range(state.n_claims):
        events = state.events.get(claim, [])
        if len(events) == 0:
            continue

        tree = build_cascade_tree(events)
        depth = compute_cascade_depth(tree)
        virality = compute_structural_virality(events)
        gen_time_stats = compute_generation_time_distribution(events)

        depths.append(depth)
        viralitys.append(virality)
        sizes.append(len(events))
        gen_times_all.append(gen_time_stats["mean_generation_time"])

    if len(depths) > 0:
        all_stats["mean_depth"] = float(np.mean(depths))
        all_stats["max_depth"] = float(np.max(depths))
        all_stats["mean_structural_virality"] = float(np.mean(viralitys))
        all_stats["mean_cascade_size"] = float(np.mean(sizes))
        all_stats["max_cascade_size"] = float(np.max(sizes))
        all_stats["mean_generation_time"] = float(np.mean(gen_times_all))

        # Fit power law to sizes
        alpha, x_min = fit_power_law_exponent(np.array(sizes))
        all_stats["power_law_exponent"] = alpha
    else:
        all_stats["mean_depth"] = 0.0
        all_stats["max_depth"] = 0.0
        all_stats["mean_structural_virality"] = 1.0
        all_stats["mean_cascade_size"] = 0.0
        all_stats["max_cascade_size"] = 0.0
        all_stats["mean_generation_time"] = 0.0
        all_stats["power_law_exponent"] = 2.5

    return all_stats


def classify_cascade_type(
    events: List[CascadeEvent],
) -> str:
    """
    Classify cascade as viral, broadcast, or hybrid.

    Based on structural virality relative to size.
    """
    if len(events) < 5:
        return "small"

    virality = compute_structural_virality(events)
    size = len(events)

    # Normalized virality (expected for random tree is ~log(n))
    expected_virality = np.log(size) if size > 1 else 1
    normalized = virality / expected_virality

    if normalized < 0.5:
        return "broadcast"  # Low virality = few generations, many seeds
    elif normalized > 1.5:
        return "viral"  # High virality = many generations, few seeds
    else:
        return "hybrid"
