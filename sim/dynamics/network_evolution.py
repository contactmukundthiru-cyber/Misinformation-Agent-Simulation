"""
Dynamic Network Evolution
==========================
Implements belief-driven network rewiring and echo chamber emergence.

The network evolves based on:
1. Homophily: Agents prefer to connect with similar others
2. Heterophily Dissolution: Different agents disconnect over time
3. Triadic Closure: Friends of friends become friends
4. Structural Balance: Unstable triads resolve toward balance

This creates emergent echo chambers rather than assuming them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from pydantic import BaseModel


class NetworkEvolutionConfig(BaseModel):
    """Configuration for dynamic network evolution."""

    # Rewiring parameters
    rewiring_rate: float = 0.02
    max_rewires_per_step: int = 100

    # Homophily dynamics
    belief_distance_threshold: float = 0.5
    homophily_rewiring_strength: float = 0.6
    heterophily_dissolution_rate: float = 0.1

    # Triadic closure
    triadic_closure_rate: float = 0.05
    max_new_ties_per_step: int = 50

    # Structural constraints
    min_degree: int = 2
    max_degree: int = 50
    preserve_family_ties: bool = True

    # Echo chamber detection
    modularity_window: int = 30
    echo_chamber_threshold: float = 0.4


@dataclass
class DynamicNetworkState:
    """Tracks dynamic network state."""

    n_agents: int
    device: torch.device

    # Current edges as adjacency lists
    # For GPU efficiency, store as COO format
    src_idx: torch.Tensor
    dst_idx: torch.Tensor
    weights: torch.Tensor

    # Edge metadata
    edge_types: torch.Tensor  # 0=family, 1=work, 2=school, 3=church, 4=social
    edge_ages: torch.Tensor  # How long edge has existed

    # Degree information
    in_degree: torch.Tensor
    out_degree: torch.Tensor

    # Network metrics over time
    modularity_history: List[float] = field(default_factory=list)
    cluster_assignments: Optional[torch.Tensor] = None
    n_clusters: int = 0

    # Bridge tie tracking
    bridge_tie_indices: Optional[torch.Tensor] = None

    @classmethod
    def from_static_network(
        cls,
        src_idx: np.ndarray,
        dst_idx: np.ndarray,
        weights: np.ndarray,
        edge_types: Optional[np.ndarray],
        n_agents: int,
        device: torch.device,
    ) -> "DynamicNetworkState":
        src_t = torch.tensor(src_idx, device=device, dtype=torch.long)
        dst_t = torch.tensor(dst_idx, device=device, dtype=torch.long)
        weights_t = torch.tensor(weights, device=device, dtype=torch.float32)

        if edge_types is not None:
            types_t = torch.tensor(edge_types, device=device, dtype=torch.long)
        else:
            types_t = torch.full_like(src_t, 4)  # Default to social

        edge_ages = torch.zeros_like(weights_t)

        # Compute degrees
        in_degree = torch.zeros(n_agents, device=device)
        out_degree = torch.zeros(n_agents, device=device)
        in_degree.scatter_add_(0, dst_t, torch.ones_like(dst_t, dtype=torch.float32))
        out_degree.scatter_add_(0, src_t, torch.ones_like(src_t, dtype=torch.float32))

        return cls(
            n_agents=n_agents,
            device=device,
            src_idx=src_t,
            dst_idx=dst_t,
            weights=weights_t,
            edge_types=types_t,
            edge_ages=edge_ages,
            in_degree=in_degree,
            out_degree=out_degree,
        )


def initialize_dynamic_network(
    static_edges: Tuple[np.ndarray, np.ndarray, np.ndarray],
    n_agents: int,
    device: torch.device,
) -> DynamicNetworkState:
    """Initialize dynamic network from static network structure."""
    src_idx, dst_idx, weights = static_edges
    return DynamicNetworkState.from_static_network(
        src_idx, dst_idx, weights, None, n_agents, device
    )


def compute_rewiring_probabilities(
    state: DynamicNetworkState,
    beliefs: torch.Tensor,
    cfg: NetworkEvolutionConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute probabilities for edge dissolution and formation.

    Returns:
        dissolution_probs: Probability each edge will dissolve (n_edges,)
        formation_candidates: Potential new edges to form
    """
    # Compute belief distance for each edge
    src_beliefs = beliefs[state.src_idx]
    dst_beliefs = beliefs[state.dst_idx]
    belief_distance = torch.abs(src_beliefs - dst_beliefs).mean(dim=1)

    # Dissolution probability increases with belief distance
    base_dissolution = cfg.heterophily_dissolution_rate
    distance_factor = torch.relu(belief_distance - cfg.belief_distance_threshold)
    dissolution_probs = base_dissolution + cfg.homophily_rewiring_strength * distance_factor

    # Protect family ties if configured
    if cfg.preserve_family_ties:
        family_mask = state.edge_types == 0
        dissolution_probs = torch.where(
            family_mask,
            dissolution_probs * 0.1,  # 90% reduction for family
            dissolution_probs
        )

    # Newer edges more likely to dissolve (relationship not established)
    age_protection = 1.0 - torch.exp(-state.edge_ages / 30.0)
    dissolution_probs = dissolution_probs * (1.0 - 0.5 * age_protection)

    # Clamp probabilities
    dissolution_probs = torch.clamp(dissolution_probs, 0.0, 0.5)

    return dissolution_probs, belief_distance


def find_triadic_closure_candidates(
    state: DynamicNetworkState,
    beliefs: torch.Tensor,
    cfg: NetworkEvolutionConfig,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find candidate edges for triadic closure.

    Friends of friends who are belief-similar become candidates.

    Returns:
        new_src: Source nodes for potential edges
        new_dst: Destination nodes for potential edges
        formation_prob: Probability of forming each edge
    """
    # Sample nodes for triadic closure (full computation too expensive)
    n_samples = min(1000, state.n_agents)
    # Use CPU randperm to avoid CUDA generator issues
    sample_nodes = torch.randperm(state.n_agents, device='cpu')[:n_samples].to(state.device)

    new_edges_src = []
    new_edges_dst = []
    formation_probs = []

    # Build neighbor sets for sampled nodes
    for node in sample_nodes:
        # Get direct neighbors
        node_mask = state.src_idx == node
        neighbors = state.dst_idx[node_mask]

        if len(neighbors) == 0:
            continue

        # Get neighbors of neighbors
        for neighbor in neighbors[:10]:  # Limit for efficiency
            neighbor_mask = state.src_idx == neighbor
            second_neighbors = state.dst_idx[neighbor_mask]

            # Filter to non-existing edges
            for candidate in second_neighbors[:5]:
                if candidate == node:
                    continue

                # Check if edge already exists
                existing = ((state.src_idx == node) & (state.dst_idx == candidate)).any()
                if existing:
                    continue

                # Compute formation probability based on belief similarity
                node_belief = beliefs[node]
                candidate_belief = beliefs[candidate]
                similarity = 1.0 - torch.abs(node_belief - candidate_belief).mean()

                if similarity > 0.5:  # Only similar agents
                    new_edges_src.append(node)
                    new_edges_dst.append(candidate)
                    formation_probs.append(cfg.triadic_closure_rate * float(similarity))

    if len(new_edges_src) == 0:
        return (
            torch.tensor([], device=state.device, dtype=torch.long),
            torch.tensor([], device=state.device, dtype=torch.long),
            torch.tensor([], device=state.device),
        )

    return (
        torch.tensor(new_edges_src, device=state.device, dtype=torch.long),
        torch.tensor(new_edges_dst, device=state.device, dtype=torch.long),
        torch.tensor(formation_probs, device=state.device),
    )


def update_network_structure(
    state: DynamicNetworkState,
    beliefs: torch.Tensor,
    cfg: NetworkEvolutionConfig,
    rng: torch.Generator,
) -> Dict[str, int]:
    """
    Update network structure based on beliefs.

    Performs:
    1. Edge dissolution based on belief distance
    2. New edge formation via triadic closure
    3. Degree constraint enforcement

    Returns statistics on changes made.
    """
    stats = {"dissolved": 0, "formed": 0}

    # Increment edge ages
    state.edge_ages = state.edge_ages + 1

    # Step 1: Compute dissolution probabilities
    dissolution_probs, belief_distances = compute_rewiring_probabilities(
        state, beliefs, cfg
    )

    # Sample edges to dissolve
    dissolve_mask = torch.bernoulli(dissolution_probs, generator=rng).bool()

    # Enforce minimum degree constraint
    for node in range(state.n_agents):
        node_edges = state.dst_idx == node
        if dissolve_mask[node_edges].sum() >= state.in_degree[node] - cfg.min_degree:
            # Would leave node below min degree, reduce dissolutions
            node_dissolve = dissolve_mask.clone()
            node_dissolve[~node_edges] = False
            n_to_keep = int(state.in_degree[node].item()) - cfg.min_degree
            if n_to_keep > 0:
                keep_indices = torch.where(node_dissolve)[0][:n_to_keep]
                dissolve_mask[keep_indices] = False

    # Limit total dissolutions
    n_dissolve = min(dissolve_mask.sum().item(), cfg.max_rewires_per_step)
    if n_dissolve > 0:
        dissolve_indices = torch.where(dissolve_mask)[0]
        if len(dissolve_indices) > n_dissolve:
            keep_mask = torch.zeros_like(dissolve_mask)
            # Use CPU randperm and index to avoid CUDA generator issue
            perm = torch.randperm(len(dissolve_indices), device='cpu')[:n_dissolve]
            selected = dissolve_indices[perm.to(state.device)]
            keep_mask[selected] = True
            dissolve_mask = keep_mask

    # Remove dissolved edges
    keep_mask = ~dissolve_mask
    state.src_idx = state.src_idx[keep_mask]
    state.dst_idx = state.dst_idx[keep_mask]
    state.weights = state.weights[keep_mask]
    state.edge_types = state.edge_types[keep_mask]
    state.edge_ages = state.edge_ages[keep_mask]

    stats["dissolved"] = int(dissolve_mask.sum().item())

    # Update degrees after dissolution
    state.in_degree = torch.zeros(state.n_agents, device=state.device)
    state.out_degree = torch.zeros(state.n_agents, device=state.device)
    state.in_degree.scatter_add_(0, state.dst_idx, torch.ones_like(state.dst_idx, dtype=torch.float32))
    state.out_degree.scatter_add_(0, state.src_idx, torch.ones_like(state.src_idx, dtype=torch.float32))

    # Step 2: Find new edge candidates via triadic closure
    new_src, new_dst, formation_probs = find_triadic_closure_candidates(
        state, beliefs, cfg, rng
    )

    if len(new_src) > 0:
        # Sample new edges to form (use CPU for bernoulli to avoid generator device issues)
        form_mask = (torch.rand(len(formation_probs), device=state.device) < formation_probs).bool()

        # Enforce max degree constraint
        for i in range(len(new_src)):
            if form_mask[i]:
                src_node = new_src[i]
                dst_node = new_dst[i]
                if (state.out_degree[src_node] >= cfg.max_degree or
                    state.in_degree[dst_node] >= cfg.max_degree):
                    form_mask[i] = False

        # Limit new edges
        n_form = min(form_mask.sum().item(), cfg.max_new_ties_per_step)
        if n_form > 0 and form_mask.sum() > n_form:
            form_indices = torch.where(form_mask)[0]
            perm = torch.randperm(len(form_indices), device='cpu')[:n_form]
            selected = form_indices[perm.to(state.device)]
            form_mask = torch.zeros_like(form_mask)
            form_mask[selected] = True

        # Add new edges
        new_edges_src = new_src[form_mask]
        new_edges_dst = new_dst[form_mask]

        if len(new_edges_src) > 0:
            new_weights = torch.ones(len(new_edges_src), device=state.device)
            new_types = torch.full((len(new_edges_src),), 4, device=state.device, dtype=torch.long)
            new_ages = torch.zeros(len(new_edges_src), device=state.device)

            state.src_idx = torch.cat([state.src_idx, new_edges_src])
            state.dst_idx = torch.cat([state.dst_idx, new_edges_dst])
            state.weights = torch.cat([state.weights, new_weights])
            state.edge_types = torch.cat([state.edge_types, new_types])
            state.edge_ages = torch.cat([state.edge_ages, new_ages])

            # Also add reverse edges (undirected network)
            state.src_idx = torch.cat([state.src_idx, new_edges_dst])
            state.dst_idx = torch.cat([state.dst_idx, new_edges_src])
            state.weights = torch.cat([state.weights, new_weights])
            state.edge_types = torch.cat([state.edge_types, new_types])
            state.edge_ages = torch.cat([state.edge_ages, new_ages])

            stats["formed"] = len(new_edges_src)

    # Update degrees after formation
    state.in_degree = torch.zeros(state.n_agents, device=state.device)
    state.out_degree = torch.zeros(state.n_agents, device=state.device)
    state.in_degree.scatter_add_(0, state.dst_idx, torch.ones_like(state.dst_idx, dtype=torch.float32))
    state.out_degree.scatter_add_(0, state.src_idx, torch.ones_like(state.src_idx, dtype=torch.float32))

    return stats


def detect_echo_chambers(
    state: DynamicNetworkState,
    beliefs: torch.Tensor,
    cfg: NetworkEvolutionConfig,
) -> Tuple[torch.Tensor, float]:
    """
    Detect echo chambers based on belief clustering in network.

    Uses a simplified modularity-based approach optimized for GPU.

    Returns:
        cluster_assignments: Cluster ID for each agent
        modularity: Network modularity score
    """
    # Simple spectral-like clustering on GPU
    # Build weighted adjacency considering belief similarity
    n = state.n_agents
    device = state.device

    # Compute edge weights modulated by belief similarity
    src_beliefs = beliefs[state.src_idx]
    dst_beliefs = beliefs[state.dst_idx]
    belief_similarity = 1.0 - torch.abs(src_beliefs - dst_beliefs).mean(dim=1)
    modulated_weights = state.weights * belief_similarity

    # Build degree matrix
    degree = torch.zeros(n, device=device)
    degree.scatter_add_(0, state.dst_idx, modulated_weights)
    degree = torch.clamp(degree, min=1e-6)

    # Compute normalized Laplacian entries for clustering
    # Using simple label propagation for efficiency
    n_clusters = min(10, max(2, n // 500))
    labels = torch.randint(0, n_clusters, (n,), device=device)

    # Simple label propagation iterations
    for _ in range(10):
        # Aggregate neighbor labels weighted by edge weight
        neighbor_votes = torch.zeros(n, n_clusters, device=device)
        for c in range(n_clusters):
            cluster_mask = labels[state.src_idx] == c
            votes = modulated_weights * cluster_mask.float()
            neighbor_votes[:, c].scatter_add_(0, state.dst_idx, votes)

        # Assign to majority cluster
        new_labels = neighbor_votes.argmax(dim=1)

        # Add stability: don't change if tie
        max_votes = neighbor_votes.max(dim=1).values
        total_votes = neighbor_votes.sum(dim=1) + 1e-6
        confidence = max_votes / total_votes
        stable = confidence > 0.4
        labels = torch.where(stable, new_labels, labels)

    state.cluster_assignments = labels
    state.n_clusters = n_clusters

    # Compute modularity
    total_weight = modulated_weights.sum()
    if total_weight < 1e-6:
        modularity = 0.0
    else:
        same_cluster = labels[state.src_idx] == labels[state.dst_idx]
        within_weight = modulated_weights[same_cluster].sum()

        # Expected within-cluster weight under null model
        expected = 0.0
        for c in range(n_clusters):
            cluster_degree = degree[labels == c].sum()
            expected += (cluster_degree ** 2) / (2 * total_weight)

        modularity = float((within_weight - expected) / total_weight)

    state.modularity_history.append(modularity)
    if len(state.modularity_history) > cfg.modularity_window:
        state.modularity_history = state.modularity_history[-cfg.modularity_window:]

    return labels, modularity


def measure_bridge_ties(
    state: DynamicNetworkState,
    beliefs: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """
    Measure bridge ties connecting different belief clusters.

    Bridge ties are edges connecting agents with substantially
    different beliefs. These are important for information diversity.

    Returns:
        bridge_mask: Boolean mask of bridge tie edges
        bridge_fraction: Fraction of edges that are bridges
    """
    # Compute belief distance for each edge
    src_beliefs = beliefs[state.src_idx]
    dst_beliefs = beliefs[state.dst_idx]
    belief_distance = torch.abs(src_beliefs - dst_beliefs).mean(dim=1)

    # Bridge ties connect agents with different beliefs
    bridge_threshold = 0.4
    bridge_mask = belief_distance > bridge_threshold

    # Also consider cluster membership if available
    if state.cluster_assignments is not None:
        cross_cluster = state.cluster_assignments[state.src_idx] != state.cluster_assignments[state.dst_idx]
        bridge_mask = bridge_mask | cross_cluster

    state.bridge_tie_indices = torch.where(bridge_mask)[0]
    bridge_fraction = bridge_mask.float().mean().item()

    return bridge_mask, bridge_fraction


def compute_network_polarization(
    state: DynamicNetworkState,
    beliefs: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute network-based polarization metrics.

    Returns multiple polarization measures:
    - belief_segregation: How segregated are beliefs in network
    - cross_cluster_exposure: How much exposure across clusters
    - effective_modularity: Modularity accounting for belief distribution
    """
    device = state.device

    # Belief segregation: variance in belief explained by network position
    if state.cluster_assignments is not None:
        n_clusters = state.n_clusters
        cluster_means = torch.zeros(n_clusters, beliefs.shape[1], device=device)
        cluster_counts = torch.zeros(n_clusters, device=device)

        for c in range(n_clusters):
            mask = state.cluster_assignments == c
            if mask.any():
                cluster_means[c] = beliefs[mask].mean(dim=0)
                cluster_counts[c] = mask.float().sum()

        # Between-cluster variance
        global_mean = beliefs.mean(dim=0)
        between_var = 0.0
        for c in range(n_clusters):
            if cluster_counts[c] > 0:
                between_var += cluster_counts[c] * ((cluster_means[c] - global_mean) ** 2).mean()
        between_var = between_var / (beliefs.shape[0] + 1e-6)

        # Total variance
        total_var = beliefs.var().item()

        belief_segregation = float(between_var / (total_var + 1e-6))
    else:
        belief_segregation = 0.0

    # Cross-cluster exposure: fraction of exposure from different clusters
    if state.cluster_assignments is not None:
        cross_mask = state.cluster_assignments[state.src_idx] != state.cluster_assignments[state.dst_idx]
        cross_weight = state.weights[cross_mask].sum()
        total_weight = state.weights.sum()
        cross_cluster_exposure = float(cross_weight / (total_weight + 1e-6))
    else:
        cross_cluster_exposure = 0.5  # Assume random

    # Recent modularity trend
    if len(state.modularity_history) > 0:
        effective_modularity = np.mean(state.modularity_history[-10:])
    else:
        effective_modularity = 0.0

    return {
        "belief_segregation": belief_segregation,
        "cross_cluster_exposure": cross_cluster_exposure,
        "effective_modularity": effective_modularity,
        "n_bridge_ties": len(state.bridge_tie_indices) if state.bridge_tie_indices is not None else 0,
    }
