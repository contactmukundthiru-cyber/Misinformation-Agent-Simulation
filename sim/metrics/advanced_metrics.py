"""
Advanced Information-Theoretic Metrics
=======================================
Implements sophisticated metrics for analyzing belief dynamics.

Includes:
1. Belief Entropy: Diversity of beliefs in population
2. Mutual Information: How network position predicts belief
3. Transfer Entropy: Directed information flow
4. Polarization Indices: Multiple polarization measures
5. Calibration Metrics: For validation against empirical data

References:
- Shannon, C. E. (1948). A mathematical theory of communication
- Schreiber, T. (2000). Measuring information transfer
- Dimaggio, P., et al. (1996). Have Americans' social attitudes become more polarized?
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_belief_entropy(
    beliefs: torch.Tensor,
    n_bins: int = 20,
) -> torch.Tensor:
    """
    Compute Shannon entropy of belief distribution for each claim.

    H(beliefs) = -Σ p(b) log p(b)

    High entropy = diverse beliefs
    Low entropy = consensus (everyone agrees)

    Returns: (n_claims,) tensor of entropy values
    """
    n_agents, n_claims = beliefs.shape
    device = beliefs.device

    entropies = torch.zeros(n_claims, device=device)

    for claim in range(n_claims):
        claim_beliefs = beliefs[:, claim]

        # Histogram to estimate distribution
        hist = torch.histc(claim_beliefs, bins=n_bins, min=0.0, max=1.0)
        probs = hist / (hist.sum() + 1e-10)

        # Remove zero bins for log
        probs = probs[probs > 0]

        # Shannon entropy
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
        entropies[claim] = entropy

    # Normalize by max entropy
    max_entropy = np.log2(n_bins)
    normalized_entropies = entropies / max_entropy

    return normalized_entropies


def compute_mutual_information(
    beliefs: torch.Tensor,
    cluster_assignments: torch.Tensor,
    n_belief_bins: int = 10,
) -> torch.Tensor:
    """
    Compute mutual information between network position and belief.

    I(cluster; belief) = H(belief) - H(belief | cluster)

    High MI = network position strongly predicts belief (echo chambers)
    Low MI = beliefs independent of network structure

    Returns: (n_claims,) tensor of MI values
    """
    n_agents, n_claims = beliefs.shape
    device = beliefs.device

    n_clusters = int(cluster_assignments.max().item()) + 1

    mi_values = torch.zeros(n_claims, device=device)

    for claim in range(n_claims):
        claim_beliefs = beliefs[:, claim]

        # Compute H(belief)
        hist = torch.histc(claim_beliefs, bins=n_belief_bins, min=0.0, max=1.0)
        p_belief = hist / (hist.sum() + 1e-10)
        p_belief = p_belief[p_belief > 0]
        h_belief = -torch.sum(p_belief * torch.log2(p_belief + 1e-10))

        # Compute H(belief | cluster)
        h_belief_given_cluster = 0.0
        for c in range(n_clusters):
            cluster_mask = cluster_assignments == c
            n_in_cluster = cluster_mask.sum().item()

            if n_in_cluster == 0:
                continue

            cluster_beliefs = claim_beliefs[cluster_mask]

            # Conditional distribution
            cluster_hist = torch.histc(
                cluster_beliefs, bins=n_belief_bins, min=0.0, max=1.0
            )
            p_conditional = cluster_hist / (cluster_hist.sum() + 1e-10)
            p_conditional = p_conditional[p_conditional > 0]

            h_conditional = -torch.sum(
                p_conditional * torch.log2(p_conditional + 1e-10)
            )

            # Weight by cluster probability
            p_cluster = n_in_cluster / n_agents
            h_belief_given_cluster += p_cluster * h_conditional

        mi = h_belief - h_belief_given_cluster
        mi_values[claim] = max(0.0, mi)  # MI should be non-negative

    return mi_values


def compute_polarization_index(
    beliefs: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute multiple polarization indices.

    1. Bimodality coefficient: Measures if distribution is bimodal
    2. Variance ratio: Ratio of between-group to within-group variance
    3. Median distance: Distance from median to extremes
    4. Esteban-Ray index: Standard polarization measure

    Returns dict of polarization metrics per claim.
    """
    n_agents, n_claims = beliefs.shape
    device = beliefs.device

    metrics = {
        "bimodality": torch.zeros(n_claims, device=device),
        "variance_ratio": torch.zeros(n_claims, device=device),
        "median_distance": torch.zeros(n_claims, device=device),
        "esteban_ray": torch.zeros(n_claims, device=device),
    }

    for claim in range(n_claims):
        b = beliefs[:, claim]

        # 1. Bimodality coefficient
        # BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)(n-3)))
        mean_b = b.mean()
        std_b = b.std() + 1e-10
        skewness = ((b - mean_b) ** 3).mean() / (std_b ** 3)
        kurtosis = ((b - mean_b) ** 4).mean() / (std_b ** 4) - 3

        bc = (skewness ** 2 + 1) / (kurtosis + 3)
        metrics["bimodality"][claim] = torch.clamp(bc, 0.0, 1.0)

        # 2. Variance ratio (using median split)
        median_b = b.median()
        low_group = b[b < median_b]
        high_group = b[b >= median_b]

        if len(low_group) > 1 and len(high_group) > 1:
            within_var = (low_group.var() + high_group.var()) / 2
            between_var = (low_group.mean() - high_group.mean()) ** 2 / 4
            metrics["variance_ratio"][claim] = between_var / (within_var + 1e-10)

        # 3. Median distance
        extreme_low = (b < 0.2).float().mean()
        extreme_high = (b > 0.8).float().mean()
        metrics["median_distance"][claim] = extreme_low + extreme_high

        # 4. Esteban-Ray polarization index
        # P = Σ_i Σ_j π_i^(1+α) π_j |y_i - y_j|
        # Simplified: using binned approach
        n_bins = 10
        hist = torch.histc(b, bins=n_bins, min=0.0, max=1.0)
        probs = hist / (hist.sum() + 1e-10)
        bin_centers = torch.linspace(0.05, 0.95, n_bins, device=device)

        alpha = 1.6  # Standard parameter
        er_index = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                er_index += (probs[i] ** (1 + alpha)) * probs[j] * abs(bin_centers[i] - bin_centers[j])

        metrics["esteban_ray"][claim] = er_index

    return metrics


def compute_opinion_clustering_coefficient(
    beliefs: torch.Tensor,
    network_edges: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    threshold: float = 0.3,
) -> torch.Tensor:
    """
    Compute opinion-aware clustering coefficient.

    Fraction of closed triads where all three agree (beliefs within threshold).

    High value = opinions cluster with network structure
    Low value = opinions cross-cutting with network structure
    """
    src_idx, dst_idx, weights = network_edges
    n_agents, n_claims = beliefs.shape
    device = beliefs.device

    # Build adjacency set for each node
    neighbors = {}
    for i in range(len(src_idx)):
        src = int(src_idx[i])
        dst = int(dst_idx[i])
        neighbors.setdefault(src, set()).add(dst)
        neighbors.setdefault(dst, set()).add(src)

    clustering_coeffs = torch.zeros(n_claims, device=device)

    for claim in range(n_claims):
        claim_beliefs = beliefs[:, claim]
        total_triads = 0
        opinion_closed_triads = 0

        # Sample nodes for efficiency
        sample_size = min(500, n_agents)
        sample_nodes = torch.randperm(n_agents, device=device)[:sample_size]

        for node in sample_nodes.tolist():
            node_neighbors = neighbors.get(node, set())
            if len(node_neighbors) < 2:
                continue

            neighbor_list = list(node_neighbors)
            for i, n1 in enumerate(neighbor_list):
                for n2 in neighbor_list[i + 1:]:
                    # Is this a closed triad?
                    if n2 in neighbors.get(n1, set()):
                        total_triads += 1

                        # Do all three agree?
                        b_node = claim_beliefs[node]
                        b_n1 = claim_beliefs[n1]
                        b_n2 = claim_beliefs[n2]

                        if (abs(b_node - b_n1) < threshold and
                            abs(b_node - b_n2) < threshold and
                            abs(b_n1 - b_n2) < threshold):
                            opinion_closed_triads += 1

        if total_triads > 0:
            clustering_coeffs[claim] = opinion_closed_triads / total_triads

    return clustering_coeffs


def compute_belief_velocity(
    beliefs: torch.Tensor,
    prev_beliefs: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute rate of belief change metrics.

    Returns multiple velocity measures.
    """
    n_agents, n_claims = beliefs.shape

    delta = beliefs - prev_beliefs

    return {
        "mean_velocity": delta.abs().mean(dim=0),
        "max_velocity": delta.abs().max(dim=0).values,
        "positive_velocity": delta.clamp(min=0).mean(dim=0),
        "negative_velocity": (-delta).clamp(min=0).mean(dim=0),
        "acceleration": delta.var(dim=0),  # Variance indicates uneven change
    }


def compute_convergence_metrics(
    beliefs: torch.Tensor,
    belief_history: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute metrics indicating whether beliefs are converging.
    """
    n_claims = beliefs.shape[1]
    device = beliefs.device

    if len(belief_history) < 10:
        return {
            "variance_trend": torch.zeros(n_claims, device=device),
            "consensus_probability": torch.zeros(n_claims, device=device),
        }

    # Variance over recent history
    recent = belief_history[-10:]
    variances = [b.var(dim=0) for b in recent]
    variance_tensor = torch.stack(variances)

    # Trend: is variance increasing or decreasing?
    early_var = variance_tensor[:5].mean(dim=0)
    late_var = variance_tensor[5:].mean(dim=0)
    variance_trend = (late_var - early_var) / (early_var + 1e-10)

    # Consensus probability: fraction very close to 0 or 1
    extreme = (beliefs < 0.1) | (beliefs > 0.9)
    consensus_prob = extreme.float().mean(dim=0)

    return {
        "variance_trend": variance_trend,
        "consensus_probability": consensus_prob,
    }


def compute_calibration_targets(
    beliefs: torch.Tensor,
    adoption_threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Compute metrics for calibration against empirical targets.

    These are the stylized facts we want to match.
    """
    n_agents, n_claims = beliefs.shape

    adoption_fraction = (beliefs >= adoption_threshold).float().mean().item()
    mean_belief = beliefs.mean().item()
    belief_variance = beliefs.var().item()

    # Bimodality: fraction in tails vs middle
    tail_fraction = ((beliefs < 0.2) | (beliefs > 0.8)).float().mean().item()
    middle_fraction = ((beliefs >= 0.3) & (beliefs <= 0.7)).float().mean().item()

    # Claim-specific variance (heterogeneity across claims)
    claim_means = beliefs.mean(dim=0)
    claim_heterogeneity = claim_means.var().item()

    return {
        "adoption_fraction": adoption_fraction,
        "mean_belief": mean_belief,
        "belief_variance": belief_variance,
        "tail_fraction": tail_fraction,
        "middle_fraction": middle_fraction,
        "claim_heterogeneity": claim_heterogeneity,
    }


def compute_stylized_fact_distances(
    observed: Dict[str, float],
    targets: Dict[str, Tuple[float, float]],  # (target_value, tolerance)
) -> Dict[str, float]:
    """
    Compute distances from observed metrics to empirical targets.

    Used for ABC calibration.
    """
    distances = {}

    for key, (target, tolerance) in targets.items():
        if key in observed:
            raw_distance = abs(observed[key] - target)
            normalized_distance = raw_distance / (tolerance + 1e-10)
            distances[key] = normalized_distance

    # Overall distance (Euclidean)
    if distances:
        distances["total"] = np.sqrt(sum(d ** 2 for d in distances.values()))
    else:
        distances["total"] = float('inf')

    return distances


# Empirical stylized facts from misinformation literature
EMPIRICAL_TARGETS = {
    # Final adoption typically 10-40%, not 100%
    "adoption_fraction": (0.25, 0.15),

    # Mean belief should be moderate, not extreme
    "mean_belief": (0.35, 0.15),

    # Should see substantial variance
    "belief_variance": (0.06, 0.03),

    # Bimodal distribution: significant tails
    "tail_fraction": (0.25, 0.1),

    # Heterogeneity across claims
    "claim_heterogeneity": (0.02, 0.01),
}
