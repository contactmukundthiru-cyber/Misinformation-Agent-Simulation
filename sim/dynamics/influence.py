"""
Dynamic Influence and Superspreader Dynamics
==============================================
Models heterogeneous influence among agents.

Key mechanisms:
1. Influence Scores: Not all agents have equal influence
2. Dynamic Influence: Influence changes based on accuracy history
3. Superspreader Identification: High-influence nodes for targeting
4. Influence Flow: Track how influence propagates through network

References:
- Katz, E., & Lazarsfeld, P. F. (1955). Personal Influence
- Watts, D. J., & Dodds, P. S. (2007). Influentials, networks, and public opinion
- Bakshy, E., et al. (2011). Everyone's an influencer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from pydantic import BaseModel


class InfluenceConfig(BaseModel):
    """Configuration for influence dynamics."""

    # Base influence parameters
    base_influence: float = 1.0
    influence_variance: float = 0.3

    # Dynamic influence updates
    accuracy_influence_boost: float = 0.1
    inaccuracy_influence_penalty: float = 0.15
    influence_decay: float = 0.02

    # Superspreader detection
    superspreader_percentile: float = 0.95
    superspreader_influence_multiplier: float = 3.0

    # Influence limits
    min_influence: float = 0.1
    max_influence: float = 10.0

    # Attention effects
    attention_influence_weight: float = 0.3


@dataclass
class InfluenceState:
    """Tracks influence-related state for agents."""

    n_agents: int
    n_claims: int
    device: torch.device

    # Base influence score (from network position)
    base_influence: torch.Tensor = field(init=False)

    # Dynamic influence (changes over time)
    dynamic_influence: torch.Tensor = field(init=False)

    # Claim-specific influence (expertise matters)
    claim_influence: torch.Tensor = field(init=False)

    # Influence history (who influenced whom)
    influence_given: torch.Tensor = field(init=False)
    influence_received: torch.Tensor = field(init=False)

    # Accuracy tracking (for dynamic updates)
    accuracy_history: torch.Tensor = field(init=False)

    # Superspreader flags
    is_superspreader: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.base_influence = torch.ones(self.n_agents, device=self.device)
        self.dynamic_influence = torch.ones(self.n_agents, device=self.device)
        self.claim_influence = torch.ones(
            self.n_agents, self.n_claims, device=self.device
        )
        self.influence_given = torch.zeros(self.n_agents, device=self.device)
        self.influence_received = torch.zeros(self.n_agents, device=self.device)
        self.accuracy_history = torch.full(
            (self.n_agents,), 0.5, device=self.device
        )
        self.is_superspreader = torch.zeros(
            self.n_agents, dtype=torch.bool, device=self.device
        )


def initialize_influence_state(
    n_agents: int,
    n_claims: int,
    network_edges: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    traits: Dict[str, torch.Tensor],
    cfg: InfluenceConfig,
    device: torch.device,
) -> InfluenceState:
    """
    Initialize influence state from network structure and traits.

    Base influence derived from:
    - Network degree (centrality)
    - Status-seeking trait
    - Random variation
    """
    state = InfluenceState(n_agents=n_agents, n_claims=n_claims, device=device)

    src_idx, dst_idx, weights = network_edges

    # Compute weighted degree centrality
    in_degree = torch.zeros(n_agents, device=device)
    in_degree.scatter_add_(0, dst_idx, weights)

    # Normalize to [0, 1] range
    max_degree = in_degree.max() + 1e-6
    normalized_degree = in_degree / max_degree

    # Status-seeking increases influence attempts
    status_seeking = traits.get(
        "status_seeking", torch.full((n_agents,), 0.5, device=device)
    )

    # Base influence from degree and status
    state.base_influence = (
        cfg.base_influence
        + 0.5 * normalized_degree
        + 0.3 * status_seeking
        + cfg.influence_variance * torch.randn(n_agents, device=device)
    )
    state.base_influence = torch.clamp(
        state.base_influence, cfg.min_influence, cfg.max_influence
    )

    # Initialize dynamic influence to base
    state.dynamic_influence = state.base_influence.clone()

    # Claim-specific influence (based on topic-trait matching)
    for claim_idx in range(n_claims):
        state.claim_influence[:, claim_idx] = state.base_influence

    # Identify initial superspreaders
    threshold = torch.quantile(state.base_influence, cfg.superspreader_percentile)
    state.is_superspreader = state.base_influence > threshold

    return state


def compute_dynamic_influence(
    state: InfluenceState,
    beliefs: torch.Tensor,
    shares: torch.Tensor,
    network_edges: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Compute influence-weighted exposure for belief updates.

    Returns influence-weighted exposure for each agent-claim pair.
    Shape: (n_agents, n_claims)
    """
    src_idx, dst_idx, weights = network_edges

    # Shares weighted by sharer's influence
    sharer_influence = state.dynamic_influence[src_idx]
    claim_influence = state.claim_influence[src_idx]

    # Combined influence weight
    influence_weight = sharer_influence.unsqueeze(1) * claim_influence

    # Compute influenced exposure
    influenced_shares = shares[src_idx] * weights.unsqueeze(1) * influence_weight

    # Aggregate to receivers
    n_agents = state.n_agents
    n_claims = beliefs.shape[1]
    influenced_exposure = torch.zeros(n_agents, n_claims, device=beliefs.device)
    influenced_exposure.index_add_(0, dst_idx, influenced_shares)

    return influenced_exposure


def update_influence_scores(
    state: InfluenceState,
    beliefs: torch.Tensor,
    prev_beliefs: torch.Tensor,
    ground_truth: torch.Tensor,
    shares: torch.Tensor,
    network_edges: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: InfluenceConfig,
) -> None:
    """
    Update influence scores based on accuracy and impact.

    Agents who share accurate information gain influence.
    Agents who share inaccurate information lose influence.
    """
    src_idx, dst_idx, weights = network_edges

    # Compute accuracy of each agent's beliefs relative to ground truth
    # ground_truth: (n_claims,) - 0 = false, 1 = true
    belief_accuracy = 1.0 - torch.abs(beliefs - ground_truth.unsqueeze(0))
    agent_accuracy = belief_accuracy.mean(dim=1)

    # Update accuracy history (exponential moving average)
    state.accuracy_history = 0.9 * state.accuracy_history + 0.1 * agent_accuracy

    # Compute belief change induced by each agent's shares
    # If agent A shared and agent B's belief changed, A influenced B
    belief_changes = torch.abs(beliefs - prev_beliefs)

    # Track influence given
    shares_made = shares.sum(dim=1)
    state.influence_given = 0.9 * state.influence_given + 0.1 * shares_made

    # Update dynamic influence based on accuracy
    accurate_agents = state.accuracy_history > 0.6
    inaccurate_agents = state.accuracy_history < 0.4

    influence_update = torch.zeros_like(state.dynamic_influence)
    influence_update[accurate_agents] = cfg.accuracy_influence_boost
    influence_update[inaccurate_agents] = -cfg.inaccuracy_influence_penalty

    # Decay toward base influence
    decay_toward_base = cfg.influence_decay * (state.base_influence - state.dynamic_influence)

    # Apply updates
    state.dynamic_influence = state.dynamic_influence + influence_update + decay_toward_base
    state.dynamic_influence = torch.clamp(
        state.dynamic_influence, cfg.min_influence, cfg.max_influence
    )

    # Update claim-specific influence
    for claim_idx in range(state.n_claims):
        claim_accuracy = belief_accuracy[:, claim_idx]
        accurate_for_claim = claim_accuracy > 0.6

        claim_update = torch.zeros_like(state.dynamic_influence)
        claim_update[accurate_for_claim] = cfg.accuracy_influence_boost * 0.5

        state.claim_influence[:, claim_idx] = torch.clamp(
            state.claim_influence[:, claim_idx] + claim_update,
            cfg.min_influence, cfg.max_influence
        )

    # Update superspreader status
    threshold = torch.quantile(state.dynamic_influence, cfg.superspreader_percentile)
    state.is_superspreader = state.dynamic_influence > threshold


def identify_superspreaders(
    state: InfluenceState,
    cfg: InfluenceConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Identify current superspreaders and compute statistics.

    Returns:
        superspreader_indices: Indices of superspreader agents
        stats: Statistics about superspreader population
    """
    superspreader_indices = torch.where(state.is_superspreader)[0]

    n_superspreaders = len(superspreader_indices)
    n_agents = state.n_agents

    if n_superspreaders > 0:
        ss_influence = state.dynamic_influence[state.is_superspreader]
        ss_accuracy = state.accuracy_history[state.is_superspreader]

        stats = {
            "n_superspreaders": n_superspreaders,
            "fraction_superspreaders": n_superspreaders / n_agents,
            "mean_ss_influence": float(ss_influence.mean()),
            "mean_ss_accuracy": float(ss_accuracy.mean()),
            "total_ss_influence": float(ss_influence.sum()),
            "ss_influence_share": float(
                ss_influence.sum() / (state.dynamic_influence.sum() + 1e-6)
            ),
        }
    else:
        stats = {
            "n_superspreaders": 0,
            "fraction_superspreaders": 0.0,
            "mean_ss_influence": 0.0,
            "mean_ss_accuracy": 0.0,
            "total_ss_influence": 0.0,
            "ss_influence_share": 0.0,
        }

    return superspreader_indices, stats


def compute_influence_flow(
    state: InfluenceState,
    shares: torch.Tensor,
    belief_changes: torch.Tensor,
    network_edges: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Compute influence flow matrix between agents.

    Returns sparse representation of who influenced whom.
    Shape: (n_edges,) - influence flow for each edge
    """
    src_idx, dst_idx, weights = network_edges

    # Shares from source weighted by influence
    source_shares = shares[src_idx]
    source_influence = state.dynamic_influence[src_idx]

    # Belief changes at destination
    dest_changes = belief_changes[dst_idx]

    # Influence flow: product of source sharing, influence, and dest change
    influence_flow = (
        source_shares.sum(dim=1)
        * source_influence
        * dest_changes.sum(dim=1)
        * weights
    )

    return influence_flow


def compute_influence_concentration(
    state: InfluenceState,
) -> Dict[str, float]:
    """
    Compute Gini coefficient and other concentration metrics for influence.

    High concentration = few agents have most influence (concerning)
    Low concentration = influence evenly distributed
    """
    influence = state.dynamic_influence.cpu().numpy()
    n = len(influence)

    # Sort influence
    sorted_influence = np.sort(influence)

    # Compute Gini coefficient
    cumulative = np.cumsum(sorted_influence)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_influence)) - (n + 1) * cumulative[-1]) / (n * cumulative[-1] + 1e-6)

    # Top 10% share
    top_10_pct = sorted_influence[int(0.9 * n):].sum() / (influence.sum() + 1e-6)

    # Herfindahl-Hirschman Index (normalized)
    normalized = influence / (influence.sum() + 1e-6)
    hhi = (normalized ** 2).sum()

    return {
        "gini_coefficient": float(gini),
        "top_10_pct_share": float(top_10_pct),
        "hhi": float(hhi),
        "effective_n_influencers": float(1.0 / (hhi + 1e-6)),
    }


import numpy as np
