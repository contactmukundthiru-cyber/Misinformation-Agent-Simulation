"""
Narrative Competition Model
============================
Models how multiple claims compete for limited belief capacity.

Claims can be:
- Mutually exclusive (believing one reduces belief in another)
- Reinforcing (believing one increases belief in another)
- Independent (no interaction)

This models real-world narrative bundles where beliefs cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel


class NarrativeConfig(BaseModel):
    """Configuration for narrative competition."""

    # Competition strength
    competition_strength: float = 0.5
    reinforcement_strength: float = 0.3

    # Belief budget
    enable_belief_budget: bool = True
    max_total_belief: float = 2.5  # Max sum of beliefs across claims
    budget_enforcement: str = "soft"  # "soft" or "hard"

    # Narrative bundling
    bundle_formation_rate: float = 0.05
    bundle_decay_rate: float = 0.02

    # Cognitive consistency
    consistency_pressure: float = 0.3


@dataclass
class NarrativeState:
    """Tracks narrative relationships and bundling."""

    n_claims: int
    device: torch.device

    # Claim competition matrix: positive = competing, negative = reinforcing
    # Shape: (n_claims, n_claims)
    competition_matrix: torch.Tensor

    # Learned claim associations from co-exposure
    # Shape: (n_claims, n_claims)
    co_occurrence: torch.Tensor

    # Ground truth labels: 0 = false, 1 = true
    # Shape: (n_claims,)
    ground_truth: torch.Tensor

    @classmethod
    def initialize(
        cls,
        n_claims: int,
        claim_topics: List[str],
        ground_truth: List[bool],
        device: torch.device,
    ) -> "NarrativeState":
        """Initialize narrative state with claim properties."""
        # Define competition based on topics
        competition = torch.zeros(n_claims, n_claims, device=device)

        # Topic-based competition rules
        topic_competition = {
            ("health_rumor", "health_rumor"): 0.3,
            ("health_rumor", "tech_conspiracy"): 0.1,
            ("economic_panic", "economic_panic"): 0.4,
            ("moral_spiral", "tech_conspiracy"): -0.2,  # Reinforcing
            ("outsider_threat", "moral_spiral"): -0.3,  # Reinforcing
            ("outsider_threat", "economic_panic"): -0.2,
        }

        for i, topic_i in enumerate(claim_topics):
            for j, topic_j in enumerate(claim_topics):
                if i != j:
                    key = (topic_i, topic_j)
                    rev_key = (topic_j, topic_i)
                    if key in topic_competition:
                        competition[i, j] = topic_competition[key]
                    elif rev_key in topic_competition:
                        competition[i, j] = topic_competition[rev_key]

        # Truth competes with falsehood
        truth_tensor = torch.tensor(ground_truth, dtype=torch.float32, device=device)

        # False claims compete with each other less
        for i in range(n_claims):
            for j in range(n_claims):
                if i != j:
                    if not ground_truth[i] and not ground_truth[j]:
                        # False claims reinforce each other (conspiracy bundling)
                        competition[i, j] -= 0.15
                    elif ground_truth[i] != ground_truth[j]:
                        # Truth competes with falsehood
                        competition[i, j] += 0.2

        return cls(
            n_claims=n_claims,
            device=device,
            competition_matrix=competition,
            co_occurrence=torch.zeros(n_claims, n_claims, device=device),
            ground_truth=truth_tensor,
        )


def initialize_narrative_state(
    n_claims: int,
    claim_topics: List[str],
    ground_truth: List[bool],
    device: torch.device,
) -> NarrativeState:
    """Initialize narrative state."""
    return NarrativeState.initialize(n_claims, claim_topics, ground_truth, device)


def compute_claim_competition(
    beliefs: torch.Tensor,
    state: NarrativeState,
    cfg: NarrativeConfig,
) -> torch.Tensor:
    """
    Compute competition effect on belief updates.

    Returns adjustment to be applied to belief update.
    Shape: (n_agents, n_claims)
    """
    n_agents, n_claims = beliefs.shape

    # For each claim, compute pressure from competing claims
    # pressure_k = sum_j (competition[j,k] * belief_j)

    # Expand for batch computation
    # beliefs: (n_agents, n_claims)
    # competition: (n_claims, n_claims)

    # Competition effect: high belief in competing claims reduces acceptance
    competition_pressure = torch.matmul(beliefs, state.competition_matrix)

    # Split into competition (positive values) and reinforcement (negative)
    competition_effect = torch.relu(competition_pressure) * cfg.competition_strength
    reinforcement_effect = -torch.relu(-competition_pressure) * cfg.reinforcement_strength

    # Net effect on belief acceptance
    net_effect = reinforcement_effect - competition_effect

    return net_effect


def apply_belief_budget_constraint(
    beliefs: torch.Tensor,
    belief_update: torch.Tensor,
    cfg: NarrativeConfig,
) -> torch.Tensor:
    """
    Apply belief budget constraint to prevent over-belief.

    Agents have limited belief capacity across all claims.
    """
    if not cfg.enable_belief_budget:
        return belief_update

    # Compute total belief if update applied
    new_beliefs = beliefs + belief_update
    total_belief = new_beliefs.sum(dim=1, keepdim=True)

    if cfg.budget_enforcement == "hard":
        # Hard constraint: renormalize if over budget
        over_budget = total_belief > cfg.max_total_belief
        scale = cfg.max_total_belief / (total_belief + 1e-6)
        scale = torch.where(over_budget, scale, torch.ones_like(scale))

        adjusted_update = belief_update * scale

    else:  # soft
        # Soft constraint: penalize updates that push over budget
        overage = torch.relu(total_belief - cfg.max_total_belief)
        penalty = overage / (cfg.max_total_belief + 1e-6)

        # Reduce positive updates when over budget
        positive_update = torch.relu(belief_update)
        adjusted_positive = positive_update * (1.0 - penalty)

        adjusted_update = adjusted_positive - torch.relu(-belief_update)

    return adjusted_update


def update_narrative_bundles(
    state: NarrativeState,
    beliefs: torch.Tensor,
    shares: torch.Tensor,
    cfg: NarrativeConfig,
) -> None:
    """
    Update learned narrative associations from co-exposure.

    When agents are exposed to multiple claims together,
    they form associative links between those claims.
    """
    # Compute co-exposure
    # Claims shared together become associated

    # High-belief agents create associations
    high_belief = (beliefs > 0.6).float()
    share_mask = (shares > 0.5).float()

    # Co-occurrence: claims shared together by same agent
    co_share = torch.matmul(share_mask.T, share_mask)

    # Normalize by claim frequency
    claim_freq = share_mask.sum(dim=0, keepdim=True) + 1e-6
    co_share = co_share / claim_freq.T

    # Update with decay
    state.co_occurrence = (
        (1 - cfg.bundle_decay_rate) * state.co_occurrence
        + cfg.bundle_formation_rate * co_share
    )

    # Learned associations modify competition matrix slightly
    # Strong co-occurrence reduces competition (claims become bundled)
    bundle_effect = -0.1 * torch.tanh(state.co_occurrence - 0.5)
    state.competition_matrix = state.competition_matrix + 0.01 * bundle_effect


def compute_consistency_pressure(
    beliefs: torch.Tensor,
    state: NarrativeState,
    cfg: NarrativeConfig,
) -> torch.Tensor:
    """
    Compute cognitive consistency pressure.

    Agents feel discomfort holding inconsistent beliefs
    and tend to resolve toward consistency.

    Returns pressure to change beliefs toward consistency.
    """
    n_agents, n_claims = beliefs.shape

    # Inconsistency: believing competing claims
    # For each pair of claims, compute tension
    belief_outer = beliefs.unsqueeze(2) * beliefs.unsqueeze(1)
    competition_expanded = state.competition_matrix.unsqueeze(0)

    tension = belief_outer * torch.relu(competition_expanded)

    # Total tension per claim
    claim_tension = tension.sum(dim=2)

    # Pressure to reduce belief in high-tension claims
    consistency_pressure = -cfg.consistency_pressure * claim_tension

    return consistency_pressure
