"""
Cascade Event Tracking
=======================
Records information cascade events as they occur.

Tracks:
- Who adopted which claim at what time
- Who was the likely source of each adoption
- Chain of custody for information spread
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel


class CascadeConfig(BaseModel):
    """Configuration for cascade tracking."""

    # Memory limits
    max_events_per_claim: int = 100000
    pruning_window_days: int = 90

    # Attribution settings
    attribution_threshold: float = 0.1
    multi_source_threshold: float = 0.3

    # Generation tracking
    track_generations: bool = True
    max_generation_depth: int = 50


@dataclass
class CascadeEvent:
    """A single cascade event (adoption)."""

    time: int
    adopter: int
    claim: int
    source: int  # -1 for seed/external
    exposure_strength: float
    generation: int
    belief_before: float
    belief_after: float


@dataclass
class CascadeState:
    """Tracks cascade state across simulation."""

    n_agents: int
    n_claims: int
    device: torch.device

    # Event storage (list of events per claim)
    events: Dict[int, List[CascadeEvent]] = field(default_factory=dict)

    # Current generation per agent per claim
    # -1 = not adopted, 0 = seed, 1+ = generation number
    generation: torch.Tensor = field(init=False)

    # Adoption time per agent per claim (-1 = not adopted)
    adoption_time: torch.Tensor = field(init=False)

    # Primary source per agent per claim
    primary_source: torch.Tensor = field(init=False)

    # Total events recorded
    total_events: int = 0

    def __post_init__(self):
        self.generation = torch.full(
            (self.n_agents, self.n_claims), -1,
            dtype=torch.long, device=self.device
        )
        self.adoption_time = torch.full(
            (self.n_agents, self.n_claims), -1,
            dtype=torch.long, device=self.device
        )
        self.primary_source = torch.full(
            (self.n_agents, self.n_claims), -1,
            dtype=torch.long, device=self.device
        )
        for claim in range(self.n_claims):
            self.events[claim] = []


def initialize_cascade_tracker(
    n_agents: int,
    n_claims: int,
    seed_agents: torch.Tensor,
    device: torch.device,
) -> CascadeState:
    """
    Initialize cascade tracker with seed agents.

    Args:
        seed_agents: Boolean mask (n_agents, n_claims) of initially-seeded agents
    """
    state = CascadeState(n_agents=n_agents, n_claims=n_claims, device=device)

    # Mark seed agents as generation 0
    state.generation = torch.where(
        seed_agents,
        torch.zeros_like(state.generation),
        state.generation
    )

    # Seed agents adopted at time 0
    state.adoption_time = torch.where(
        seed_agents,
        torch.zeros_like(state.adoption_time),
        state.adoption_time
    )

    # Seed agents have external source (-1)
    state.primary_source = torch.where(
        seed_agents,
        torch.full_like(state.primary_source, -1),
        state.primary_source
    )

    # Record seed events
    seed_indices = torch.where(seed_agents)
    for agent, claim in zip(seed_indices[0].tolist(), seed_indices[1].tolist()):
        event = CascadeEvent(
            time=0,
            adopter=agent,
            claim=claim,
            source=-1,
            exposure_strength=1.0,
            generation=0,
            belief_before=0.0,
            belief_after=0.85,
        )
        state.events[claim].append(event)
        state.total_events += 1

    return state


def record_adoption_event(
    state: CascadeState,
    time: int,
    beliefs: torch.Tensor,
    prev_beliefs: torch.Tensor,
    exposure_sources: torch.Tensor,
    adoption_threshold: float,
    cfg: CascadeConfig,
) -> int:
    """
    Record new adoption events this timestep.

    Args:
        exposure_sources: (n_agents, n_claims, n_agents) sparse tensor
            or (n_agents, n_claims) with dominant source per exposure

    Returns:
        Number of new adoptions recorded
    """
    # Find new adoptions: crossed threshold this step
    newly_adopted = (beliefs >= adoption_threshold) & (prev_beliefs < adoption_threshold)

    # Also not already marked as adopted
    not_yet_tracked = state.adoption_time < 0
    new_adoptions = newly_adopted & not_yet_tracked

    if not new_adoptions.any():
        return 0

    n_new = 0
    adoption_indices = torch.where(new_adoptions)

    for agent, claim in zip(adoption_indices[0].tolist(), adoption_indices[1].tolist()):
        # Determine source: agent with highest exposure contribution
        # For now, use a simplified heuristic based on neighbor beliefs

        # Find most likely source from exposure_sources if provided
        # Otherwise, mark as unknown (-2)
        if exposure_sources is not None and exposure_sources.dim() == 3:
            agent_exposure = exposure_sources[agent, claim, :]
            if agent_exposure.max() > cfg.attribution_threshold:
                source = int(agent_exposure.argmax())
            else:
                source = -2  # Unknown
        else:
            source = -2

        # Determine generation
        if source >= 0 and state.generation[source, claim] >= 0:
            gen = int(state.generation[source, claim]) + 1
        elif source == -1:
            gen = 0  # Seed
        else:
            # Unknown source: estimate from neighbors or mark as unknown gen
            gen = 1  # Assume first generation from seed

        gen = min(gen, cfg.max_generation_depth)

        # Record event
        event = CascadeEvent(
            time=time,
            adopter=agent,
            claim=claim,
            source=source,
            exposure_strength=float(beliefs[agent, claim] - prev_beliefs[agent, claim]),
            generation=gen,
            belief_before=float(prev_beliefs[agent, claim]),
            belief_after=float(beliefs[agent, claim]),
        )

        # Check memory limit
        if len(state.events[claim]) < cfg.max_events_per_claim:
            state.events[claim].append(event)
            state.total_events += 1

        # Update state tensors
        state.generation[agent, claim] = gen
        state.adoption_time[agent, claim] = time
        state.primary_source[agent, claim] = source

        n_new += 1

    return n_new


def get_cascade_for_claim(
    state: CascadeState,
    claim: int,
) -> List[CascadeEvent]:
    """Get all events for a specific claim."""
    return state.events.get(claim, [])


def prune_old_events(
    state: CascadeState,
    current_time: int,
    cfg: CascadeConfig,
) -> int:
    """
    Remove events older than pruning window.

    Returns number of events pruned.
    """
    cutoff = current_time - cfg.pruning_window_days
    n_pruned = 0

    for claim in range(state.n_claims):
        original_len = len(state.events[claim])
        state.events[claim] = [
            e for e in state.events[claim] if e.time >= cutoff
        ]
        n_pruned += original_len - len(state.events[claim])

    state.total_events -= n_pruned
    return n_pruned


def compute_source_attribution(
    beliefs: torch.Tensor,
    shares: torch.Tensor,
    network_edges: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    claim_idx: int,
) -> torch.Tensor:
    """
    Compute likely source for each potential adopter.

    Returns: (n_agents,) tensor of most likely source agent indices
    """
    src_idx, dst_idx, weights = network_edges
    n_agents = beliefs.shape[0]

    # Exposure from each source to each target
    source_exposure = torch.zeros(n_agents, n_agents, device=beliefs.device)

    # Shares from source weighted by edge
    share_mask = shares[:, claim_idx] > 0.5
    for i in range(len(src_idx)):
        src = src_idx[i]
        dst = dst_idx[i]
        if share_mask[src]:
            source_exposure[dst, src] += weights[i] * beliefs[src, claim_idx]

    # Most likely source for each agent
    likely_source = source_exposure.argmax(dim=1)

    # Mark as -1 if no exposure
    max_exposure = source_exposure.max(dim=1).values
    likely_source = torch.where(
        max_exposure > 0.01,
        likely_source,
        torch.full_like(likely_source, -2)
    )

    return likely_source


def get_cascade_summary(
    state: CascadeState,
    claim: int,
) -> Dict[str, float]:
    """Get summary statistics for a claim's cascade."""
    events = state.events.get(claim, [])

    if len(events) == 0:
        return {
            "total_adoptions": 0,
            "max_generation": 0,
            "mean_generation": 0.0,
            "cascade_duration": 0,
            "seed_count": 0,
        }

    generations = [e.generation for e in events]
    times = [e.time for e in events]
    seeds = [e for e in events if e.source == -1]

    return {
        "total_adoptions": len(events),
        "max_generation": max(generations),
        "mean_generation": sum(generations) / len(generations),
        "cascade_duration": max(times) - min(times),
        "seed_count": len(seeds),
    }
