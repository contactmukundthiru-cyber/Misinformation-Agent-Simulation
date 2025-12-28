"""
Source Memory and Credibility Tracking
=======================================
Models how agents track and evaluate information sources.

Key mechanisms:
1. Source Memory: Track who told you what
2. Credibility Assessment: Build source reputation over time
3. Source Confusion: Memory decay can lead to source misattribution
4. Credibility Transfer: Trust in source affects belief acceptance
5. Echo Detection: Recognize when multiple sources share same origin

References:
- Johnson, M. K., et al. (1993). Source monitoring
- Guillory, J. J., & Geraci, L. (2013). Correcting erroneous inferences
- Fazio, L. K., et al. (2015). Knowledge does not protect against illusory truth
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class SourceCredibility:
    """Tracks credibility assessment for different source types."""

    n_agents: int
    n_source_types: int
    device: torch.device

    # Base credibility for each source type per agent
    # Shape: (n_agents, n_source_types)
    base_credibility: torch.Tensor

    # Dynamic credibility (changes based on experience)
    # Shape: (n_agents, n_source_types)
    dynamic_credibility: torch.Tensor

    # Accuracy history (was source right in past?)
    # Shape: (n_agents, n_source_types)
    accuracy_history: torch.Tensor

    # Confidence in credibility assessment
    # Shape: (n_agents, n_source_types)
    assessment_confidence: torch.Tensor

    @classmethod
    def initialize(
        cls,
        n_agents: int,
        n_source_types: int,
        trust_baselines: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> "SourceCredibility":
        # Build base credibility from trust baselines
        base_cred = torch.zeros(n_agents, n_source_types, device=device)

        source_mapping = {
            "trust_friends": 0,
            "trust_local_news": 1,
            "trust_national_news": 2,
            "trust_gov": 3,
            "trust_church": 4,
            "trust_outgroups": 5,
        }

        for key, idx in source_mapping.items():
            if key in trust_baselines:
                base_cred[:, idx] = trust_baselines[key]

        return cls(
            n_agents=n_agents,
            n_source_types=n_source_types,
            device=device,
            base_credibility=base_cred,
            dynamic_credibility=base_cred.clone(),
            accuracy_history=torch.full(
                (n_agents, n_source_types), 0.5, device=device
            ),
            assessment_confidence=torch.full(
                (n_agents, n_source_types), 0.3, device=device
            ),
        )


@dataclass
class SourceMemory:
    """
    Tracks exposure history with source attribution.

    Uses a compressed representation to efficiently track
    who exposed whom to which claims.
    """

    n_agents: int
    n_claims: int
    n_source_types: int
    device: torch.device

    # Total exposure from each source type for each claim
    # Shape: (n_agents, n_claims, n_source_types)
    exposure_by_source: torch.Tensor

    # Weighted exposure (by source credibility)
    # Shape: (n_agents, n_claims)
    credibility_weighted_exposure: torch.Tensor

    # First source for each claim (primacy effect)
    # Shape: (n_agents, n_claims)
    first_source: torch.Tensor

    # Number of distinct sources (independence signal)
    # Shape: (n_agents, n_claims)
    source_diversity: torch.Tensor

    # Recency-weighted source (recent sources weighted more)
    # Shape: (n_agents, n_claims, n_source_types)
    recent_source_weights: torch.Tensor

    @classmethod
    def initialize(
        cls,
        n_agents: int,
        n_claims: int,
        n_source_types: int,
        device: torch.device,
    ) -> "SourceMemory":
        return cls(
            n_agents=n_agents,
            n_claims=n_claims,
            n_source_types=n_source_types,
            device=device,
            exposure_by_source=torch.zeros(
                n_agents, n_claims, n_source_types, device=device
            ),
            credibility_weighted_exposure=torch.zeros(
                n_agents, n_claims, device=device
            ),
            first_source=torch.full(
                (n_agents, n_claims), -1, dtype=torch.long, device=device
            ),
            source_diversity=torch.zeros(n_agents, n_claims, device=device),
            recent_source_weights=torch.zeros(
                n_agents, n_claims, n_source_types, device=device
            ),
        )


def initialize_source_memory(
    n_agents: int,
    n_claims: int,
    n_source_types: int,
    trust_baselines: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[SourceMemory, SourceCredibility]:
    """Initialize source memory and credibility systems."""
    memory = SourceMemory.initialize(
        n_agents, n_claims, n_source_types, device
    )
    credibility = SourceCredibility.initialize(
        n_agents, n_source_types, trust_baselines, device
    )
    return memory, credibility


def update_source_memory(
    memory: SourceMemory,
    exposure: torch.Tensor,
    source_attribution: torch.Tensor,
    decay_rate: float = 0.95,
) -> None:
    """
    Update source memory with new exposures.

    Args:
        exposure: Exposure intensity (n_agents, n_claims)
        source_attribution: Which source type for each exposure
            Shape: (n_agents, n_claims, n_source_types) - one-hot or soft
    """
    # Decay existing memory
    memory.exposure_by_source = decay_rate * memory.exposure_by_source

    # Add new exposure with source attribution
    exposure_with_source = exposure.unsqueeze(2) * source_attribution
    memory.exposure_by_source = memory.exposure_by_source + exposure_with_source

    # Update first source (only where not yet set)
    # Find dominant source for this exposure
    dominant_source = source_attribution.argmax(dim=2)
    new_exposure_mask = (exposure > 0.1) & (memory.first_source < 0)
    memory.first_source = torch.where(
        new_exposure_mask, dominant_source, memory.first_source
    )

    # Update source diversity (count of source types with non-trivial exposure)
    source_counts = (memory.exposure_by_source > 0.1).float().sum(dim=2)
    memory.source_diversity = source_counts

    # Update recency weights (recent exposures weighted more)
    recency_decay = 0.8
    memory.recent_source_weights = (
        recency_decay * memory.recent_source_weights
        + (1.0 - recency_decay) * source_attribution
    )


def compute_source_weighted_exposure(
    memory: SourceMemory,
    credibility: SourceCredibility,
) -> torch.Tensor:
    """
    Compute exposure weighted by source credibility.

    Returns: credibility-weighted exposure (n_agents, n_claims)
    """
    # Use dynamic credibility
    cred_expanded = credibility.dynamic_credibility.unsqueeze(1)

    # Weight exposure by credibility
    weighted = memory.exposure_by_source * cred_expanded

    # Sum across source types
    total_weighted = weighted.sum(dim=2)

    # Store for later use
    memory.credibility_weighted_exposure = total_weighted

    return total_weighted


def update_source_credibility(
    credibility: SourceCredibility,
    source_type: torch.Tensor,
    accuracy_signal: torch.Tensor,
    learning_rate: float = 0.1,
) -> None:
    """
    Update source credibility based on accuracy feedback.

    When an agent learns that a source was right/wrong about something,
    they update their assessment of that source's credibility.

    Args:
        source_type: Which source type (n_agents, n_claims) or indices
        accuracy_signal: Was source right? (n_agents, n_claims) in [-1, 1]
    """
    # Update accuracy history (exponential moving average)
    for src_type in range(credibility.n_source_types):
        mask = source_type == src_type
        if mask.any():
            src_accuracy = (accuracy_signal * mask.float()).sum(dim=1)
            src_count = mask.float().sum(dim=1) + 1e-6

            avg_accuracy = src_accuracy / src_count

            # Update only for agents with this source type
            has_source = (mask.float().sum(dim=1) > 0)

            credibility.accuracy_history[:, src_type] = torch.where(
                has_source,
                (1 - learning_rate) * credibility.accuracy_history[:, src_type]
                + learning_rate * (avg_accuracy * 0.5 + 0.5),
                credibility.accuracy_history[:, src_type],
            )

    # Update dynamic credibility based on accuracy history
    credibility.dynamic_credibility = (
        0.5 * credibility.base_credibility
        + 0.5 * credibility.accuracy_history
    )

    # Update confidence (more interactions = higher confidence)
    credibility.assessment_confidence = torch.clamp(
        credibility.assessment_confidence + 0.01,
        0.0, 0.95
    )


def retrieve_source_history(
    memory: SourceMemory,
    claim_idx: int,
) -> Dict[str, torch.Tensor]:
    """
    Retrieve source history for a specific claim.

    Returns dict with:
    - exposure_by_source: Exposure from each source type
    - first_source: First source that exposed agent
    - source_diversity: Number of distinct sources
    """
    return {
        "exposure_by_source": memory.exposure_by_source[:, claim_idx, :],
        "first_source": memory.first_source[:, claim_idx],
        "source_diversity": memory.source_diversity[:, claim_idx],
        "recent_weights": memory.recent_source_weights[:, claim_idx, :],
    }


def compute_echo_chamber_signal(
    memory: SourceMemory,
    network_clusters: torch.Tensor,
) -> torch.Tensor:
    """
    Detect echo chamber effect: apparent diversity that's actually homogeneous.

    When multiple 'sources' are from the same cluster, the information
    isn't truly independent. This computes effective source diversity
    accounting for network structure.

    Returns: effective diversity (n_agents, n_claims)
    """
    # Simple approximation: if source diversity is high but mostly from
    # same source type, effective diversity is lower
    source_concentration = memory.exposure_by_source.max(dim=2).values
    total_exposure = memory.exposure_by_source.sum(dim=2) + 1e-6
    concentration_ratio = source_concentration / total_exposure

    # High concentration = echo chamber, low effective diversity
    effective_diversity = memory.source_diversity * (1.0 - concentration_ratio)

    return effective_diversity


def source_confusion_probability(
    memory: SourceMemory,
    time_since_exposure: torch.Tensor,
    base_confusion_rate: float = 0.1,
    time_decay: float = 0.95,
) -> torch.Tensor:
    """
    Compute probability of source misattribution.

    As time passes, people forget where they heard something,
    potentially misattributing the source.

    This can lead to the "sleeper effect" where a low-credibility
    source's message gains influence as source memory fades.
    """
    # Confusion increases with time
    time_factor = 1.0 - time_decay ** time_since_exposure

    # And increases when multiple sources are present
    diversity_factor = torch.tanh(memory.source_diversity * 0.5)

    confusion_prob = base_confusion_rate + 0.3 * time_factor + 0.2 * diversity_factor

    return torch.clamp(confusion_prob, 0.0, 0.8)
