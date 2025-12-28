from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from sim.config import WorldConfig
from sim.disease.strains import Strain


def compute_social_exposure(
    shares: torch.Tensor,
    edges: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_agents: int,
) -> torch.Tensor:
    src_idx, dst_idx, weights = edges
    contributions = shares[src_idx] * weights.unsqueeze(1)
    exposures = torch.zeros((n_agents, shares.shape[1]), device=shares.device, dtype=shares.dtype)
    exposures.index_add_(0, dst_idx, contributions)
    return exposures


def compute_social_proof(
    beliefs: torch.Tensor,
    edges: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    neighbor_weight_sum: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    src_idx, dst_idx, weights = edges
    high = (beliefs > threshold).to(beliefs.dtype)
    contributions = high[src_idx] * weights.unsqueeze(1)
    proof = torch.zeros_like(beliefs)
    proof.index_add_(0, dst_idx, contributions)
    return proof / neighbor_weight_sum.unsqueeze(1)


def compute_institution_exposure(
    media_diet: torch.Tensor,
    trust: Dict[str, torch.Tensor],
    strains: List[Strain],
    world_cfg: WorldConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = media_diet.device
    dtype = media_diet.dtype
    n_agents = media_diet.shape[0]
    n_claims = len(strains)

    channel_idx = {
        "local_social": 0,
        "national_social": 1,
        "tv": 2,
        "local_news": 3,
        "church": 4,
    }

    local_news = media_diet[:, channel_idx["local_news"]]
    tv = media_diet[:, channel_idx["tv"]]
    church = media_diet[:, channel_idx["church"]]
    local_social = media_diet[:, channel_idx["local_social"]]
    national_social = media_diet[:, channel_idx["national_social"]]

    base_media = (
        world_cfg.local_media_reach * local_news + world_cfg.national_media_reach * tv
    ) * (1 + 0.5 * world_cfg.media_fragmentation)
    church_media = world_cfg.church_centrality * church
    social_media = world_cfg.feed_injection_rate * (
        local_social + national_social * (1 + world_cfg.algorithmic_amplification)
    )

    exposure = torch.zeros((n_agents, n_claims), device=device, dtype=dtype)
    debunk = torch.zeros_like(exposure)

    for k, strain in enumerate(strains):
        topic_bias = 1.0
        if "moral" in strain.topic or "spiritual" in strain.topic:
            topic_bias += 0.35
        outrage = strain.emotional_profile.get("anger", 0.0)
        feed_boost = social_media * (1 + world_cfg.outrage_amplification * outrage)

        exposure[:, k] = strain.memeticity * (base_media + church_media * topic_bias + feed_boost)

        debunk_base = (
            trust["trust_gov"] * world_cfg.gov_reach
            + trust["trust_local_news"] * world_cfg.local_media_reach
            + trust["trust_national_news"] * world_cfg.national_media_reach
        )
        response = 0.5 + 0.5 * world_cfg.governance_response_speed
        transparency = 0.5 + 0.5 * world_cfg.governance_transparency
        debunk[:, k] = (
            world_cfg.debunk_intensity
            * debunk_base
            * response
            * transparency
            * strain.falsifiability
            * (1 - strain.stealth)
        )

    return exposure, debunk
