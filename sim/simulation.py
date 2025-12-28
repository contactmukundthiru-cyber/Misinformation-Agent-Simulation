from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from sim.config import MetricsConfig, SimulationConfig
from sim.disease.belief_update_torch import update_beliefs
from sim.disease.exposure import (
    compute_institution_exposure,
    compute_social_exposure,
    compute_social_proof,
)
from sim.disease.sharing import compute_share_probabilities
from sim.disease.operator import load_operator
from sim.disease.strains import load_strains, mutate_strains
from sim.io.metadata import build_run_metadata
from sim.io.plots import plot_adoption_curves, plot_belief_histogram, plot_polarization
from sim.io.snapshot import collect_snapshot
from sim.metrics.metrics import compute_daily_metrics, compute_daily_metrics_torch
from sim.rng import RNGManager
from sim.town.generator import generate_town
from sim.world.institutions import update_trust
from sim.world.media import feed_injection
from sim.world.moderation import apply_moderation


@dataclass
class SimulationOutputs:
    metrics: pd.DataFrame
    snapshots: pd.DataFrame
    summary: Dict[str, float]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def claim_alignment(strains) -> List[float]:
    mapping = {
        "health_rumor": 0.5,
        "economic_panic": 0.55,
        "moral_spiral": 0.7,
        "tech_conspiracy": 0.6,
        "outsider_threat": 0.8,
    }
    return [mapping.get(s.topic, 0.55) for s in strains]


def detect_communities(
    src_idx: np.ndarray, dst_idx: np.ndarray, n_agents: int, cfg: MetricsConfig
) -> np.ndarray | None:
    if cfg.community_backend == "none" or n_agents > cfg.community_max_nodes:
        return None

    backend = cfg.community_backend
    if backend in ("auto", "igraph"):
        try:
            import igraph as ig

            graph = ig.Graph(n=n_agents, edges=list(zip(src_idx.tolist(), dst_idx.tolist())), directed=False)
            communities = graph.community_multilevel()
            return np.array(communities.membership, dtype=np.int32)
        except Exception:
            if backend == "igraph":
                return None

    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(n_agents))
        graph.add_edges_from(zip(src_idx.tolist(), dst_idx.tolist()))
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
        labels = np.zeros(n_agents, dtype=np.int32)
        for cid, members in enumerate(communities):
            for node in members:
                labels[node] = cid
        return labels
    except Exception:
        return None


def run_simulation(cfg: SimulationConfig, out_dir: str | Path) -> SimulationOutputs:
    """Run the contagion simulation and write outputs to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.sim.device)
    rng_manager = RNGManager(cfg.sim.seed, cfg.sim.deterministic)

    strains = load_strains(cfg.strains)
    if cfg.sim.n_claims and cfg.sim.n_claims != len(strains):
        if cfg.sim.n_claims < len(strains):
            strains = strains[: cfg.sim.n_claims]
        else:
            strains = strains + strains[: cfg.sim.n_claims - len(strains)]
    n_claims = len(strains)
    n_agents = cfg.sim.n_agents
    operator = load_operator(cfg.world.operator_enabled)

    metadata = build_run_metadata(cfg, device)
    with (out_dir / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    town = generate_town(
        rng_manager.numpy,
        n_agents,
        cfg.town,
        cfg.traits,
        cfg.world,
        cfg.network,
    )

    beliefs = torch.full(
        (n_agents, n_claims),
        fill_value=cfg.belief_update.baseline_belief,
        dtype=torch.float32,
        device=device,
    )
    seed_frac = cfg.sim.seed_fraction
    for k in range(n_claims):
        seeds = rng_manager.numpy.choice(n_agents, size=max(1, int(seed_frac * n_agents)), replace=False)
        beliefs[seeds, k] = 0.85

    baseline = torch.full_like(beliefs, fill_value=cfg.belief_update.baseline_belief)
    exposure_memory = torch.zeros_like(beliefs)

    src_idx, dst_idx, weights = town.aggregate_edges
    edge_tensors = (
        torch.tensor(src_idx, device=device, dtype=torch.int64),
        torch.tensor(dst_idx, device=device, dtype=torch.int64),
        torch.tensor(weights, device=device, dtype=torch.float32),
    )
    neighbor_weight_sum = torch.tensor(town.neighbor_weight_sum, device=device, dtype=torch.float32)

    traits = {
        "skepticism": torch.tensor(town.traits.skepticism, device=device),
        "conformity": torch.tensor(town.traits.conformity, device=device),
        "status_seeking": torch.tensor(town.traits.status_seeking, device=device),
        "conflict_tolerance": torch.tensor(town.traits.conflict_tolerance, device=device),
    }

    emotions = {}
    if cfg.world.emotions_enabled and town.traits.emotions:
        emotions = {k: torch.tensor(v, device=device) for k, v in town.traits.emotions.items()}

    trust = {
        "trust_gov": torch.tensor(town.trust.trust_gov, device=device),
        "trust_church": torch.tensor(town.trust.trust_church, device=device),
        "trust_local_news": torch.tensor(town.trust.trust_local_news, device=device),
        "trust_national_news": torch.tensor(town.trust.trust_national_news, device=device),
        "trust_friends": torch.tensor(town.trust.trust_friends, device=device),
        "trust_outgroups": torch.tensor(town.trust.trust_outgroups, device=device),
    }
    media_diet = torch.tensor(town.media_diet.weights, device=device)

    ideology = torch.tensor(town.ideology, device=device)
    alignment_targets = torch.tensor(claim_alignment(strains), device=device)
    match = 1 - torch.abs(ideology.unsqueeze(1) - alignment_targets.unsqueeze(0))

    adoption_threshold = cfg.sim.adoption_threshold
    communities = detect_communities(src_idx, dst_idx, n_agents, cfg.metrics)
    if communities is None:
        if cfg.metrics.include_neighborhood_clusters:
            combined_clusters = town.neighborhood_ids
        else:
            combined_clusters = np.arange(n_agents, dtype=np.int32)
    else:
        if cfg.metrics.include_neighborhood_clusters:
            combined_clusters = town.neighborhood_ids * 1000 + communities
        else:
            combined_clusters = communities
    if not cfg.metrics.cluster_penetration_enabled:
        combined_clusters = None
    cluster_sizes = pd.DataFrame(
        {
            "cluster_id": combined_clusters,
            "neighborhood_id": town.neighborhood_ids,
        }
    )
    cluster_summary = (
        cluster_sizes.groupby("cluster_id")
        .agg(size=("cluster_id", "count"), neighborhood_id=("neighborhood_id", "first"))
        .reset_index()
    )
    cluster_summary.to_csv(out_dir / "community_sizes.csv", index=False)

    metrics_rows: List[Dict[str, float]] = []
    snapshots: List[pd.DataFrame] = []
    prev_beliefs = beliefs.clone()
    prev_new_adopters = torch.zeros(n_claims, device=device, dtype=beliefs.dtype)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        if cfg.sim.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    for day in range(cfg.sim.steps):
        strains = mutate_strains(strains, rng_manager.numpy)

        world_effective = cfg.world
        if cfg.world.intervention_day is not None and day >= cfg.world.intervention_day:
            world_effective = cfg.world.model_copy()
            if cfg.world.intervention_type == "moderation":
                world_effective.moderation_strictness = min(
                    1.0, cfg.world.moderation_strictness * (1 + cfg.world.intervention_strength)
                )
            elif cfg.world.intervention_type == "debunk":
                world_effective.debunk_intensity = min(
                    1.0, cfg.world.debunk_intensity * (1 + cfg.world.intervention_strength)
                )

        share_probs = compute_share_probabilities(
            beliefs, traits, emotions, cfg.sharing, world_effective, strains
        )
        share_probs, warnings = apply_moderation(share_probs, strains, world_effective, cfg.moderation)

        shares = torch.bernoulli(share_probs, generator=rng_manager.torch(device))

        social_exposure = compute_social_exposure(shares, edge_tensors, n_agents)
        social_proof = compute_social_proof(
            beliefs, edge_tensors, neighbor_weight_sum, cfg.belief_update.social_proof_threshold
        )

        institution_exposure, debunk_pressure = compute_institution_exposure(
            media_diet, trust, strains, world_effective
        )
        feed_exposure = feed_injection(media_diet, strains, world_effective)

        total_exposure = social_exposure + institution_exposure + feed_exposure
        total_exposure = operator.apply(total_exposure)

        inst_trust = (
            trust["trust_gov"]
            + trust["trust_church"]
            + trust["trust_local_news"]
            + trust["trust_national_news"]
        ) / 4.0
        trust_signal = (
            social_exposure * trust["trust_friends"].unsqueeze(1)
            + institution_exposure * inst_trust.unsqueeze(1)
            + feed_exposure * trust["trust_friends"].unsqueeze(1)
        )
        denom = total_exposure + 1e-6
        trust_signal = trust_signal / denom

        debunk_pressure = debunk_pressure + warnings.unsqueeze(0) * (
            0.5 + traits["skepticism"].unsqueeze(1)
        ) * trust["trust_local_news"].unsqueeze(1)

        beliefs, exposure_memory = update_beliefs(
            beliefs,
            total_exposure,
            trust_signal,
            social_proof,
            debunk_pressure,
            traits["skepticism"],
            match,
            exposure_memory,
            baseline,
            cfg.belief_update,
            world_effective.reactance_enabled,
            traits["conflict_tolerance"],
        )

        trust = update_trust(trust, beliefs, debunk_pressure, world_effective)

        if cfg.output.save_snapshots and (day == 0 or (day + 1) % cfg.sim.snapshot_interval == 0):
            belief_cpu = beliefs.detach().cpu().numpy()
            frame = pd.DataFrame(belief_cpu, columns=[f"claim_{i}" for i in range(n_claims)])
            frame.insert(0, "agent_id", np.arange(n_agents))
            collect_snapshot(snapshots, day, frame)

        if cfg.metrics.use_gpu_metrics:
            daily_metrics, prev_new_adopters = compute_daily_metrics_torch(
                day,
                beliefs,
                prev_beliefs,
                trust,
                adoption_threshold,
                combined_clusters,
                prev_new_adopters,
                cfg.metrics.cluster_penetration_enabled,
            )
        else:
            belief_cpu = beliefs.detach().cpu().numpy()
            prev_cpu = prev_beliefs.detach().cpu().numpy()
            trust_cpu = {k: v.detach().cpu().numpy() for k, v in trust.items()}
            daily_metrics, prev_new_adopters_cpu = compute_daily_metrics(
                day,
                belief_cpu,
                prev_cpu,
                trust_cpu,
                adoption_threshold,
                combined_clusters,
                prev_new_adopters.detach().cpu().numpy(),
                cfg.metrics.cluster_penetration_enabled,
            )
            prev_new_adopters = torch.tensor(prev_new_adopters_cpu, device=device, dtype=beliefs.dtype)
        metrics_rows.extend(daily_metrics)
        prev_beliefs = beliefs.clone()

    torch.set_grad_enabled(prev_grad)

    metrics_df = pd.DataFrame(metrics_rows)
    snapshots_df = pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()

    summary = build_summary(metrics_df, cfg.sim.steps, cfg.world.intervention_day)

    if cfg.output.save_plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_adoption_curves(metrics_df, plots_dir)
        plot_polarization(metrics_df, plots_dir)
        if not snapshots_df.empty:
            last_day = int(snapshots_df["day"].max())
            last_snapshot = snapshots_df[snapshots_df["day"] == last_day]
            plot_belief_histogram(last_snapshot.drop(columns=["day", "agent_id"]), plots_dir, last_day)

    metrics_df.to_csv(out_dir / "daily_metrics.csv", index=False)
    if cfg.output.save_snapshots and not snapshots_df.empty:
        try:
            snapshots_df.to_parquet(out_dir / "belief_snapshots.parquet", index=False)
        except ImportError:
            logging.warning("pyarrow not available; saving snapshots as CSV.")
            snapshots_df.to_csv(out_dir / "belief_snapshots.csv", index=False)

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return SimulationOutputs(metrics=metrics_df, snapshots=snapshots_df, summary=summary)


def build_summary(metrics: pd.DataFrame, steps: int, intervention_day: int | None) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for claim in sorted(metrics["claim"].unique()):
        subset = metrics[metrics["claim"] == claim]
        peak_idx = subset["adoption_fraction"].idxmax()
        peak_day = int(subset.loc[peak_idx, "day"])
        summary[f"claim_{claim}_peak_day"] = peak_day
        summary[f"claim_{claim}_peak_adoption"] = float(subset.loc[peak_idx, "adoption_fraction"])
        summary[f"claim_{claim}_final_adoption"] = float(subset.iloc[-1]["adoption_fraction"])
        summary[f"claim_{claim}_final_polarization"] = float(subset.iloc[-1]["polarization"])
        if intervention_day is not None:
            pre = subset[(subset["day"] >= max(intervention_day - 30, 0)) & (subset["day"] < intervention_day)]
            post = subset[(subset["day"] >= intervention_day) & (subset["day"] <= intervention_day + 30)]
            if not pre.empty and not post.empty:
                effect = float(post["adoption_fraction"].mean() - pre["adoption_fraction"].mean())
            else:
                effect = 0.0
            summary[f"claim_{claim}_intervention_effect"] = effect
        else:
            summary[f"claim_{claim}_intervention_effect"] = 0.0
    summary["steps"] = steps
    return summary
