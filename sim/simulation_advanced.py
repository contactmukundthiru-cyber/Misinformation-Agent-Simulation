"""
Advanced Simulation Core
=========================
Integrates all cognitive architecture components into a unified simulation.

This is the publication-ready simulation that:
1. Uses psychologically-grounded belief dynamics
2. Evolves network structure dynamically
3. Tracks information cascades
4. Supports calibration against empirical data
5. Produces realistic adoption patterns
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Core simulation components
from sim.config import SimulationConfig, BeliefUpdateConfig
from sim.rng import RNGManager
from sim.town.generator import generate_town

# Cognitive architecture
from sim.cognition.dual_process import (
    CognitiveState,
    DualProcessConfig,
    initialize_cognitive_states,
)
from sim.cognition.motivated_reasoning import (
    IdentityConfig,
    IdentityState,
    initialize_identity_state,
)
from sim.cognition.attention import (
    AttentionConfig,
    AttentionState,
)
from sim.cognition.source_memory import (
    SourceMemory,
    SourceCredibility,
    initialize_source_memory,
)

# Dynamics
from sim.dynamics.network_evolution import (
    NetworkEvolutionConfig,
    DynamicNetworkState,
    initialize_dynamic_network,
    update_network_structure,
    detect_echo_chambers,
    measure_bridge_ties,
)
from sim.dynamics.influence import (
    InfluenceConfig,
    InfluenceState,
    initialize_influence_state,
    compute_dynamic_influence,
    update_influence_scores,
    identify_superspreaders,
)

# Cascades
from sim.cascades.tracker import (
    CascadeConfig,
    CascadeState,
    initialize_cascade_tracker,
    record_adoption_event,
)
from sim.cascades.analysis import analyze_cascade_statistics
from sim.cascades.r_effective import REffectiveTracker, compute_true_r_effective

# Narratives
from sim.narratives.competition import (
    NarrativeConfig,
    NarrativeState,
    initialize_narrative_state,
    update_narrative_bundles,
)
from sim.narratives.inoculation import (
    InoculationConfig,
    InoculationState,
    initialize_inoculation_state,
    apply_prebunking,
)

# Advanced belief update
from sim.disease.belief_update_advanced import (
    AdvancedBeliefConfig,
    CognitiveArchitecture,
    advanced_belief_update,
    compute_evidence_quality,
    compute_emotional_resonance,
)

# Existing components
from sim.disease.strains import load_strains, Strain
from sim.disease.exposure import compute_social_exposure, compute_social_proof
from sim.disease.sharing import compute_share_probabilities
from sim.world.moderation import apply_moderation
from sim.world.media import feed_injection
from sim.io.metadata import build_run_metadata
from sim.io.plots import plot_adoption_curves, plot_polarization
from sim.metrics.advanced_metrics import (
    compute_belief_entropy,
    compute_mutual_information,
    compute_polarization_index,
    compute_calibration_targets,
)


@dataclass
class AdvancedSimulationConfig:
    """Extended configuration for advanced simulation."""

    base: SimulationConfig

    # Cognitive architecture configs
    dual_process: DualProcessConfig = field(default_factory=DualProcessConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)

    # Dynamics configs
    network_evolution: NetworkEvolutionConfig = field(default_factory=NetworkEvolutionConfig)
    influence: InfluenceConfig = field(default_factory=InfluenceConfig)

    # Cascade configs
    cascade: CascadeConfig = field(default_factory=CascadeConfig)

    # Narrative configs
    narrative: NarrativeConfig = field(default_factory=NarrativeConfig)
    inoculation: InoculationConfig = field(default_factory=InoculationConfig)

    # Advanced belief update
    advanced_belief: AdvancedBeliefConfig = field(default_factory=AdvancedBeliefConfig)

    # Intervention settings
    prebunking_day: Optional[int] = None
    prebunking_fraction: float = 0.3
    accuracy_nudge_day: Optional[int] = None


@dataclass
class AdvancedSimulationOutputs:
    """Outputs from advanced simulation."""

    metrics: pd.DataFrame
    cascade_stats: Dict[str, float]
    network_evolution: pd.DataFrame
    echo_chamber_history: List[float]
    r_effective_history: Dict[int, List[float]]
    calibration_targets: Dict[str, float]
    summary: Dict[str, float]


def run_advanced_simulation(
    cfg: AdvancedSimulationConfig,
    out_dir: str | Path,
) -> AdvancedSimulationOutputs:
    """
    Run the advanced cognitive architecture simulation.

    This is the main entry point for the publication-ready simulation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = cfg.base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng_manager = RNGManager(base_cfg.sim.seed, base_cfg.sim.deterministic)

    # Load strains
    strains = load_strains(base_cfg.strains)
    n_claims = len(strains)
    n_agents = base_cfg.sim.n_agents

    logging.info(f"Starting advanced simulation: {n_agents} agents, {n_claims} claims")

    # Generate town
    town = generate_town(
        rng_manager.numpy,
        n_agents,
        base_cfg.town,
        base_cfg.traits,
        base_cfg.world,
        base_cfg.network,
    )

    # Initialize beliefs
    beliefs = torch.full(
        (n_agents, n_claims),
        fill_value=cfg.advanced_belief.baseline_belief,
        dtype=torch.float32,
        device=device,
    )

    # Seed initial believers
    seed_frac = base_cfg.sim.seed_fraction
    seed_mask = torch.zeros(n_agents, n_claims, dtype=torch.bool, device=device)
    for k in range(n_claims):
        seeds = rng_manager.numpy.choice(n_agents, size=max(1, int(seed_frac * n_agents)), replace=False)
        beliefs[seeds, k] = 0.85
        seed_mask[seeds, k] = True

    # Convert town data to tensors
    traits = {
        "skepticism": torch.tensor(town.traits.skepticism, device=device),
        "numeracy": torch.tensor(town.traits.numeracy, device=device),
        "conformity": torch.tensor(town.traits.conformity, device=device),
        "status_seeking": torch.tensor(town.traits.status_seeking, device=device),
        "need_for_closure": torch.tensor(town.traits.need_for_closure, device=device),
    }

    emotions = {
        "fear": torch.tensor(town.traits.emotions.get("fear", np.zeros(n_agents)), device=device),
        "anger": torch.tensor(town.traits.emotions.get("anger", np.zeros(n_agents)), device=device),
        "hope": torch.tensor(town.traits.emotions.get("hope", np.zeros(n_agents)), device=device),
    }

    trust = {
        "trust_gov": torch.tensor(town.trust.trust_gov, device=device),
        "trust_church": torch.tensor(town.trust.trust_church, device=device),
        "trust_local_news": torch.tensor(town.trust.trust_local_news, device=device),
        "trust_national_news": torch.tensor(town.trust.trust_national_news, device=device),
        "trust_friends": torch.tensor(town.trust.trust_friends, device=device),
        "trust_outgroups": torch.tensor(town.trust.trust_outgroups, device=device),
    }
    traits["trust_church"] = trust["trust_church"]
    traits["trust_outgroups"] = trust["trust_outgroups"]
    traits["ideology"] = torch.tensor(town.ideology, device=device)

    # Initialize dynamic network
    network_state = initialize_dynamic_network(
        town.aggregate_edges, n_agents, device
    )

    # Initialize cognitive architecture
    cognitive_state = initialize_cognitive_states(
        n_agents, n_claims, traits, cfg.dual_process, device
    )

    claim_topics = [s.topic for s in strains]
    identity_state = initialize_identity_state(
        n_agents, n_claims, traits, claim_topics, cfg.identity, device
    )

    attention_state = AttentionState.initialize(n_agents, n_claims, device)

    source_memory, source_credibility = initialize_source_memory(
        n_agents, n_claims, 6, trust, device
    )

    # Ground truth: all claims are false (misinformation)
    ground_truth = [False] * n_claims
    narrative_state = initialize_narrative_state(
        n_claims, claim_topics, ground_truth, device
    )

    inoculation_state = initialize_inoculation_state(n_agents, n_claims, device)

    # Create cognitive architecture container
    arch = CognitiveArchitecture(
        cognitive_state=cognitive_state,
        identity_state=identity_state,
        attention_state=attention_state,
        source_memory=source_memory,
        source_credibility=source_credibility,
        narrative_state=narrative_state,
        inoculation_state=inoculation_state,
    )

    # Initialize influence tracking
    influence_state = initialize_influence_state(
        n_agents, n_claims,
        (network_state.src_idx, network_state.dst_idx, network_state.weights),
        traits, cfg.influence, device
    )

    # Initialize cascade tracking
    cascade_state = initialize_cascade_tracker(
        n_agents, n_claims, seed_mask, device
    )
    r_eff_tracker = REffectiveTracker(n_claims=n_claims)

    # Build claim emotional profiles
    claim_emotions = torch.tensor([
        [s.emotional_profile.get("fear", 0), s.emotional_profile.get("anger", 0), s.emotional_profile.get("hope", 0)]
        for s in strains
    ], device=device)

    # Claim positions on identity scale
    claim_positions = torch.tensor([
        {"health_rumor": 0.4, "economic_panic": 0.6, "moral_spiral": 0.7,
         "tech_conspiracy": 0.5, "outsider_threat": 0.8}.get(s.topic, 0.5)
        for s in strains
    ], device=device)

    claim_falsifiability = torch.tensor([s.falsifiability for s in strains], device=device)

    # Metrics collection
    metrics_rows = []
    network_evolution_rows = []
    echo_chamber_history = []
    belief_history = []

    # Main simulation loop
    prev_beliefs = beliefs.clone()

    torch.set_grad_enabled(False)

    for day in range(base_cfg.sim.steps):
        # ===== INTERVENTIONS =====
        if cfg.prebunking_day is not None and day == cfg.prebunking_day:
            n_prebunked = apply_prebunking(
                inoculation_state,
                target_claims=list(range(n_claims)),
                target_fraction=cfg.prebunking_fraction,
                inoculation_type=2,  # Active inoculation
                cfg=cfg.inoculation,
                rng=rng_manager.torch(device),
            )
            logging.info(f"Day {day}: Prebunked {n_prebunked} agents")

        # ===== SHARING BEHAVIOR =====
        share_probs = compute_share_probabilities(
            beliefs, traits, emotions, base_cfg.sharing, base_cfg.world, strains
        )
        share_probs, warnings = apply_moderation(
            share_probs, strains, base_cfg.world, base_cfg.moderation
        )
        shares = torch.bernoulli(share_probs, generator=rng_manager.torch(device))

        # ===== EXPOSURE COMPUTATION =====
        edge_tensors = (network_state.src_idx, network_state.dst_idx, network_state.weights)

        # Influence-weighted exposure
        influenced_exposure = compute_dynamic_influence(
            influence_state, beliefs, shares, edge_tensors
        )

        social_exposure = compute_social_exposure(shares, edge_tensors, n_agents)
        social_proof = compute_social_proof(
            beliefs, edge_tensors,
            network_state.in_degree + 1e-6,
            base_cfg.belief_update.social_proof_threshold
        )

        feed_exposure = feed_injection(
            torch.tensor(town.media_diet.weights, device=device),
            strains, base_cfg.world
        )

        total_exposure = social_exposure + influenced_exposure + feed_exposure

        # ===== COMPUTE SUPPORTING SIGNALS =====
        emotional_resonance = compute_emotional_resonance(emotions, claim_emotions)

        mean_trust = (trust["trust_friends"] + trust["trust_local_news"] + trust["trust_national_news"]) / 3
        source_credibility_signal = mean_trust.unsqueeze(1).expand(-1, n_claims)

        evidence_quality = compute_evidence_quality(
            source_credibility_signal, claim_falsifiability, social_proof
        )

        debunk_pressure = warnings.unsqueeze(0).expand(n_agents, -1) * 0.3

        # ===== ADVANCED BELIEF UPDATE =====
        beliefs, diagnostics = advanced_belief_update(
            beliefs=beliefs,
            exposure=total_exposure,
            social_proof=social_proof,
            emotional_resonance=emotional_resonance,
            evidence_quality=evidence_quality,
            source_credibility_signal=source_credibility_signal,
            debunk_pressure=debunk_pressure,
            traits=traits,
            claim_positions=claim_positions,
            arch=arch,
            cfg=cfg.advanced_belief,
            dual_cfg=cfg.dual_process,
            identity_cfg=cfg.identity,
            attention_cfg=cfg.attention,
            narrative_cfg=cfg.narrative,
            inoculation_cfg=cfg.inoculation,
        )

        # ===== UPDATE NETWORK STRUCTURE =====
        if day % 7 == 0:  # Update weekly for efficiency
            network_stats = update_network_structure(
                network_state, beliefs, cfg.network_evolution, rng_manager.torch(device)
            )
            network_evolution_rows.append({
                "day": day,
                "edges_dissolved": network_stats["dissolved"],
                "edges_formed": network_stats["formed"],
                "n_edges": len(network_state.src_idx),
            })

            # Detect echo chambers
            clusters, modularity = detect_echo_chambers(
                network_state, beliefs, cfg.network_evolution
            )
            echo_chamber_history.append(modularity)

            # Measure bridge ties
            bridge_mask, bridge_fraction = measure_bridge_ties(network_state, beliefs)

        # ===== UPDATE CASCADES =====
        n_new_adoptions = record_adoption_event(
            cascade_state, day, beliefs, prev_beliefs,
            None, base_cfg.sim.adoption_threshold, cfg.cascade
        )

        # Compute R_effective
        r_eff = compute_true_r_effective(r_eff_tracker, cascade_state, day)

        # ===== UPDATE INFLUENCE SCORES =====
        ground_truth_tensor = narrative_state.ground_truth
        update_influence_scores(
            influence_state, beliefs, prev_beliefs, ground_truth_tensor,
            shares, edge_tensors, cfg.influence
        )

        # ===== UPDATE NARRATIVES =====
        update_narrative_bundles(narrative_state, beliefs, shares, cfg.narrative)

        # ===== COLLECT METRICS =====
        adoption_mask = beliefs >= base_cfg.sim.adoption_threshold
        adoption_fraction = adoption_mask.float().mean(dim=0)
        mean_belief = beliefs.mean(dim=0)
        variance = beliefs.var(dim=0)

        # Advanced metrics
        entropy = compute_belief_entropy(beliefs)
        if network_state.cluster_assignments is not None:
            mi = compute_mutual_information(beliefs, network_state.cluster_assignments)
        else:
            mi = torch.zeros(n_claims, device=device)
        polarization = compute_polarization_index(beliefs)

        for k in range(n_claims):
            metrics_rows.append({
                "day": day,
                "claim": k,
                "adoption_fraction": float(adoption_fraction[k]),
                "mean_belief": float(mean_belief[k]),
                "variance": float(variance[k]),
                "polarization": float(polarization["bimodality"][k]),  # For plot compatibility
                "entropy": float(entropy[k]),
                "mutual_information": float(mi[k]),
                "polarization_bimodality": float(polarization["bimodality"][k]),
                "polarization_esteban_ray": float(polarization["esteban_ray"][k]),
                "r_effective": r_eff.get(k, 0.0),
                "n_new_adoptions": n_new_adoptions,
            })

        # Store for history
        belief_history.append(beliefs.clone())
        if len(belief_history) > 30:
            belief_history = belief_history[-30:]

        prev_beliefs = beliefs.clone()

        if day % 50 == 0:
            logging.info(f"Day {day}: adoption={float(adoption_fraction.mean()):.3f}")

    # ===== POST-SIMULATION ANALYSIS =====
    metrics_df = pd.DataFrame(metrics_rows)
    network_evolution_df = pd.DataFrame(network_evolution_rows)

    cascade_stats = analyze_cascade_statistics(cascade_state)

    calibration_targets = compute_calibration_targets(
        beliefs, base_cfg.sim.adoption_threshold
    )

    # Summary statistics
    summary = {
        "final_adoption_mean": float(adoption_fraction.mean()),
        "final_belief_mean": float(mean_belief.mean()),
        "final_entropy_mean": float(entropy.mean()),
        "max_modularity": max(echo_chamber_history) if echo_chamber_history else 0,
        **cascade_stats,
        **calibration_targets,
    }

    # Save outputs
    metrics_df.to_csv(out_dir / "daily_metrics.csv", index=False)
    network_evolution_df.to_csv(out_dir / "network_evolution.csv", index=False)

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "cascade_stats.json").open("w") as f:
        json.dump(cascade_stats, f, indent=2)

    # Generate plots
    if base_cfg.output.save_plots:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plot_adoption_curves(metrics_df, plots_dir)
        plot_polarization(metrics_df, plots_dir)

    logging.info(f"Simulation complete. Final adoption: {summary['final_adoption_mean']:.3f}")

    return AdvancedSimulationOutputs(
        metrics=metrics_df,
        cascade_stats=cascade_stats,
        network_evolution=network_evolution_df,
        echo_chamber_history=echo_chamber_history,
        r_effective_history={k: r_eff_tracker.r_eff_history.get(k, []) for k in range(n_claims)},
        calibration_targets=calibration_targets,
        summary=summary,
    )
