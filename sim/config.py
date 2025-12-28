from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError


class BetaDist(BaseModel):
    alpha: float = 2.0
    beta: float = 2.0


class SimConfig(BaseModel):
    n_agents: int = 1000
    n_claims: int = 5
    steps: int = 365
    snapshot_interval: int = 30
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    deterministic: bool = True
    seed_fraction: float = 0.01
    adoption_threshold: float = 0.7
    use_tf32: bool = False


class TownConfig(BaseModel):
    n_neighborhoods: int = 12
    neighborhood_grid: Tuple[int, int] = (3, 4)
    household_size_mean: float = 3.0
    household_size_std: float = 1.0
    workplace_size_mean: float = 18.0
    school_size_mean: float = 22.0
    church_size_mean: float = 40.0
    church_attendance_rate: float = 0.25
    min_age: int = 0
    max_age: int = 90
    children_fraction: float = 0.22
    senior_fraction: float = 0.16
    education_levels: List[str] = ["low", "mid", "high"]
    occupation_types: List[str] = ["service", "office", "manual", "student", "retired"]


class TraitConfig(BaseModel):
    personality: BetaDist = BetaDist(alpha=2.2, beta=2.2)
    cognitive: BetaDist = BetaDist(alpha=2.0, beta=2.4)
    social: BetaDist = BetaDist(alpha=2.1, beta=2.1)
    emotion: BetaDist = BetaDist(alpha=2.0, beta=2.0)


class NetworkConfig(BaseModel):
    homophily_strength: float = 0.6
    geography_strength: float = 0.5
    geo_scale: float = 1.5
    intra_neighborhood_p: float = 0.05
    inter_neighborhood_p: float = 0.01
    layer_multipliers: Dict[str, float] = {
        "family": 1.6,
        "work": 1.1,
        "school": 1.0,
        "church": 1.2,
        "neighborhood": 0.8,
    }


class WorldConfig(BaseModel):
    trust_baselines: Dict[str, float] = {
        "gov": 0.55,
        "church": 0.45,
        "local_news": 0.5,
        "national_news": 0.45,
        "friends": 0.6,
        "outgroups": 0.3,
    }
    trust_variance: float = 0.12
    church_centrality: float = 0.4
    local_media_reach: float = 0.4
    national_media_reach: float = 0.35
    gov_reach: float = 0.3
    algorithmic_amplification: float = 0.4
    moderation_strictness: float = 0.4
    platform_friction: float = 0.2
    governance_response_speed: float = 0.5
    governance_transparency: float = 0.5
    media_fragmentation: float = 0.4
    outrage_amplification: float = 0.3
    feed_injection_rate: float = 0.15
    debunk_intensity: float = 0.25
    emotions_enabled: bool = True
    reactance_enabled: bool = False
    trust_update_enabled: bool = False
    trust_erosion_rate: float = 0.02
    operator_enabled: bool = False
    intervention_day: int | None = None
    intervention_strength: float = 0.0
    intervention_type: Literal["moderation", "debunk"] = "moderation"


class BeliefUpdateConfig(BaseModel):
    eta: float = 0.25
    rho: float = 0.18
    alpha: float = 0.9
    beta: float = 0.6
    gamma: float = 0.4
    delta: float = 0.5
    lambda_skepticism: float = 0.7
    mu_debunk: float = 0.8
    exposure_memory_decay: float = 0.75
    belief_decay: float = 0.02
    baseline_belief: float = 0.05
    social_proof_threshold: float = 0.6
    reactance_strength: float = 0.3


class SharingConfig(BaseModel):
    base_share_rate: float = 0.02
    belief_sensitivity: float = 2.2
    emotion_sensitivity: float = 0.6
    status_sensitivity: float = 0.5
    conformity_sensitivity: float = 0.4
    moderation_risk_sensitivity: float = 0.5


class ModerationConfig(BaseModel):
    warning_effect: float = 0.35
    downrank_effect: float = 0.4


class StrainConfig(BaseModel):
    name: str
    topic: str
    memeticity: float
    emotional_profile: Dict[str, float]
    falsifiability: float
    stealth: float
    mutation_rate: float
    violation_risk: float


class OutputConfig(BaseModel):
    save_plots: bool = True
    save_snapshots: bool = True


class MetricsConfig(BaseModel):
    community_backend: Literal["auto", "networkx", "igraph", "none"] = "auto"
    community_max_nodes: int = 20000
    include_neighborhood_clusters: bool = True
    cluster_penetration_enabled: bool = True
    use_gpu_metrics: bool = True


class SimulationConfig(BaseModel):
    sim: SimConfig = SimConfig()
    town: TownConfig = TownConfig()
    traits: TraitConfig = TraitConfig()
    network: NetworkConfig = NetworkConfig()
    world: WorldConfig = WorldConfig()
    belief_update: BeliefUpdateConfig = BeliefUpdateConfig()
    sharing: SharingConfig = SharingConfig()
    moderation: ModerationConfig = ModerationConfig()
    strains: List[StrainConfig] = Field(default_factory=list)
    metrics: MetricsConfig = MetricsConfig()
    output: OutputConfig = OutputConfig()


class ConfigError(Exception):
    pass


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> SimulationConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    base_path = data.get("base")
    if base_path:
        base_data = yaml.safe_load((path.parent / base_path).read_text()) or {}
        data = deep_merge(base_data, data)
        data.pop("base", None)
    try:
        return SimulationConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def dump_config(cfg: SimulationConfig, path: str | Path) -> None:
    path = Path(path)
    path.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
