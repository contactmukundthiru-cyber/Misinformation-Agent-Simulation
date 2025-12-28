from pathlib import Path

from sim.config import load_config


def test_load_config():
    cfg = load_config(Path("configs/world_baseline.yaml"))
    assert cfg.sim.n_agents == 5000
    assert len(cfg.strains) == 5
    assert cfg.world.trust_baselines["gov"] > 0
