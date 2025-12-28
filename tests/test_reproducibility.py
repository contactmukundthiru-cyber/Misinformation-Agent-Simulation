from pathlib import Path

from sim.config import SimulationConfig
from sim.simulation import run_simulation


def test_reproducibility(tmp_path: Path):
    cfg = SimulationConfig()
    cfg.sim.n_agents = 200
    cfg.sim.steps = 10
    cfg.sim.snapshot_interval = 5
    cfg.sim.seed = 99
    cfg.sim.device = "cpu"
    cfg.output.save_plots = False
    cfg.output.save_snapshots = False

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    metrics1 = run_simulation(cfg, out1).metrics
    metrics2 = run_simulation(cfg, out2).metrics

    assert metrics1.equals(metrics2)
