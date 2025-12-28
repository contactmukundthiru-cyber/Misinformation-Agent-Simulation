import numpy as np

from sim.config import SimulationConfig
from sim.town.generator import generate_town
from sim.rng import RNGManager


def test_network_edges():
    cfg = SimulationConfig()
    rng = RNGManager(456).numpy
    town = generate_town(rng, 300, cfg.town, cfg.traits, cfg.world, cfg.network)
    src, dst, weights = town.aggregate_edges
    assert len(src) == len(dst) == len(weights)
    assert np.all(weights > 0)
    assert np.all(town.neighbor_weight_sum > 0)
