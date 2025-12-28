import numpy as np

from sim.config import SimulationConfig
from sim.town.generator import generate_town
from sim.rng import RNGManager


def test_generate_town_shapes():
    cfg = SimulationConfig()
    rng = RNGManager(123).numpy
    town = generate_town(rng, 200, cfg.town, cfg.traits, cfg.world, cfg.network)
    assert town.neighborhood_ids.shape[0] == 200
    assert town.household_ids.shape[0] == 200
    assert town.traits.personality.shape == (200, 5)
    assert np.all(town.media_diet.weights.sum(axis=1) > 0.99)
