import torch

from sim.config import BeliefUpdateConfig
from sim.disease.belief_update_torch import update_beliefs


def test_belief_update_increases_with_exposure():
    cfg = BeliefUpdateConfig()
    beliefs = torch.full((5, 2), 0.1)
    exposure = torch.ones((5, 2)) * 0.8
    trust_signal = torch.ones((5, 2)) * 0.5
    social_proof = torch.zeros((5, 2))
    debunk = torch.zeros((5, 2))
    skepticism = torch.zeros(5)
    match = torch.ones((5, 2)) * 0.5
    exposure_memory = torch.zeros((5, 2))
    baseline = torch.full((5, 2), 0.05)
    reactance = torch.zeros(5)

    updated, _ = update_beliefs(
        beliefs,
        exposure,
        trust_signal,
        social_proof,
        debunk,
        skepticism,
        match,
        exposure_memory,
        baseline,
        cfg,
        False,
        reactance,
    )
    assert torch.all(updated > beliefs)
