import numpy as np
import torch

from sim.metrics.metrics import compute_daily_metrics_torch


def test_metrics_torch_basic():
    beliefs = torch.tensor([[0.1, 0.8], [0.9, 0.2]], dtype=torch.float32)
    prev = torch.tensor([[0.1, 0.7], [0.8, 0.1]], dtype=torch.float32)
    trust = {
        "trust_gov": torch.tensor([0.5, 0.6]),
        "trust_church": torch.tensor([0.4, 0.5]),
        "trust_local_news": torch.tensor([0.6, 0.5]),
        "trust_national_news": torch.tensor([0.4, 0.4]),
        "trust_friends": torch.tensor([0.7, 0.7]),
    }
    communities = np.array([0, 1])
    prev_new = torch.zeros(2)

    metrics, new = compute_daily_metrics_torch(
        0, beliefs, prev, trust, 0.7, communities, prev_new, True
    )
    assert len(metrics) == 2
    assert new.shape[0] == 2
