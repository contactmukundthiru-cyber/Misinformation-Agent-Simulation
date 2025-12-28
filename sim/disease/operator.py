from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OperatorPolicy:
    """
    Research-only stub for adversarial operator policies.

    By default, this policy is disabled and returns inputs unchanged.
    """

    enabled: bool = False

    def apply(self, exposure: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return exposure
        return exposure


def load_operator(enabled: bool = False) -> OperatorPolicy:
    return OperatorPolicy(enabled=enabled)
