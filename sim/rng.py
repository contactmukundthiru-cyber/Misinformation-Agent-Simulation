from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RNGManager:
    seed: int
    deterministic: bool = True

    def __post_init__(self) -> None:
        self.numpy = np.random.default_rng(self.seed)
        self.torch_cpu = torch.Generator(device="cpu").manual_seed(self.seed)
        if torch.cuda.is_available():
            self.torch_cuda = torch.Generator(device="cuda").manual_seed(self.seed)
        else:
            self.torch_cuda = None
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    def torch(self, device: torch.device) -> torch.Generator:
        if device.type == "cuda" and self.torch_cuda is not None:
            return self.torch_cuda
        return self.torch_cpu
