from __future__ import annotations

from pathlib import Path
from typing import List


def available_worlds(config_dir: str | Path) -> List[str]:
    path = Path(config_dir)
    return sorted([p.name for p in path.glob("world_*.yaml")])
