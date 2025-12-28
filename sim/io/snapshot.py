from __future__ import annotations

from typing import List

import pandas as pd


def collect_snapshot(
    snapshots: List[pd.DataFrame],
    day: int,
    beliefs: pd.DataFrame,
) -> None:
    beliefs = beliefs.copy()
    beliefs.insert(0, "day", day)
    snapshots.append(beliefs)


def build_snapshot_frame(beliefs: pd.DataFrame) -> pd.DataFrame:
    return beliefs
