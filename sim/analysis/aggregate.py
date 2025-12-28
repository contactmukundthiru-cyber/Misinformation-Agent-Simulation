from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def aggregate_metrics(
    runs: Iterable[Tuple[str, pd.DataFrame]],
    metric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run_name, frame in runs:
        frame = frame.copy()
        frame["run"] = run_name
        rows.append(frame)

    combined = pd.concat(rows, ignore_index=True)
    grouped = combined.groupby(["day", "claim"])
    agg = grouped[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(col).rstrip("_") for col in agg.columns]

    n_runs = combined["run"].nunique()
    for col in metric_cols:
        std_col = f"{col}_std"
        ci_col = f"{col}_ci95"
        agg[ci_col] = 1.96 * agg[std_col].fillna(0) / max(np.sqrt(n_runs), 1)

    summary_rows = []
    for run_name, frame in runs:
        for claim in sorted(frame["claim"].unique()):
            subset = frame[frame["claim"] == claim]
            peak_idx = subset["adoption_fraction"].idxmax()
            summary_rows.append(
                {
                    "run": run_name,
                    "claim": claim,
                    "peak_day": int(subset.loc[peak_idx, "day"]),
                    "peak_adoption": float(subset.loc[peak_idx, "adoption_fraction"]),
                    "final_adoption": float(subset.iloc[-1]["adoption_fraction"]),
                    "final_polarization": float(subset.iloc[-1]["polarization"]),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    numeric_cols = ["peak_day", "peak_adoption", "final_adoption", "final_polarization"]
    summary_grouped = summary_df.groupby("claim")[numeric_cols].agg(["mean", "std"]).reset_index()
    summary_grouped.columns = ["_".join(col).rstrip("_") for col in summary_grouped.columns]
    return agg, summary_grouped
