from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml


def parse_args() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args, _ = parser.parse_known_args()
    return Path(args.run)


def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text()) or {}


def load_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    return yaml.safe_load(meta_path.read_text()) or {}


def plot_curve(metrics: pd.DataFrame, metric: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    for claim in sorted(metrics["claim"].unique()):
        subset = metrics[metrics["claim"] == claim]
        ax.plot(subset["day"], subset[metric], label=f"claim {claim}")
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)


def plot_histogram(snapshot: pd.DataFrame, day: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    claim_cols = [c for c in snapshot.columns if c.startswith("claim_")]
    for col in claim_cols:
        ax.hist(snapshot[col], bins=20, alpha=0.4, label=col)
    ax.set_title(f"Belief Distribution (Day {day})")
    ax.set_xlabel("Belief")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)


def main() -> None:
    run_dir = parse_args()
    st.title("Town Misinformation Simulator Dashboard")
    st.caption(str(run_dir))

    metrics_path = run_dir / "daily_metrics.csv"
    if not metrics_path.exists():
        st.error("Run directory missing daily_metrics.csv")
        return

    metrics = pd.read_csv(metrics_path)
    plot_curve(metrics, "adoption_fraction", "Adoption Curves")
    plot_curve(metrics, "polarization", "Polarization")

    snapshots_path = run_dir / "belief_snapshots.parquet"
    csv_path = run_dir / "belief_snapshots.csv"
    if snapshots_path.exists():
        snapshots = pd.read_parquet(snapshots_path)
    elif csv_path.exists():
        snapshots = pd.read_csv(csv_path)
    else:
        snapshots = None

    if snapshots is not None:
        days = sorted(snapshots["day"].unique().tolist())
        day = st.slider("Snapshot Day", min_value=int(min(days)), max_value=int(max(days)), value=int(max(days)))
        snapshot = snapshots[snapshots["day"] == day].drop(columns=["day", "agent_id"])
        plot_histogram(snapshot, day)

    config = load_config(run_dir)
    if config:
        st.subheader("World Parameters")
        world = config.get("world", {})
        st.json(world)

    metadata = load_metadata(run_dir)
    if metadata:
        st.subheader("Run Metadata")
        st.json(metadata)

    community_path = run_dir / "community_sizes.csv"
    if community_path.exists():
        st.subheader("Community Sizes")
        communities = pd.read_csv(community_path)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(communities["size"], bins=20, color="#3b7")
        ax.set_xlabel("Community Size")
        ax.set_ylabel("Count")
        ax.set_title("Community Size Distribution")
        fig.tight_layout()
        st.pyplot(fig)


if __name__ == "__main__":
    main()
