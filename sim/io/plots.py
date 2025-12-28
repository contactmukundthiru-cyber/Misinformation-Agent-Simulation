from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_adoption_curves(metrics: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    for claim in sorted(metrics["claim"].unique()):
        subset = metrics[metrics["claim"] == claim]
        ax.plot(subset["day"], subset["adoption_fraction"], label=f"claim {claim}")
    ax.set_title("Adoption Curves")
    ax.set_xlabel("Day")
    ax.set_ylabel("Adoption Fraction")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "adoption_curves.png", dpi=150)
    plt.close(fig)


def plot_polarization(metrics: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    for claim in sorted(metrics["claim"].unique()):
        subset = metrics[metrics["claim"] == claim]
        ax.plot(subset["day"], subset["polarization"], label=f"claim {claim}")
    ax.set_title("Polarization Over Time")
    ax.set_xlabel("Day")
    ax.set_ylabel("Polarization")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "polarization.png", dpi=150)
    plt.close(fig)


def plot_belief_histogram(beliefs: pd.DataFrame, out_dir: str | Path, day: int) -> None:
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    claim_cols = [c for c in beliefs.columns if c.startswith("claim_")]
    for col in claim_cols:
        ax.hist(beliefs[col], bins=20, alpha=0.4, label=col)
    ax.set_title(f"Belief Distributions (Day {day})")
    ax.set_xlabel("Belief")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"belief_hist_day_{day}.png", dpi=150)
    plt.close(fig)
