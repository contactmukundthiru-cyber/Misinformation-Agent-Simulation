from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import pandas as pd

from sim.analysis.aggregate import aggregate_metrics
from sim.config import dump_config, load_config
from sim.io.logging import setup_logging
from sim.simulation import run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sim", description="Town Misinformation Simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a single simulation")
    run.add_argument("--config", required=True, help="Path to config YAML")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--steps", type=int, default=None)
    run.add_argument("--n", type=int, default=None)
    run.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    run.add_argument("--out", required=True, help="Output directory")

    sweep = sub.add_parser("sweep", help="Run a config sweep")
    sweep.add_argument("--configs", nargs="+", required=True)
    sweep.add_argument("--seeds", nargs="+", type=int, required=True)
    sweep.add_argument("--out", required=True)

    dash = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash.add_argument("--run", required=True, help="Run directory with outputs")

    bench = sub.add_parser("bench", help="Run a performance benchmark")
    bench.add_argument("--config", required=True, help="Path to config YAML")
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--steps", type=int, default=200)
    bench.add_argument("--n", type=int, default=10000)
    bench.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    bench.add_argument("--repeat", type=int, default=3)
    bench.add_argument("--out", required=True, help="Output directory for bench logs")

    aggregate = sub.add_parser("aggregate", help="Aggregate metrics from existing runs")
    aggregate.add_argument("--runs", nargs="+", required=True, help="Run directories or glob patterns")
    aggregate.add_argument("--out", required=False, help="Output directory for aggregate CSVs")

    return parser.parse_args()


def override_config(cfg, args: argparse.Namespace) -> None:
    if args.seed is not None:
        cfg.sim.seed = args.seed
    if args.steps is not None:
        cfg.sim.steps = args.steps
    if args.n is not None:
        cfg.sim.n_agents = args.n
    if args.device is not None:
        cfg.sim.device = args.device


def run_single(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    override_config(cfg, args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config_resolved.yaml")
    run_simulation(cfg, out_dir)


def run_sweep(args: argparse.Namespace) -> None:
    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)
    metrics_frames = []
    for config_path in args.configs:
        for seed in args.seeds:
            cfg = load_config(config_path)
            cfg.sim.seed = seed
            run_name = f"{Path(config_path).stem}_seed_{seed}"
            out_dir = base_out / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            dump_config(cfg, out_dir / "config_resolved.yaml")
            logging.info("Running %s", run_name)
            outputs = run_simulation(cfg, out_dir)
            metrics = outputs.metrics.copy()
            metrics_frames.append((run_name, metrics))

    if metrics_frames:
        metric_cols = [
            "adoption_fraction",
            "mean_belief",
            "variance",
            "polarization",
            "r0_like",
            "cluster_penetration",
            "trust_gov",
            "trust_church",
            "trust_local_news",
            "trust_national_news",
            "trust_friends",
        ]
        agg, summary = aggregate_metrics(metrics_frames, metric_cols)
        agg.to_csv(base_out / "aggregate_metrics.csv", index=False)
        summary.to_csv(base_out / "aggregate_summary.csv", index=False)


def run_dashboard(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--", "--run", str(run_dir)],
        check=True,
    )


def run_bench(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg.sim.seed = args.seed
    cfg.sim.steps = args.steps
    cfg.sim.n_agents = args.n
    cfg.sim.device = args.device
    cfg.output.save_plots = False
    cfg.output.save_snapshots = False
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    timings = []
    for idx in range(args.repeat):
        run_dir = out_dir / f"bench_{idx + 1}"
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, run_dir / "config_resolved.yaml")
        start = time.perf_counter()
        run_simulation(cfg, run_dir)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        logging.info("Bench %d/%d: %.2fs", idx + 1, args.repeat, elapsed)

    report = pd.DataFrame({"run": list(range(1, args.repeat + 1)), "seconds": timings})
    report.to_csv(out_dir / "bench_times.csv", index=False)


def run_aggregate(args: argparse.Namespace) -> None:
    run_dirs = []
    for pattern in args.runs:
        matched = glob(pattern)
        if matched:
            run_dirs.extend([Path(p) for p in matched])
        else:
            run_dirs.append(Path(pattern))

    frames = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "daily_metrics.csv"
        if not metrics_path.exists():
            logging.warning("Skipping %s (no daily_metrics.csv)", run_dir)
            continue
        frames.append((run_dir.name, pd.read_csv(metrics_path)))

    if not frames:
        logging.error("No runs found to aggregate.")
        return

    metric_cols = [
        "adoption_fraction",
        "mean_belief",
        "variance",
        "polarization",
        "r0_like",
        "cluster_penetration",
        "trust_gov",
        "trust_church",
        "trust_local_news",
        "trust_national_news",
        "trust_friends",
    ]
    agg, summary = aggregate_metrics(frames, metric_cols)
    out_dir = Path(args.out) if args.out else run_dirs[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    summary.to_csv(out_dir / "aggregate_summary.csv", index=False)


def main() -> None:
    setup_logging()
    args = parse_args()
    if args.command == "run":
        run_single(args)
    elif args.command == "sweep":
        run_sweep(args)
    elif args.command == "dashboard":
        run_dashboard(args)
    elif args.command == "bench":
        run_bench(args)
    elif args.command == "aggregate":
        run_aggregate(args)


if __name__ == "__main__":
    main()
