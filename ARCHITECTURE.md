# Architecture

## Data flow
1. **Config** (`sim/config.py`): load YAML, validate, and resolve defaults.
2. **Town generation** (`sim/town/`): assign demographics, traits, trust priors, and create multilayer edges.
3. **Simulation loop** (`sim/simulation.py`):
   - Compute sharing probabilities and moderation effects.
   - Aggregate social + institutional + feed exposures on GPU.
   - Update beliefs with repetition and decay.
   - Update trust (optional) and record metrics.
4. **Outputs** (`sim/io/`): metrics CSV, parquet snapshots, plots, summary JSON.
   - `run_metadata.json` captures versions, device, and seed.
5. **Dashboard** (`sim/dashboard/app.py`): visualizes metrics and snapshots.

## Module map
- `sim/config.py`: pydantic config models + YAML merging.
- `sim/rng.py`: deterministic RNG manager for numpy/torch.
- `sim/town/`: demographics + network construction.
- `sim/disease/`: strains, exposure, sharing, belief update (torch).
- `sim/world/`: moderation, institutional trust updates, feed injection.
- `sim/metrics/`: adoption, polarization, cluster penetration, R0-like proxy.
- `sim/io/`: plotting, snapshot capture, logging.
- `sim/analysis/`: sweep aggregation and summary utilities.

## Extensibility
- Add new claims in `configs/*.yaml` under `strains`.
- Create new worlds via YAML overrides (use `base: world_baseline.yaml`).
- Implement alternative policies in `sim/disease/operator.py` (research-only).
