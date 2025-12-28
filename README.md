# Town Misinformation Contagion Simulator

A cognitively-grounded agent-based model for studying how misinformation spreads through social networks. The simulator builds synthetic towns with realistic social structures, implements psychological theories of belief formation, and enables testing of intervention strategies.

## Features

### Cognitive Architecture
- **Dual-Process Theory** (Kahneman, 2011): System 1 (fast/intuitive) and System 2 (slow/analytical) processing
- **Motivated Reasoning**: Identity-protective cognition with confirmation bias and psychological reactance
- **Attention Economics**: Bounded rationality with finite cognitive resources
- **Source Memory**: Credibility tracking and sleeper effects

### Social Network Modeling
- **Multi-layer Networks**: Family, workplace, school, church, and neighborhood ties
- **Homophily**: Belief-similarity-based connection patterns
- **Dynamic Evolution**: Networks rewire based on belief divergence
- **Echo Chamber Emergence**: Filter bubbles form as emergent property

### Misinformation Dynamics
- **Continuous Belief States**: Beliefs range from 0.0 to 1.0 per claim
- **Cascade Tracking**: Full genealogy of information spread
- **Structural Virality**: Distinguishes viral from broadcast patterns
- **R-effective Computation**: True epidemiological reproduction numbers

### Intervention Testing
- **Content Moderation**: Configurable removal/suppression effects
- **Prebunking/Inoculation**: Pre-exposure resistance building
- **Narrative Competition**: Competing claims and counter-messaging

### Technical Features
- **GPU Acceleration**: PyTorch-based vectorized computations (CUDA optional)
- **Deterministic Execution**: Full reproducibility with seed control
- **Comprehensive Metrics**: Adoption, polarization, entropy, cascade analysis
- **Interactive Dashboard**: Streamlit-based visualization

## Installation

### Using uv (Recommended)
```bash
cd town_misinfo_sim
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Using pip
```bash
cd town_misinfo_sim
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Optional Dependencies
```bash
# For advanced community detection
pip install python-igraph

# For development
pip install -e ".[dev]"
```

## Quick Start

### Run a Baseline Simulation
```bash
python -m sim run \
  --config configs/world_baseline.yaml \
  --seed 42 \
  --steps 30 \
  --n 1000 \
  --out runs/baseline/
```

### Run an Alternate World
```bash
python -m sim run \
  --config configs/world_strong_moderation.yaml \
  --seed 42 \
  --steps 30 \
  --n 1000 \
  --out runs/moderation/
```

### Run a Parameter Sweep
```bash
python -m sim sweep \
  --configs configs/world_*.yaml \
  --seeds 1 2 3 4 5 \
  --out runs/sweep/
```

### Aggregate Results
```bash
python -m sim aggregate --runs runs/sweep/* --out runs/sweep/
```

### Launch Dashboard
```bash
python -m sim dashboard --run runs/baseline/
```

### Benchmark Performance
```bash
python -m sim bench \
  --config configs/world_baseline.yaml \
  --n 10000 \
  --steps 200 \
  --device cuda \
  --repeat 3 \
  --out runs/bench/
```

## World Configurations

Pre-configured scenarios in `configs/`:

| Configuration | Description |
|---------------|-------------|
| `world_baseline.yaml` | Standard balanced parameters |
| `world_high_trust_gov.yaml` | High institutional trust scenario |
| `world_low_trust_gov.yaml` | Low institutional trust scenario |
| `world_strong_moderation.yaml` | Aggressive content moderation |
| `world_collapsed_local_media.yaml` | Local media ecosystem collapse |
| `world_high_religion_hub.yaml` | High religious institution centrality |
| `world_outrage_algorithm.yaml` | Algorithmic amplification of outrage |

## Configuration Structure

```yaml
sim:
  steps: 30              # Simulation duration (days)
  n_agents: 1000         # Population size
  device: cuda           # cuda, mps, or cpu
  seed: 42               # Random seed

town:
  # Demographics and institution parameters

network:
  # Homophily and geography controls per layer

world:
  # Trust baselines, media reach, moderation

belief_update:
  # Coefficients for belief dynamics

sharing:
  # Share probability parameters

strains:
  # Synthetic claims with emotional profiles

metrics:
  # Community detection and metric settings
```

## Output Files

Each simulation run produces:

| File | Description |
|------|-------------|
| `daily_metrics.csv` | Daily adoption, belief, polarization metrics |
| `summary.json` | Peak statistics and intervention effects |
| `cascade_stats.json` | Information cascade analysis |
| `config_resolved.yaml` | Final resolved configuration |
| `run_metadata.json` | Versions, device, seed, git commit |
| `snapshots.parquet` | Belief state snapshots at intervals |

## Project Structure

```
town_misinfo_sim/
├── configs/                 # World configuration files
├── docs/                    # Additional documentation
│   └── METHODS.md          # Formal methods with equations
├── sim/                     # Main source code
│   ├── calibration/        # ABC calibration framework
│   ├── cascades/           # Cascade tracking and analysis
│   ├── cognition/          # Cognitive architecture modules
│   ├── dashboard/          # Streamlit visualization
│   ├── disease/            # Belief update and exposure
│   ├── dynamics/           # Network evolution
│   ├── io/                 # Input/output utilities
│   ├── metrics/            # Metric computation
│   ├── narratives/         # Competing narratives
│   ├── town/               # Town and network generation
│   └── world/              # World effects (moderation, media)
├── scripts/                 # Validation and analysis scripts
├── tests/                   # Unit tests
├── ARCHITECTURE.md         # Module structure documentation
├── BREAKTHROUGH_INNOVATIONS.md  # Technical innovations
└── pyproject.toml          # Project configuration
```

## Validation Results

The model has been validated with 1,200+ simulations:

| Metric | Value | Target |
|--------|-------|--------|
| Mean adoption (50 seeds) | 26.5% ± 3.0% | 20-40% |
| Seeds in target range | 98% | >90% |
| Strong moderation effect | -57% | Significant |
| Prebunking effect (30%) | -68% | Significant |

See `validation_results/NATURE_SUMMARY.md` for comprehensive validation.

## Quality Checks

```bash
# Run tests
python -m pytest

# Lint code
ruff check sim tests

# Type checking
mypy sim
```

## Performance Notes

- Belief updates run on GPU when available (`--device cuda`)
- For large simulations, keep `n_claims` small
- Disable cluster penetration for faster sweeps:
  ```yaml
  metrics:
    cluster_penetration_enabled: false
  ```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Base learning rate | 0.15 | Controls adoption speed |
| Social proof weight | 0.22 | Influence of peer beliefs |
| Skepticism dampening | 0.40 | Individual resistance |
| Decay rate | 0.008 | Belief fade over time |
| Adoption threshold | 0.70 | Belief level for "adoption" |

## Theoretical Foundations

The model integrates established theories:

- **Dual-Process Theory** (Kahneman, 2011)
- **Motivated Reasoning** (Kunda, 1990)
- **Inoculation Theory** (McGuire, 1961; van der Linden, 2017)
- **Social Proof** (Cialdini, 1984)
- **Attention Economics** (Simon, 1971)
- **Structural Virality** (Goel et al., 2016)

## Safety and Ethics

This simulator is designed for **research and defensive scenario analysis**. It does not include a misinformation optimizer. An optional research-only operator stub exists but is disabled by default.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{misinfo_sim,
  author = {Thiru, Mukund},
  title = {Town Misinformation Contagion Simulator},
  url = {https://github.com/contactmukundthiru-cyber/Misinformation-Agent-Simulation},
  year = {2024}
}
```

## References

1. Kahneman, D. (2011). *Thinking, fast and slow*. Farrar, Straus and Giroux.
2. Kunda, Z. (1990). The case for motivated reasoning. *Psychological Bulletin*.
3. van der Linden, S. (2017). Inoculating against misinformation. *Science*.
4. Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. *Science*.
5. Goel, S., et al. (2016). The structural virality of online diffusion. *Management Science*.
