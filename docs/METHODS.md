# Formal Methods Documentation

## Model Overview

This agent-based model simulates misinformation spread in a social network using a cognitively-grounded architecture. The model integrates established psychological theories into a unified computational framework.

---

## 1. Agent Architecture

### 1.1 Agent State Variables

Each agent i maintains the following state:

| Variable | Symbol | Range | Description |
|----------|--------|-------|-------------|
| Belief | b_ik | [0, 1] | Belief strength for claim k |
| Skepticism | s_i | [0, 1] | Dispositional skepticism trait |
| Conformity | c_i | [0, 1] | Social conformity tendency |
| Numeracy | n_i | [0, 1] | Analytical reasoning ability |
| Trust (gov) | τ_gov,i | [0, 1] | Trust in government institutions |
| Trust (friends) | τ_friends,i | [0, 1] | Trust in social contacts |

### 1.2 Trait Initialization

Traits are drawn from beta distributions calibrated to empirical surveys:

```
s_i ~ Beta(α_s, β_s)  where E[s] = 0.5, Var[s] = 0.04
c_i ~ Beta(α_c, β_c)  where E[c] = 0.55, Var[c] = 0.03
```

---

## 2. Network Structure

### 2.1 Topology

The social network is a multi-layer small-world network with:
- **Family layer**: High weight (w = 1.6), household-based
- **Work layer**: Medium weight (w = 1.1), workplace clusters
- **Neighborhood layer**: Lower weight (w = 0.8), geographic proximity

### 2.2 Network Statistics

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Clustering coefficient | ~0.3 | Matches empirical social networks |
| Average path length | ~4.5 | "Six degrees of separation" |
| Homophily strength | 0.6 | Calibrated to survey data |

---

## 3. Belief Update Mechanism

### 3.1 Dual-Process Architecture

Following Kahneman (2011), belief updates integrate System 1 (fast/intuitive) and System 2 (slow/analytical) processing:

```
Acceptance = (1 - α) × S1(exposure) + α × S2(exposure, evidence)
```

Where:
- α = f(cognitive_load, threat_level) ∈ [0, 1]
- S1 uses heuristic cues (source familiarity, emotional salience)
- S2 evaluates evidence quality and logical consistency

### 3.2 System 1 Processing

```python
S1_output = σ(w_familiarity × familiarity +
              w_emotion × emotional_resonance +
              w_fluency × processing_fluency)
```

### 3.3 System 2 Processing

```python
S2_output = σ(w_evidence × evidence_quality +
              w_source × source_credibility +
              w_consistency × logical_consistency)
```

---

## 4. Social Influence

### 4.1 Social Proof

Social proof is computed as the fraction of neighbors with beliefs above threshold:

```
social_proof_ik = Σ_j∈N(i) w_ij × I(b_jk > θ) / Σ_j∈N(i) w_ij
```

Where:
- N(i) = neighborhood of agent i
- w_ij = edge weight
- θ = 0.6 (threshold for "believing")

### 4.2 Cascade Dynamics

Adoption follows threshold dynamics:
```
Adopt if: Acceptance + β × social_proof > θ_adopt
```

Where β = 0.22 (calibrated social proof weight)

---

## 5. Motivated Reasoning

### 5.1 Identity Protection

When claims threaten core identity:
```
threat_level = |claim_position - identity_position| × identity_strength
defensive_processing = σ(threat_level × threat_sensitivity)
```

### 5.2 Confirmation Bias

```
biased_acceptance = acceptance × (1 + γ × sign(b_current - 0.5))
```

Where γ = confirmation strength parameter

---

## 6. Interventions

### 6.1 Prebunking (Inoculation)

Based on McGuire (1961) and van der Linden (2017):

```
resistance_i = inoculation_level × (1 + active_multiplier × I(active))
effective_exposure = exposure × (1 - resistance_i)
```

### 6.2 Moderation

```
share_probability = base_rate × (1 - moderation_effect × violation_risk)
```

---

## 7. Key Parameters

| Parameter | Symbol | Value | Calibration |
|-----------|--------|-------|-------------|
| Base learning rate | η | 0.15 | Grid search (target: 25-30% adoption) |
| Social proof weight | β | 0.22 | Grid search |
| Skepticism dampening | δ_s | 0.40 | Literature-informed |
| Decay rate | λ | 0.008 | Stability analysis |
| Adoption threshold | θ | 0.70 | Standard threshold |

---

## 8. Validation Targets

### 8.1 Empirical Benchmarks

| Target | Value | Source |
|--------|-------|--------|
| Misinformation exposure | 25-45% | Guess et al. (2019) |
| Sharing rate | 2-8% | Vosoughi et al. (2018) |
| Cascade size distribution | Power law | Goel et al. (2016) |

### 8.2 Achieved Metrics

| Metric | Achieved | Target |
|--------|----------|--------|
| Final adoption | 26.5% ± 3.0% | 20-40% |
| R_effective | 13.14 ± 1.92 | > 1.0 |
| Moderation effect | -57% | Significant |

---

## 9. Sensitivity Analysis

### 9.1 Phase Transitions

The model exhibits phase transition behavior characteristic of epidemiological models:

- **Subcritical** (η < 0.14): Minimal spread, R_eff < 1
- **Critical** (η ≈ 0.15): Realistic adoption (25-30%)
- **Supercritical** (η > 0.18): Near-total adoption

### 9.2 Stability

Model produces stable equilibria:
- Adoption plateaus by day ~30
- Long-term (365 day) change < 10% from day 90

---

## 10. Computational Details

### 10.1 Implementation

- **Framework**: PyTorch (vectorized operations)
- **Precision**: Float32
- **Determinism**: Optional (torch.use_deterministic_algorithms)

### 10.2 Performance

| Configuration | Time | Memory |
|--------------|------|--------|
| 1000 agents, 30 days | ~15s | ~500MB |
| 5000 agents, 30 days | ~45s | ~2GB |
| 1000 agents, 365 days | ~180s | ~500MB |

---

## References

1. Kahneman, D. (2011). Thinking, fast and slow. Farrar, Straus and Giroux.
2. Kunda, Z. (1990). The case for motivated reasoning. Psychological Bulletin.
3. McGuire, W. J. (1961). The effectiveness of supportive and refutational defenses. Sociometry.
4. van der Linden, S. (2017). Inoculating against misinformation. Science.
5. Centola, D. (2010). The spread of behavior in an online social network. Science.
6. Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. Science.
7. Brady, W. J., et al. (2020). The MAD model of moral contagion. Psychological Inquiry.
