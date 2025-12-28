"""
Prior Distributions for ABC
============================
Defines prior distributions for model parameters.

Priors encode our knowledge about reasonable parameter ranges
before seeing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np


@dataclass
class ParameterPrior:
    """Prior distribution for a single parameter."""

    name: str
    distribution: str  # "uniform", "normal", "beta", "loguniform"
    params: Tuple  # Distribution-specific parameters
    description: str

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Sample n values from prior."""
        if self.distribution == "uniform":
            low, high = self.params
            return rng.uniform(low, high, size=n)
        elif self.distribution == "normal":
            mean, std = self.params
            return rng.normal(mean, std, size=n)
        elif self.distribution == "beta":
            alpha, beta = self.params
            return rng.beta(alpha, beta, size=n)
        elif self.distribution == "loguniform":
            log_low, log_high = np.log10(self.params[0]), np.log10(self.params[1])
            log_samples = rng.uniform(log_low, log_high, size=n)
            return 10 ** log_samples
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def log_prob(self, value: float) -> float:
        """Compute log probability of value under prior."""
        if self.distribution == "uniform":
            low, high = self.params
            if low <= value <= high:
                return -np.log(high - low)
            return -np.inf
        elif self.distribution == "normal":
            mean, std = self.params
            return -0.5 * ((value - mean) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))
        elif self.distribution == "beta":
            alpha, beta = self.params
            if 0 < value < 1:
                from scipy import special
                return ((alpha - 1) * np.log(value) + (beta - 1) * np.log(1 - value)
                        - special.betaln(alpha, beta))
            return -np.inf
        elif self.distribution == "loguniform":
            low, high = self.params
            if low <= value <= high:
                return -np.log(value) - np.log(np.log(high / low))
            return -np.inf
        return 0.0


def define_parameter_priors() -> Dict[str, ParameterPrior]:
    """
    Define prior distributions for all calibratable parameters.

    These represent reasonable ranges based on theory and prior studies.
    """
    priors = {}

    # Belief update parameters
    priors["eta"] = ParameterPrior(
        name="eta",
        distribution="uniform",
        params=(0.01, 0.5),
        description="Learning rate for belief updates",
    )

    priors["rho"] = ParameterPrior(
        name="rho",
        distribution="uniform",
        params=(0.05, 0.4),
        description="Correction effect strength",
    )

    priors["alpha"] = ParameterPrior(
        name="alpha",
        distribution="uniform",
        params=(0.3, 1.5),
        description="Exposure weight in belief update",
    )

    priors["beta"] = ParameterPrior(
        name="beta",
        distribution="uniform",
        params=(0.2, 1.0),
        description="Trust signal weight",
    )

    priors["delta"] = ParameterPrior(
        name="delta",
        distribution="uniform",
        params=(0.1, 0.8),
        description="Social proof weight",
    )

    priors["lambda_skepticism"] = ParameterPrior(
        name="lambda_skepticism",
        distribution="uniform",
        params=(0.3, 1.2),
        description="Skepticism dampening effect",
    )

    priors["belief_decay"] = ParameterPrior(
        name="belief_decay",
        distribution="loguniform",
        params=(0.001, 0.1),
        description="Natural belief decay rate",
    )

    # Dual-process parameters
    priors["base_s1_tendency"] = ParameterPrior(
        name="base_s1_tendency",
        distribution="beta",
        params=(4, 3),  # Mode around 0.6
        description="Base tendency toward System 1 processing",
    )

    priors["cognitive_load_s1_boost"] = ParameterPrior(
        name="cognitive_load_s1_boost",
        distribution="uniform",
        params=(0.1, 0.5),
        description="How much cognitive load increases S1 reliance",
    )

    # Motivated reasoning parameters
    priors["confirmation_strength"] = ParameterPrior(
        name="confirmation_strength",
        distribution="uniform",
        params=(0.1, 0.7),
        description="Strength of confirmation bias",
    )

    priors["identity_threat_sensitivity"] = ParameterPrior(
        name="identity_threat_sensitivity",
        distribution="uniform",
        params=(0.2, 0.9),
        description="Sensitivity to identity-threatening information",
    )

    priors["reactance_strength"] = ParameterPrior(
        name="reactance_strength",
        distribution="uniform",
        params=(0.1, 0.6),
        description="Strength of psychological reactance",
    )

    # Network evolution parameters
    priors["rewiring_rate"] = ParameterPrior(
        name="rewiring_rate",
        distribution="loguniform",
        params=(0.001, 0.1),
        description="Rate of network rewiring per timestep",
    )

    priors["heterophily_dissolution_rate"] = ParameterPrior(
        name="heterophily_dissolution_rate",
        distribution="uniform",
        params=(0.02, 0.3),
        description="Rate of dissolving ties with different-belief agents",
    )

    # Sharing parameters
    priors["base_share_rate"] = ParameterPrior(
        name="base_share_rate",
        distribution="loguniform",
        params=(0.005, 0.1),
        description="Base probability of sharing",
    )

    priors["belief_sensitivity"] = ParameterPrior(
        name="belief_sensitivity",
        distribution="uniform",
        params=(1.0, 4.0),
        description="How much belief affects sharing probability",
    )

    priors["emotion_sensitivity"] = ParameterPrior(
        name="emotion_sensitivity",
        distribution="uniform",
        params=(0.2, 1.0),
        description="How much emotion affects sharing",
    )

    # World parameters
    priors["moderation_strictness"] = ParameterPrior(
        name="moderation_strictness",
        distribution="uniform",
        params=(0.0, 1.0),
        description="Content moderation strictness",
    )

    priors["algorithmic_amplification"] = ParameterPrior(
        name="algorithmic_amplification",
        distribution="uniform",
        params=(0.0, 1.0),
        description="Algorithmic amplification strength",
    )

    priors["outrage_amplification"] = ParameterPrior(
        name="outrage_amplification",
        distribution="uniform",
        params=(0.0, 0.8),
        description="Amplification of high-outrage content",
    )

    return priors


def sample_from_priors(
    priors: Dict[str, ParameterPrior],
    rng: np.random.Generator,
    n_samples: int = 1,
) -> List[Dict[str, float]]:
    """
    Sample complete parameter sets from priors.

    Returns list of parameter dictionaries.
    """
    samples = []

    for _ in range(n_samples):
        params = {}
        for name, prior in priors.items():
            params[name] = float(prior.sample(rng, n=1)[0])
        samples.append(params)

    return samples


def compute_prior_probability(
    params: Dict[str, float],
    priors: Dict[str, ParameterPrior],
) -> float:
    """
    Compute joint prior probability of parameter set.

    Returns log probability (sum of individual log probs).
    """
    log_prob = 0.0

    for name, value in params.items():
        if name in priors:
            log_prob += priors[name].log_prob(value)

    return log_prob


def perturb_params(
    params: Dict[str, float],
    priors: Dict[str, ParameterPrior],
    rng: np.random.Generator,
    scale: float = 0.1,
) -> Dict[str, float]:
    """
    Perturb parameter values for MCMC or SMC.

    Uses a Gaussian kernel with width proportional to prior range.
    """
    perturbed = {}

    for name, value in params.items():
        if name in priors:
            prior = priors[name]

            if prior.distribution == "uniform":
                low, high = prior.params
                width = scale * (high - low)
                new_value = value + rng.normal(0, width)
                new_value = np.clip(new_value, low, high)
            elif prior.distribution == "loguniform":
                log_value = np.log10(value)
                log_low, log_high = np.log10(prior.params[0]), np.log10(prior.params[1])
                width = scale * (log_high - log_low)
                new_log_value = log_value + rng.normal(0, width)
                new_log_value = np.clip(new_log_value, log_low, log_high)
                new_value = 10 ** new_log_value
            elif prior.distribution == "beta":
                # Use logit transform for beta
                logit = np.log(value / (1 - value + 1e-10))
                new_logit = logit + rng.normal(0, scale)
                new_value = 1 / (1 + np.exp(-new_logit))
                new_value = np.clip(new_value, 0.01, 0.99)
            else:
                new_value = value + rng.normal(0, scale * value)

            perturbed[name] = float(new_value)
        else:
            perturbed[name] = value

    return perturbed
