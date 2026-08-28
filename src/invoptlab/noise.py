from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .capabilities import Capability
from .core import as_array


@dataclass(slots=True)
class NoNoise:
    def apply(self, decision: Any, **_: Any) -> tuple[Any, dict[str, Any]]:
        return decision, {"type": "none"}


@dataclass(slots=True)
class AdditiveNoise:
    scale: float = 0.1
    distribution: str = "gaussian"
    project: Callable[[Any, Any], Any] | None = None

    def apply(self, decision: Any, *, context: Any, rng: np.random.Generator, **_: Any):
        clean = as_array(decision)
        if self.distribution == "gaussian":
            error = rng.normal(scale=self.scale, size=clean.shape)
        elif self.distribution == "laplace":
            error = rng.laplace(scale=self.scale, size=clean.shape)
        elif self.distribution == "uniform":
            error = rng.uniform(-self.scale, self.scale, size=clean.shape)
        else:
            raise ValueError("distribution must be gaussian, laplace, or uniform")
        noisy = clean + error
        if self.project is not None:
            noisy = self.project(context, noisy)
        return noisy, {"type": "additive", "distribution": self.distribution, "scale": self.scale}


@dataclass(slots=True)
class RandomFeasibleNoise:
    probability: float = 0.1

    def apply(self, decision: Any, *, context: Any, problem: Any, rng: np.random.Generator, **_: Any):
        if rng.random() >= self.probability:
            return decision, {"type": "random_feasible", "changed": False, "probability": self.probability}
        if Capability.SUPPORTS_ENUMERATION not in problem.capabilities:
            raise ValueError("RandomFeasibleNoise requires an enumeration-capable problem")
        alternatives = list(problem.oracle.enumerate(context))
        choices = [x for x in alternatives if not np.array_equal(np.asarray(x), np.asarray(decision))]
        noisy = decision if not choices else choices[int(rng.integers(len(choices)))]
        return noisy, {"type": "random_feasible", "changed": bool(choices), "probability": self.probability}


@dataclass(slots=True)
class EpsilonOptimalNoise:
    epsilon: float = 0.1

    def apply(self, decision: Any, *, context: Any, problem: Any, theta: np.ndarray, rng: np.random.Generator, **_: Any):
        alternatives = list(problem.oracle.enumerate(context))
        values = np.asarray([problem.objective.value(theta, context, x) for x in alternatives])
        candidates = np.flatnonzero(values <= values.min() + self.epsilon)
        index = int(rng.choice(candidates))
        return alternatives[index], {
            "type": "epsilon_optimal",
            "epsilon": self.epsilon,
            "suboptimality": float(values[index] - values.min()),
        }


@dataclass(slots=True)
class BoltzmannNoise:
    temperature: float = 0.1

    def apply(self, decision: Any, *, context: Any, problem: Any, theta: np.ndarray, rng: np.random.Generator, **_: Any):
        if self.temperature <= 0:
            return decision, {"type": "boltzmann", "temperature": self.temperature, "changed": False}
        alternatives = list(problem.oracle.enumerate(context))
        values = np.asarray([problem.objective.value(theta, context, x) for x in alternatives])
        logits = -(values - values.min()) / self.temperature
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(rng.choice(len(alternatives), p=probabilities))
        return alternatives[index], {
            "type": "boltzmann",
            "temperature": self.temperature,
            "probability": float(probabilities[index]),
            "changed": not np.array_equal(np.asarray(alternatives[index]), np.asarray(decision)),
        }


@dataclass(slots=True)
class BinaryFlipNoise:
    probability: float = 0.05

    def apply(self, decision: Any, *, rng: np.random.Generator, **_: Any):
        clean = as_array(decision).astype(int)
        mask = rng.random(clean.shape) < self.probability
        noisy = np.where(mask, 1 - clean, clean)
        return noisy, {"type": "binary_flip", "probability": self.probability, "flips": int(mask.sum())}


@dataclass(slots=True)
class ContaminationNoise:
    base: Any
    probability: float = 0.05
    contaminator: Any = None

    def apply(self, decision: Any, *, rng: np.random.Generator, **kwargs: Any):
        value, metadata = self.base.apply(decision, rng=rng, **kwargs)
        if rng.random() >= self.probability:
            return value, {**metadata, "contaminated": False}
        contaminator = self.contaminator or RandomFeasibleNoise(1.0)
        value, contamination = contaminator.apply(decision, rng=rng, **kwargs)
        return value, {**metadata, "contaminated": True, "contamination": contamination}

