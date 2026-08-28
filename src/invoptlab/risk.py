from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MeanRisk:
    name: str = "mean"

    def aggregate(self, values: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        values = np.asarray(values, dtype=float)
        if weights is None:
            weights = np.ones(values.size)
        weights = np.asarray(weights, dtype=float)
        normalized = weights / max(weights.sum(), 1e-15)
        return float(np.dot(normalized, values)), normalized


@dataclass(slots=True)
class CVaRRisk:
    tail_fraction: float = 0.2
    name: str = "cvar"

    def aggregate(self, values: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must lie in (0, 1]")
        values = np.asarray(values, dtype=float)
        count = max(1, int(np.ceil(self.tail_fraction * values.size)))
        tail = np.argsort(values)[-count:]
        coefficients = np.zeros(values.size)
        coefficients[tail] = 1.0 / count
        return float(np.dot(coefficients, values)), coefficients


@dataclass(slots=True)
class TrimmedMeanRisk:
    trim_fraction: float = 0.1
    name: str = "trimmed_mean"

    def aggregate(self, values: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        if not 0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must lie in [0, 0.5)")
        values = np.asarray(values, dtype=float)
        trim = int(np.floor(self.trim_fraction * values.size))
        order = np.argsort(values)
        retained = order[trim : values.size - trim if trim else values.size]
        coefficients = np.zeros(values.size)
        coefficients[retained] = 1.0 / max(retained.size, 1)
        return float(np.dot(coefficients, values)), coefficients


@dataclass(slots=True)
class QuantileRisk:
    quantile: float = 0.9
    name: str = "quantile"

    def aggregate(self, values: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        values = np.asarray(values, dtype=float)
        index = int(np.clip(np.ceil(self.quantile * values.size) - 1, 0, values.size - 1))
        selected = np.argsort(values)[index]
        coefficients = np.zeros(values.size)
        coefficients[selected] = 1.0
        return float(values[selected]), coefficients

