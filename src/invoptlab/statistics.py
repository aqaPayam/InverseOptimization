from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .core import ForwardProblem, InverseDataset


def percentile_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    alpha = (1 - confidence) / 2
    return np.quantile(values, alpha, axis=0), np.quantile(values, 1 - alpha, axis=0)


@dataclass(slots=True)
class BootstrapResult:
    parameters: np.ndarray
    mean: np.ndarray
    standard_error: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def bootstrap_parameters(
    estimator_factory: Callable[[], Any],
    problem: ForwardProblem,
    dataset: InverseDataset,
    *,
    repetitions: int = 100,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    parameters = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(dataset), size=len(dataset))
        sample = InverseDataset([dataset.observations[int(index)] for index in indices], dataset.name)
        estimator = estimator_factory().fit(problem, sample)
        parameters.append(estimator.theta_.copy())
    values = np.vstack(parameters)
    lower, upper = percentile_interval(values, confidence)
    return BootstrapResult(
        parameters=values,
        mean=np.mean(values, axis=0),
        standard_error=np.std(values, axis=0, ddof=1),
        lower=lower,
        upper=upper,
    )


def leave_one_out_influence(
    estimator_factory: Callable[[], Any],
    problem: ForwardProblem,
    dataset: InverseDataset,
) -> np.ndarray:
    baseline = estimator_factory().fit(problem, dataset).theta_
    influences = []
    for excluded in range(len(dataset)):
        subset = InverseDataset(
            [obs for index, obs in enumerate(dataset) if index != excluded],
            f"{dataset.name}-without-{excluded}",
        )
        estimate = estimator_factory().fit(problem, subset).theta_
        influences.append(np.linalg.norm(estimate - baseline))
    return np.asarray(influences)


def summarize_repeated_metrics(records: list[dict[str, float]], confidence: float = 0.95):
    if not records:
        return {}
    common = set.intersection(*(set(record) for record in records))
    summary: dict[str, dict[str, float]] = {}
    for key in sorted(common):
        values = np.asarray([record[key] for record in records], dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        # Normal approximation is adequate for the lightweight run summary;
        # bootstrap intervals remain available for parameter estimates.
        z = 1.96 if abs(confidence - 0.95) < 1e-9 else 1.96
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "standard_error": standard_error,
            "lower": float(np.mean(values) - z * standard_error),
            "upper": float(np.mean(values) + z * standard_error),
            "count": float(values.size),
        }
    return summary

