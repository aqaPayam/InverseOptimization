from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .core import ForwardProblem, InverseDataset, as_array
from .geometry import ConsistencyConstraints


def decisions_equal(first: Any, second: Any, tolerance: float = 1e-8) -> bool:
    try:
        return bool(np.allclose(as_array(first), as_array(second), atol=tolerance, rtol=0))
    except Exception:
        return first == second


def decision_distance(first: Any, second: Any) -> float:
    try:
        return float(np.linalg.norm(as_array(first) - as_array(second)))
    except Exception:
        return 0.0 if first == second else 1.0


def angular_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate_norm = np.linalg.norm(estimate)
    truth_norm = np.linalg.norm(truth)
    if estimate_norm <= 1e-15 or truth_norm <= 1e-15:
        return float("nan")
    cosine = float(np.clip(np.dot(estimate, truth) / (estimate_norm * truth_norm), -1.0, 1.0))
    return float(np.arccos(cosine))


def parameter_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate = as_array(estimate).reshape(-1)
    truth = as_array(truth).reshape(-1)
    angle = angular_error(estimate, truth)
    normalized_estimate = estimate / max(np.linalg.norm(estimate), 1e-15)
    normalized_truth = truth / max(np.linalg.norm(truth), 1e-15)
    return {
        "parameter_angular_error": angle,
        "parameter_cosine_similarity": float(np.cos(angle)) if np.isfinite(angle) else float("nan"),
        "parameter_l1_error_normalized": float(np.linalg.norm(normalized_estimate - normalized_truth, 1)),
        "parameter_l2_error_normalized": float(np.linalg.norm(normalized_estimate - normalized_truth)),
        "parameter_linf_error_normalized": float(np.linalg.norm(normalized_estimate - normalized_truth, np.inf)),
        "parameter_sign_accuracy": float(np.mean(np.sign(estimate) == np.sign(truth))),
    }


def evaluate_predictions(
    problem: ForwardProblem,
    dataset: InverseDataset,
    theta: np.ndarray,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, float]]:
    predictions: list[Any] = []
    rows: list[dict[str, Any]] = []
    observed_matches: list[float] = []
    observed_distances: list[float] = []
    clean_matches: list[float] = []
    clean_distances: list[float] = []
    true_regrets: list[float] = []
    surrogate_gaps: list[float] = []
    for index, observation in enumerate(dataset):
        solution = problem.solve(theta, observation.context)
        prediction = solution.decision
        predictions.append(prediction)
        observed_match = float(decisions_equal(prediction, observation.decision))
        observed_distance = decision_distance(prediction, observation.decision)
        observed_matches.append(observed_match)
        observed_distances.append(observed_distance)
        observed_value = problem.objective.value(theta, observation.context, observation.decision)
        surrogate_gap = float(observed_value - solution.value)
        surrogate_gaps.append(surrogate_gap)
        row: dict[str, Any] = {
            "index": index,
            "observed_match": observed_match,
            "observed_distance": observed_distance,
            "surrogate_suboptimality": surrogate_gap,
            "predicted_value": solution.value,
        }
        if observation.clean_decision is not None:
            clean_match = float(decisions_equal(prediction, observation.clean_decision))
            clean_distance = decision_distance(prediction, observation.clean_decision)
            clean_matches.append(clean_match)
            clean_distances.append(clean_distance)
            row.update(clean_match=clean_match, clean_distance=clean_distance)
        if observation.true_theta is not None:
            optimum = problem.solve(observation.true_theta, observation.context)
            predicted_true_value = problem.objective.value(
                observation.true_theta, observation.context, prediction
            )
            regret = float(predicted_true_value - optimum.value)
            true_regrets.append(regret)
            row["true_regret"] = regret
        rows.append(row)
    summary = {
        "observed_decision_accuracy": float(np.mean(observed_matches)),
        "mean_observed_decision_distance": float(np.mean(observed_distances)),
        "mean_surrogate_suboptimality": float(np.mean(surrogate_gaps)),
        "max_surrogate_suboptimality": float(np.max(surrogate_gaps)),
    }
    if clean_matches:
        summary.update(
            clean_decision_accuracy=float(np.mean(clean_matches)),
            mean_clean_decision_distance=float(np.mean(clean_distances)),
        )
    if true_regrets:
        regrets = np.asarray(true_regrets)
        summary.update(
            mean_true_regret=float(np.mean(regrets)),
            cumulative_true_regret=float(np.sum(regrets)),
            max_true_regret=float(np.max(regrets)),
            true_regret_cvar_20=float(np.mean(np.sort(regrets)[-max(1, int(np.ceil(0.2 * regrets.size))) :])),
        )
    truths = [obs.true_theta for obs in dataset if obs.true_theta is not None]
    if truths and all(np.allclose(truths[0], truth) for truth in truths[1:]):
        summary.update(parameter_metrics(theta, truths[0]))
    return predictions, rows, summary


def consistency_metrics(constraints: ConsistencyConstraints, theta: np.ndarray) -> dict[str, float]:
    if not constraints.records:
        return {
            "constraint_count": 0.0,
            "constraint_violation_rate": 0.0,
            "mean_constraint_violation": 0.0,
            "max_constraint_violation": 0.0,
        }
    violations = constraints.violations(theta)
    slacks = constraints.slacks(theta)
    return {
        "constraint_count": float(len(constraints.records)),
        "constraint_violation_rate": float(np.mean(violations > 1e-9)),
        "mean_constraint_violation": float(np.mean(violations)),
        "max_constraint_violation": float(np.max(violations)),
        "minimum_margin": float(np.min(slacks)),
        "median_margin": float(np.median(slacks)),
        "mean_margin": float(np.mean(slacks)),
    }


def cumulative(values: list[float] | np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(values, dtype=float))

