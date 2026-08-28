from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from .core import CallableObjective, ForwardProblem, LinearObjective, Observation, as_array
from .exceptions import CapabilityError


def euclidean_distance(first: Any, second: Any) -> float:
    return float(np.linalg.norm(as_array(first) - as_array(second)))


def squared_euclidean_distance(first: Any, second: Any) -> float:
    difference = as_array(first) - as_array(second)
    return float(np.dot(difference.reshape(-1), difference.reshape(-1)))


def hamming_distance(first: Any, second: Any) -> float:
    return float(np.sum(np.asarray(first) != np.asarray(second)))


def zero_distance(_: Any, __: Any) -> float:
    return 0.0


class InverseLoss(Protocol):
    name: str

    def value_and_subgradient(
        self, problem: ForwardProblem, theta: np.ndarray, observation: Observation
    ) -> tuple[float, np.ndarray, dict[str, Any]]: ...


def _parameter_gradient(problem: ForwardProblem, theta: np.ndarray, context: Any, decision: Any) -> np.ndarray:
    objective = problem.objective
    if isinstance(objective, LinearObjective):
        return objective.features(context, decision)
    if isinstance(objective, CallableObjective) and objective.gradient is not None:
        return as_array(objective.gradient(theta, context, decision)).reshape(-1)
    # Stable central finite difference for black-box objectives.
    epsilon = 1e-6
    gradient = np.zeros_like(theta, dtype=float)
    for index in range(theta.size):
        direction = np.zeros_like(theta)
        direction[index] = epsilon
        gradient[index] = (
            objective.value(theta + direction, context, decision)
            - objective.value(theta - direction, context, decision)
        ) / (2 * epsilon)
    return gradient


@dataclass(slots=True)
class SuboptimalityLoss:
    name: str = "suboptimality"

    def value_and_subgradient(self, problem: ForwardProblem, theta: np.ndarray, observation: Observation):
        prediction = problem.solve(theta, observation.context)
        value = problem.objective.value(theta, observation.context, observation.decision) - prediction.value
        gradient = _parameter_gradient(problem, theta, observation.context, observation.decision)
        gradient -= _parameter_gradient(problem, theta, observation.context, prediction.decision)
        return max(0.0, float(value)), gradient, {"competitor": prediction.decision}


@dataclass(slots=True)
class AugmentedSuboptimalityLoss:
    distance: Callable[[Any, Any], float] = hamming_distance
    margin_scale: float = 1.0
    name: str = "augmented_suboptimality"

    def value_and_subgradient(self, problem: ForwardProblem, theta: np.ndarray, observation: Observation):
        if not hasattr(problem.oracle, "loss_augmented_solve"):
            raise CapabilityError("ASL requires an oracle with loss_augmented_solve")

        def scaled_distance(first: Any, second: Any) -> float:
            return self.margin_scale * self.distance(first, second)

        competitor = problem.oracle.loss_augmented_solve(
            problem.objective,
            theta,
            observation.context,
            observation.decision,
            scaled_distance,
        )
        value = (
            problem.objective.value(theta, observation.context, observation.decision)
            - competitor.value
        )
        gradient = _parameter_gradient(problem, theta, observation.context, observation.decision)
        gradient -= _parameter_gradient(problem, theta, observation.context, competitor.decision)
        return max(0.0, float(value)), gradient, {
            "competitor": competitor.decision,
            "margin": scaled_distance(observation.decision, competitor.decision),
        }


@dataclass(slots=True)
class DecisionDistanceLoss:
    distance: Callable[[Any, Any], float] = euclidean_distance
    name: str = "decision_distance"

    def value_and_subgradient(self, problem: ForwardProblem, theta: np.ndarray, observation: Observation):
        prediction = problem.solve(theta, observation.context)
        value = self.distance(observation.decision, prediction.decision)
        # Decision-distance loss is generally discontinuous. A zero vector signals
        # that gradient-based training should not use it without a custom surrogate.
        return float(value), np.zeros_like(theta), {"competitor": prediction.decision, "nondifferentiable": True}


@dataclass(slots=True)
class KKTResidualLoss:
    residual: Callable[[ForwardProblem, np.ndarray, Observation], tuple[float, np.ndarray]]
    name: str = "kkt_residual"

    def value_and_subgradient(self, problem: ForwardProblem, theta: np.ndarray, observation: Observation):
        value, gradient = self.residual(problem, theta, observation)
        return float(value), as_array(gradient).reshape(-1), {}


def evaluate_losses(
    loss: InverseLoss,
    problem: ForwardProblem,
    theta: np.ndarray,
    observations: list[Observation],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    values: list[float] = []
    gradients: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for observation in observations:
        value, gradient, diagnostic = loss.value_and_subgradient(problem, theta, observation)
        values.append(value)
        # Observation weights are applied once by the selected risk aggregator.
        # Keeping per-observation gradients unweighted avoids squaring weights.
        gradients.append(gradient)
        diagnostics.append(diagnostic)
    return np.asarray(values), np.vstack(gradients), diagnostics
