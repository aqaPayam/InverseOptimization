from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .core import EstimatorHistory, ForwardProblem, InverseDataset, StepRecord
from .exceptions import SolverError, ValidationError
from .geometry import ConsistencyConstraints, build_consistency_constraints
from .losses import AugmentedSuboptimalityLoss, InverseLoss, evaluate_losses
from .risk import MeanRisk


class Estimator(Protocol):
    theta_: np.ndarray
    history_: EstimatorHistory

    def fit(self, problem: ForwardProblem, dataset: InverseDataset) -> "Estimator": ...


@dataclass
class IncenterEstimator:
    tolerance: float = 1e-8
    max_iterations: int = 2_000
    sequential_history: bool = False
    theta_: np.ndarray = field(init=False)
    radius_: float = field(init=False)
    history_: EstimatorHistory = field(init=False, default_factory=EstimatorHistory)
    constraints_: ConsistencyConstraints = field(init=False)
    result_: Any = field(init=False, default=None)

    def _solve(self, problem: ForwardProblem, constraints: ConsistencyConstraints) -> tuple[np.ndarray, float, Any]:
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise SolverError("IncenterEstimator requires scipy") from exc

        dimension = problem.parameter_space.dimension
        matrix = constraints.normalized_matrix
        space = problem.parameter_space
        initial_theta = space.center()
        if space.kind == "l2_ball" and matrix.size:
            direction = -np.mean(matrix, axis=0)
            norm = np.linalg.norm(direction)
            initial_theta = direction / norm * (0.8 * space.radius) if norm > 1e-12 else space.sample(1, seed=0)[0]
        initial_radius = float(np.min(-(matrix @ initial_theta))) if matrix.size else 0.0
        initial = np.concatenate([initial_theta, [initial_radius]])

        constraints_spec: list[dict[str, Any]] = []
        if matrix.size:
            constraints_spec.append({"type": "ineq", "fun": lambda z: -(matrix @ z[:-1] + z[-1])})
        if space.kind == "l2_ball":
            constraints_spec.append(
                {"type": "ineq", "fun": lambda z: space.radius**2 - float(np.dot(z[:-1], z[:-1]))}
            )
        elif space.kind == "simplex":
            constraints_spec.extend(
                [
                    {"type": "ineq", "fun": lambda z: z[:-1]},
                    {"type": "eq", "fun": lambda z: float(np.sum(z[:-1]) - space.radius)},
                ]
            )
        bounds = None
        if space.kind == "box":
            bounds = list(zip(space.lower, space.upper)) + [(None, None)]

        result = minimize(
            lambda z: -float(z[-1]),
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_spec,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )
        if not result.success:
            raise SolverError(f"Incenter optimization failed: {result.message}")
        theta = space.project(result.x[:-1])
        radius = float(result.x[-1])
        return theta, radius, result

    def fit(self, problem: ForwardProblem, dataset: InverseDataset) -> "IncenterEstimator":
        problem.validate_dataset(dataset, check_feasibility=True)
        self.history_ = EstimatorHistory()
        if self.sequential_history:
            for step in range(1, len(dataset) + 1):
                constraints = build_consistency_constraints(problem, dataset[:step])
                theta, radius, result = self._solve(problem, constraints)
                self.history_.append(
                    StepRecord(step, theta.copy(), radius=radius, diagnostics={"solver_message": result.message})
                )
        self.constraints_ = build_consistency_constraints(problem, dataset)
        self.theta_, self.radius_, self.result_ = self._solve(problem, self.constraints_)
        if not self.history_.steps or self.history_.steps[-1].step != len(dataset):
            self.history_.append(StepRecord(len(dataset), self.theta_.copy(), radius=self.radius_))
        return self

    def predict(self, problem: ForwardProblem, context: Any) -> Any:
        return problem.solve(self.theta_, context).decision


@dataclass
class ConsistencyEstimator(IncenterEstimator):
    """A stable feasible representative; currently uses the normalized incenter."""


@dataclass
class ProjectedSubgradientEstimator:
    loss: InverseLoss = field(default_factory=AugmentedSuboptimalityLoss)
    risk: Any = field(default_factory=MeanRisk)
    learning_rate: float = 0.5
    epochs: int = 500
    regularization: float = 1e-3
    stochastic: bool = False
    mirror_descent: bool = False
    seed: int = 0
    record_every: int = 1
    theta0: np.ndarray | None = None
    theta_: np.ndarray = field(init=False)
    history_: EstimatorHistory = field(init=False, default_factory=EstimatorHistory)
    result_: dict[str, Any] = field(init=False, default_factory=dict)

    def fit(self, problem: ForwardProblem, dataset: InverseDataset) -> "ProjectedSubgradientEstimator":
        if self.epochs < 1 or self.learning_rate <= 0:
            raise ValidationError("epochs and learning_rate must be positive")
        problem.validate_dataset(dataset, check_feasibility=False)
        rng = np.random.default_rng(self.seed)
        theta = problem.parameter_space.center() if self.theta0 is None else problem.parameter_space.project(self.theta0)
        if np.linalg.norm(theta) < 1e-12 and problem.parameter_space.kind == "l2_ball":
            theta = problem.parameter_space.sample(1, seed=self.seed, boundary=True)[0] * 0.1
        weights = np.asarray([obs.weight for obs in dataset], dtype=float)
        self.history_ = EstimatorHistory()
        best_theta = theta.copy()
        best_loss = float("inf")
        for epoch in range(1, self.epochs + 1):
            if self.stochastic:
                index = int(rng.integers(len(dataset)))
                selected = [dataset.observations[index]]
                values, gradients, diagnostics = evaluate_losses(self.loss, problem, theta, selected)
                objective_value = float(values[0])
                gradient = gradients[0]
                last_diagnostics = diagnostics[0]
            else:
                values, gradients, diagnostics = evaluate_losses(
                    self.loss, problem, theta, dataset.observations
                )
                objective_value, coefficients = self.risk.aggregate(values, weights)
                gradient = coefficients @ gradients
                last_diagnostics = {"active_competitors": [item.get("competitor") for item in diagnostics]}
            objective_value += 0.5 * self.regularization * float(np.dot(theta, theta))
            gradient = gradient + self.regularization * theta
            rate = self.learning_rate / np.sqrt(epoch)
            if self.mirror_descent and problem.parameter_space.kind == "simplex":
                shifted = -rate * gradient
                shifted -= shifted.max()
                theta = theta * np.exp(shifted)
                theta = problem.parameter_space.project(theta)
            else:
                theta = problem.parameter_space.project(theta - rate * gradient)
            if objective_value < best_loss:
                best_loss = objective_value
                best_theta = theta.copy()
            if epoch == 1 or epoch == self.epochs or epoch % self.record_every == 0:
                self.history_.append(
                    StepRecord(epoch, theta.copy(), loss=objective_value, diagnostics=last_diagnostics)
                )
        self.theta_ = best_theta
        self.result_ = {"best_loss": best_loss, "epochs": self.epochs}
        return self

    def partial_fit(
        self,
        problem: ForwardProblem,
        observation: Any,
        *,
        step: int | None = None,
    ) -> "ProjectedSubgradientEstimator":
        if not hasattr(self, "theta_"):
            self.theta_ = problem.parameter_space.center()
            self.history_ = EstimatorHistory()
        current_step = step or len(self.history_.steps) + 1
        value, gradient, diagnostics = self.loss.value_and_subgradient(problem, self.theta_, observation)
        rate = self.learning_rate / np.sqrt(current_step)
        gradient = observation.weight * gradient + self.regularization * self.theta_
        if self.mirror_descent and problem.parameter_space.kind == "simplex":
            shifted = -rate * gradient
            shifted -= shifted.max()
            self.theta_ = problem.parameter_space.project(self.theta_ * np.exp(shifted))
        else:
            self.theta_ = problem.parameter_space.project(self.theta_ - rate * gradient)
        self.history_.append(StepRecord(current_step, self.theta_.copy(), loss=float(value), diagnostics=diagnostics))
        return self

    def predict(self, problem: ForwardProblem, context: Any) -> Any:
        return problem.solve(self.theta_, context).decision


@dataclass
class OnlineEstimator:
    base_estimator: ProjectedSubgradientEstimator
    passes_per_observation: int = 1
    theta_: np.ndarray = field(init=False)
    history_: EstimatorHistory = field(init=False, default_factory=EstimatorHistory)

    def fit(self, problem: ForwardProblem, dataset: InverseDataset) -> "OnlineEstimator":
        self.base_estimator.history_ = EstimatorHistory()
        if hasattr(self.base_estimator, "theta_"):
            del self.base_estimator.theta_
        for step, observation in enumerate(dataset, start=1):
            for _ in range(self.passes_per_observation):
                self.base_estimator.partial_fit(problem, observation, step=step)
            record = self.base_estimator.history_.steps[-1]
            if self.passes_per_observation > 1:
                # Retain one externally meaningful state per observation.
                self.base_estimator.history_.steps = self.base_estimator.history_.steps[: -self.passes_per_observation]
                self.base_estimator.history_.steps.append(record)
        self.theta_ = self.base_estimator.theta_.copy()
        self.history_ = self.base_estimator.history_
        return self

    def predict(self, problem: ForwardProblem, context: Any) -> Any:
        return problem.solve(self.theta_, context).decision

