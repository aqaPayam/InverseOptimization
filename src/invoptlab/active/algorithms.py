from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..exceptions import SolverError, ValidationError
from .public import PublicDecisionProblem
from .types import ActiveAction, AlgorithmContext, AlgorithmObservation


Array = np.ndarray


class ActiveAlgorithm(ABC):
    """Minimal interface for algorithms evaluated by the benchmark.

    ``propose`` returns both the current parameter estimate and next query.
    After the environment returns the public observation, ``observe`` updates
    the algorithm and ``current_estimate`` exposes theta_hat after that update.
    """

    name: str = "active-algorithm"

    @abstractmethod
    def reset(self, context: AlgorithmContext, rng: np.random.Generator) -> None:
        ...

    @abstractmethod
    def propose(self, history: Sequence[AlgorithmObservation]) -> ActiveAction:
        ...

    @abstractmethod
    def observe(self, observation: AlgorithmObservation) -> None:
        ...

    @abstractmethod
    def current_estimate(self) -> Array:
        ...

    def diagnostics(self) -> Mapping[str, Any]:
        """Public post-update diagnostics to persist with the current step."""

        return {}


class CallbackActiveAlgorithm(ActiveAlgorithm):
    """Adapter for concise user algorithms implemented as Python callbacks."""

    def __init__(
        self,
        propose: Callable[[AlgorithmContext, Sequence[AlgorithmObservation]], ActiveAction],
        update: Callable[[AlgorithmContext, AlgorithmObservation], Array | None] | None = None,
        *,
        name: str = "callback-algorithm",
    ):
        self._propose_callback = propose
        self._update_callback = update
        self.name = name

    def reset(self, context, rng) -> None:
        self.context = context
        self.rng = rng
        self._estimate = np.zeros(context.dimension)

    def propose(self, history) -> ActiveAction:
        action = self._propose_callback(self.context, history)
        self._estimate = np.asarray(action.theta_hat, dtype=float).copy()
        return action

    def observe(self, observation) -> None:
        if self._update_callback is not None:
            estimate = self._update_callback(self.context, observation)
            if estimate is not None:
                self._estimate = np.asarray(estimate, dtype=float).copy()

    def current_estimate(self) -> Array:
        return self._estimate.copy()


class RandomActiveAlgorithm(ActiveAlgorithm):
    """Meaningless baseline used only to verify benchmark plumbing."""

    name = "random-smoke-test"

    def __init__(self, *, without_replacement: bool = False):
        self.without_replacement = without_replacement

    @staticmethod
    def _random_theta(dimension: int, rng: np.random.Generator) -> Array:
        value = rng.normal(size=dimension)
        norm = np.linalg.norm(value)
        return value / norm if norm > 1e-15 else np.eye(1, dimension, 0).reshape(-1)

    def reset(self, context, rng) -> None:
        self.context = context
        self.rng = rng
        self._estimate = self._random_theta(context.dimension, rng)
        self._available = list(range(context.query_candidates.shape[0]))

    def propose(self, history) -> ActiveAction:
        del history
        if self.without_replacement and self._available:
            position = int(self.rng.integers(len(self._available)))
            index = self._available.pop(position)
        else:
            index = int(self.rng.integers(self.context.query_candidates.shape[0]))
        return ActiveAction(
            query=self.context.query_candidates[index].copy(),
            theta_hat=self._estimate.copy(),
            diagnostics={"candidate_index": index, "purpose": "plumbing-only"},
        )

    def observe(self, observation) -> None:
        del observation
        self._estimate = self._random_theta(self.context.dimension, self.rng)

    def current_estimate(self) -> Array:
        return self._estimate.copy()


class UniformRandomIncenterAlgorithm(ActiveAlgorithm):
    """Uniform random queries with a sequential hard-consistency incenter.

    The estimate uses only public ``(s, y)`` observations. Complete partial
    observations are skipped because their latent decision is unavailable.
    """

    name = "uniform-random-sequential-incenter"

    def __init__(
        self,
        *,
        tolerance: float = 1e-8,
        max_iterations: int = 2_000,
        alternative_budget: int = 256,
        deduplication_decimals: int = 10,
    ):
        if tolerance <= 0 or max_iterations < 1 or alternative_budget < 1:
            raise ValidationError(
                "tolerance, max_iterations, and alternative_budget must be positive"
            )
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.alternative_budget = int(alternative_budget)
        self.deduplication_decimals = int(deduplication_decimals)

    @staticmethod
    def _unit_initial_estimate(dimension: int) -> Array:
        return np.ones(dimension, dtype=float) / np.sqrt(dimension)

    def reset(self, context, rng) -> None:
        if not isinstance(context.decision_problem, PublicDecisionProblem):
            raise ValidationError(
                "UniformRandomIncenterAlgorithm requires a public decision problem"
            )
        self.context = context
        self.decision_problem = context.decision_problem
        self.rng = rng
        self._estimate = self._unit_initial_estimate(context.dimension)
        self.incenter_radius_ = 0.0
        self.constraints_ = np.empty((0, context.dimension))
        self.constraint_sources_: list[dict[str, object]] = []
        self.incenter_history_: list[dict[str, object]] = []
        query_config = context.public_environment.get("query_space", {})
        self.allow_repeated_queries = bool(
            query_config.get("allow_repeated_queries", True)
        )
        self._available_queries = list(range(context.query_candidates.shape[0]))

    def _select_query_index(self) -> int:
        if self.allow_repeated_queries:
            return int(self.rng.integers(self.context.query_candidates.shape[0]))
        if not self._available_queries:
            raise ValidationError("the non-repeating query set has been exhausted")
        position = int(self.rng.integers(len(self._available_queries)))
        return self._available_queries.pop(position)

    def propose(self, history) -> ActiveAction:
        del history
        index = self._select_query_index()
        return ActiveAction(
            query=self.context.query_candidates[index].copy(),
            theta_hat=self._estimate.copy(),
            diagnostics={
                "query_rule": "uniform-random",
                "candidate_index": index,
                "constraint_count": int(self.constraints_.shape[0]),
                "incenter_radius": self.incenter_radius_,
                "all_constraints_exact": all(
                    bool(item["exact"]) for item in self.constraint_sources_
                ),
            },
        )

    def _append_deduplicated(self, new_normals: Array) -> None:
        combined = (
            np.vstack([self.constraints_, new_normals])
            if self.constraints_.size and new_normals.size
            else new_normals.copy()
            if new_normals.size
            else self.constraints_.copy()
        )
        unique: list[Array] = []
        seen: set[tuple[float, ...]] = set()
        for normal in combined:
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                continue
            normalized = normal / norm
            key = tuple(np.round(normalized, self.deduplication_decimals))
            if key not in seen:
                seen.add(key)
                unique.append(normal.copy())
        self.constraints_ = (
            np.vstack(unique) if unique else np.empty((0, self.context.dimension))
        )

    def _solve_incenter(self) -> tuple[Array, float, str]:
        try:
            from scipy.optimize import minimize
        except ImportError as exc:  # pragma: no cover - scipy is a core dependency
            raise SolverError("the sequential incenter requires scipy") from exc

        norms = np.linalg.norm(self.constraints_, axis=1)
        matrix = self.constraints_[norms > 1e-12] / norms[norms > 1e-12, None]
        if not matrix.size:
            return self._estimate.copy(), self.incenter_radius_, "no constraints"

        candidate = self._estimate.copy()
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm > 1e-12:
            candidate *= min(0.8, 0.8 / candidate_norm)
        margins = -(matrix @ candidate)
        if np.min(margins) < 0:
            candidate = np.zeros(self.context.dimension)
            initial_radius = 0.0
        else:
            initial_radius = max(0.0, 0.8 * float(np.min(margins)))
        initial = np.concatenate([candidate, [initial_radius]])
        constraints = [
            {"type": "ineq", "fun": lambda z: -(matrix @ z[:-1] + z[-1])},
            {
                "type": "ineq",
                "fun": lambda z: 1.0 - float(np.dot(z[:-1], z[:-1])),
            },
            {"type": "ineq", "fun": lambda z: float(z[-1])},
        ]
        result = minimize(
            lambda z: -float(z[-1]),
            initial,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )
        if not result.success:
            raise SolverError(f"sequential incenter optimization failed: {result.message}")
        theta = np.asarray(result.x[:-1], dtype=float)
        theta_norm = float(np.linalg.norm(theta))
        if theta_norm > 1.0 + 10 * self.tolerance:
            theta /= theta_norm
        radius = max(0.0, float(result.x[-1]))
        return theta, radius, str(result.message)

    def observe(self, observation) -> None:
        batch = self.decision_problem.consistency_normals(
            observation.query,
            observation.observed_decision,
            observation.observation_mask,
            self._estimate,
            self.rng,
            alternative_budget=self.alternative_budget,
        )
        self.constraint_sources_.append(
            {
                "step": observation.step,
                "method": batch.method,
                "exact": batch.exact,
                "alternatives_considered": batch.alternatives_considered,
                "constraints_generated": int(batch.normals.shape[0]),
                "skipped_reason": batch.skipped_reason,
            }
        )
        self._append_deduplicated(batch.normals)
        solver_message = "observation skipped"
        if batch.skipped_reason is None and self.constraints_.size:
            self._estimate, self.incenter_radius_, solver_message = self._solve_incenter()
        self.incenter_history_.append(
            {
                "step": observation.step,
                "theta_hat": self._estimate.copy(),
                "incenter_radius": self.incenter_radius_,
                "constraint_count": int(self.constraints_.shape[0]),
                "solver_message": solver_message,
                "constraint_method": batch.method,
                "constraints_exact": batch.exact,
                "skipped_reason": batch.skipped_reason,
            }
        )

    def current_estimate(self) -> Array:
        return self._estimate.copy()

    def diagnostics(self) -> Mapping[str, Any]:
        if not self.incenter_history_:
            return {
                "constraint_count": 0,
                "incenter_radius": self.incenter_radius_,
            }
        latest = self.incenter_history_[-1]
        return {
            "constraint_count": latest["constraint_count"],
            "incenter_radius": latest["incenter_radius"],
            "solver_message": latest["solver_message"],
            "constraint_method": latest["constraint_method"],
            "constraints_exact": latest["constraints_exact"],
            "skipped_reason": latest["skipped_reason"],
        }


def create_uniform_random_incenter_algorithm() -> UniformRandomIncenterAlgorithm:
    return UniformRandomIncenterAlgorithm()
