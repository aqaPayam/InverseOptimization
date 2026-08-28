from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .capabilities import Capability
from .core import ForwardSolution, Objective, as_array
from .exceptions import CapabilityError, SolverError


@dataclass
class CallableOracle:
    """Adapt user-provided forward solvers to the invoptlab oracle interface."""

    solver: Callable[[Objective, np.ndarray, Any], Any]
    enumerator: Callable[[Any], Sequence[Any]] | None = None
    augmented_solver: Callable[[Objective, np.ndarray, Any, Any, Callable[[Any, Any], float]], Any] | None = None
    declared_capabilities: set[Capability] = field(default_factory=set)

    @property
    def capabilities(self) -> set[Capability]:
        result = set(self.declared_capabilities)
        if self.enumerator is not None:
            result |= {Capability.FINITE_FEASIBLE_SET, Capability.SUPPORTS_ENUMERATION, Capability.SUPPORTS_SEPARATION}
        if self.augmented_solver is not None:
            result.add(Capability.SUPPORTS_SEPARATION)
        return result

    def solve(self, objective: Objective, theta: np.ndarray, context: Any) -> ForwardSolution:
        result = self.solver(objective, theta, context)
        if isinstance(result, ForwardSolution):
            return result
        decision = result[0] if isinstance(result, tuple) else result
        value = result[1] if isinstance(result, tuple) and len(result) > 1 else objective.value(theta, context, decision)
        return ForwardSolution(decision, float(value))

    def enumerate(self, context: Any) -> Sequence[Any]:
        if self.enumerator is None:
            raise CapabilityError("This callable oracle does not support enumeration")
        return list(self.enumerator(context))

    def loss_augmented_solve(
        self,
        objective: Objective,
        theta: np.ndarray,
        context: Any,
        observed_decision: Any,
        distance: Callable[[Any, Any], float],
    ) -> ForwardSolution:
        if self.augmented_solver is not None:
            result = self.augmented_solver(objective, theta, context, observed_decision, distance)
            if isinstance(result, ForwardSolution):
                return result
            decision = result[0] if isinstance(result, tuple) else result
            value = objective.value(theta, context, decision) - distance(observed_decision, decision)
            return ForwardSolution(decision, float(value))
        if self.enumerator is None:
            raise CapabilityError("ASL requires an augmented solver or finite enumeration")
        decisions = self.enumerate(context)
        values = [objective.value(theta, context, item) - distance(observed_decision, item) for item in decisions]
        index = int(np.argmin(values))
        return ForwardSolution(decisions[index], float(values[index]))


@dataclass
class ScipyOracle:
    """Continuous forward solver backed by ``scipy.optimize.minimize``."""

    initial_guess: Callable[[Any], Any]
    bounds: Callable[[Any], Any] | None = None
    constraints: Callable[[Any], Any] | None = None
    method: str = "SLSQP"
    options: dict[str, Any] = field(default_factory=lambda: {"maxiter": 1_000, "ftol": 1e-9})
    declared_capabilities: set[Capability] = field(default_factory=set)

    @property
    def capabilities(self) -> set[Capability]:
        return set(self.declared_capabilities) | {Capability.SUPPORTS_WARM_START}

    def solve(self, objective: Objective, theta: np.ndarray, context: Any) -> ForwardSolution:
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise SolverError("ScipyOracle requires scipy") from exc
        x0 = as_array(self.initial_guess(context)).reshape(-1)
        result = minimize(
            lambda x: objective.value(theta, context, x),
            x0,
            method=self.method,
            bounds=None if self.bounds is None else self.bounds(context),
            constraints=() if self.constraints is None else self.constraints(context),
            options=self.options,
        )
        if not result.success:
            raise SolverError(f"Forward scipy solve failed: {result.message}")
        return ForwardSolution(result.x, float(result.fun), metadata={"iterations": result.nit, "message": result.message})

    def enumerate(self, context: Any):
        raise CapabilityError("A continuous ScipyOracle cannot enumerate its feasible set")


@dataclass
class CVXPYOracle:
    """CVXPY adapter for user-defined continuous convex forward problems.

    ``model_builder(context, theta)`` must return ``(problem, variable)`` where
    ``problem`` is a ``cvxpy.Problem`` and ``variable.value`` is the decision.
    """

    model_builder: Callable[[Any, np.ndarray], tuple[Any, Any]]
    solver: str | None = None
    solve_options: dict[str, Any] = field(default_factory=dict)
    decoder: Callable[[np.ndarray], Any] = lambda value: value
    warm_start: bool = True

    @property
    def capabilities(self) -> set[Capability]:
        return {
            Capability.CONVEX_FORWARD_PROBLEM,
            Capability.SUPPORTS_DUALITY,
            Capability.SUPPORTS_KKT,
            Capability.SUPPORTS_WARM_START,
        }

    def solve(self, objective: Objective, theta: np.ndarray, context: Any) -> ForwardSolution:
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise SolverError("CVXPYOracle requires `pip install invoptlab[convex]`") from exc
        problem, variable = self.model_builder(context, theta)
        value = problem.solve(solver=self.solver, warm_start=self.warm_start, **self.solve_options)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise SolverError(f"CVXPY forward solve ended with status {problem.status}")
        decision = self.decoder(np.asarray(variable.value))
        return ForwardSolution(decision, float(value), status=problem.status, metadata={"solver_stats": repr(problem.solver_stats)})

    def enumerate(self, context: Any):
        raise CapabilityError("CVXPYOracle does not enumerate a continuous feasible set")

