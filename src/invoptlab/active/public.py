from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import CapabilityError, ValidationError
from .decision_spaces import (
    ContinuousPolytopeDecisionSpace,
    DAGPathDecisionSpace,
    DecisionSpace,
    FixedCardinalityDecisionSpace,
    IndependentBinaryDecisionSpace,
    StructuredBinaryDecisionSpace,
)


Array = np.ndarray


@dataclass(slots=True)
class ConsistencyNormalBatch:
    normals: Array
    exact: bool
    method: str
    alternatives_considered: int
    skipped_reason: str | None = None


class PublicDecisionProblem:
    """Read-only public access to the known feasible decision problem.

    It contains no true parameter, expert state, or noise realization. A deep
    copy prevents an algorithm from changing the environment's own decision
    space through this interface.
    """

    def __init__(self, decision_space: DecisionSpace):
        self._space = copy.deepcopy(decision_space)
        self.kind = decision_space.kind.value
        self.dimension = decision_space.dimension
        self.is_discrete = decision_space.is_discrete

    def contains(self, decision: Array) -> bool:
        return self._space.contains(np.asarray(decision))

    def minimize(self, cost: Array, rng: np.random.Generator) -> Array:
        return self._space.min_decision(
            np.asarray(cost, dtype=float),
            rng,
            tie_breaking="lexicographic",
        )

    def enumerate_decisions(self) -> tuple[Array, ...]:
        return tuple(item.copy() for item in self._space.enumerate_decisions())

    def minimize_batch(self, costs: Array, rng: np.random.Generator) -> Array:
        """Exact public MIN for multiple costs, with the same deterministic ties.

        Vectorized for binary/cardinality sets, ordinary public solver otherwise.
        This changes neither the feasible set nor the optimizer's accuracy.
        """
        costs = np.asarray(costs, dtype=float)
        if costs.ndim != 2 or costs.shape[1] != self.dimension or not np.all(np.isfinite(costs)):
            raise ValidationError("cost batch must be finite with shape (N, dimension)")
        if isinstance(self._space, IndependentBinaryDecisionSpace):
            return (costs < 0).astype(float)
        if isinstance(self._space, FixedCardinalityDecisionSpace):
            order = np.argsort(costs, axis=1, kind="stable")[:, :self._space.cardinality]
            decisions = np.zeros_like(costs)
            np.put_along_axis(decisions, order, 1., axis=1)
            return decisions
        if isinstance(self._space, StructuredBinaryDecisionSpace):
            return self._space.min_decision_batch(costs, rng)
        if not len(costs):
            return np.empty_like(costs)
        return np.vstack([self.minimize(cost, rng) for cost in costs])

    def description(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "kind": self.kind,
            "dimension": self.dimension,
            "is_discrete": self.is_discrete,
        }
        if isinstance(self._space, FixedCardinalityDecisionSpace):
            value["cardinality"] = self._space.cardinality
        elif isinstance(self._space, ContinuousPolytopeDecisionSpace):
            value.update(
                lower=self._space.lower.tolist(),
                upper=self._space.upper.tolist(),
                A=self._space.A.tolist(),
                b=self._space.b.tolist(),
            )
        elif isinstance(self._space, DAGPathDecisionSpace):
            value.update(
                edges=[list(edge) for edge in self._space.edges],
                source=self._space.source,
                sink=self._space.sink,
            )
        elif isinstance(self._space, StructuredBinaryDecisionSpace):
            value.update(
                A_eq=self._space.A_eq.tolist(),
                b_eq=self._space.b_eq.tolist(),
                C_ub=self._space.C_ub.tolist(),
                r_ub=self._space.r_ub.tolist(),
            )
        return value

    @staticmethod
    def _deduplicate(normals: list[Array], decimals: int = 12) -> Array:
        unique: list[Array] = []
        seen: set[tuple[float, ...]] = set()
        for normal in normals:
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                continue
            key = tuple(np.round(np.asarray(normal) / norm, decimals=decimals))
            if key not in seen:
                seen.add(key)
                unique.append(np.asarray(normal, dtype=float).copy())
        return np.vstack(unique) if unique else np.empty((0, 0))

    def _oracle_alternatives(
        self,
        current_theta: Array,
        query: Array,
        rng: np.random.Generator,
        budget: int,
    ) -> list[Array]:
        probes: list[Array] = [query * current_theta]
        for coordinate in range(self.dimension):
            basis = np.zeros(self.dimension)
            basis[coordinate] = 1.0
            probes.extend([basis, -basis])
        remaining = max(0, budget - len(probes))
        probes.extend(rng.normal(size=(remaining, self.dimension)))
        return [self.minimize(cost, rng) for cost in probes[:budget]]

    def consistency_normals(
        self,
        query: Array,
        observed_decision: Array,
        observation_mask: Array | None,
        current_theta: Array,
        rng: np.random.Generator,
        *,
        alternative_budget: int = 256,
    ) -> ConsistencyNormalBatch:
        """Return normals ``a`` for the public inequalities ``a^T theta <= 0``."""

        query = np.asarray(query, dtype=float).reshape(-1)
        observed = np.asarray(observed_decision, dtype=float).reshape(-1)
        current_theta = np.asarray(current_theta, dtype=float).reshape(-1)
        if query.size != self.dimension or observed.size != self.dimension:
            raise ValidationError("query and observed decision must match the decision dimension")
        if current_theta.size != self.dimension:
            raise ValidationError("current parameter estimate has the wrong dimension")
        if observation_mask is not None:
            mask = np.asarray(observation_mask).reshape(-1)
            if mask.size != self.dimension:
                raise ValidationError("observation mask has the wrong dimension")
            if np.any(mask == 0):
                return ConsistencyNormalBatch(
                    np.empty((0, self.dimension)),
                    exact=False,
                    method="partial-observation-skipped",
                    alternatives_considered=0,
                    skipped_reason="the complete observed decision is unavailable",
                )
        if not self.contains(observed):
            return ConsistencyNormalBatch(
                np.empty((0, self.dimension)),
                exact=False,
                method="infeasible-observation-skipped",
                alternatives_considered=0,
                skipped_reason="the observed decision is not feasible",
            )

        normals: list[Array] = []
        alternatives_considered = 0
        exact = True
        method = "enumeration"

        if isinstance(self._space, IndependentBinaryDecisionSpace):
            method = "binary-coordinate-optimality"
            for coordinate in range(self.dimension):
                alternative = observed.copy()
                alternative[coordinate] = 1.0 - alternative[coordinate]
                normals.append(query * (observed - alternative))
            alternatives_considered = self.dimension
        elif isinstance(self._space, FixedCardinalityDecisionSpace):
            method = "fixed-cardinality-swaps"
            selected = np.flatnonzero(observed > 0.5)
            unselected = np.flatnonzero(observed <= 0.5)
            for remove in selected:
                for add in unselected:
                    alternative = observed.copy()
                    alternative[remove] = 0.0
                    alternative[add] = 1.0
                    normals.append(query * (observed - alternative))
            alternatives_considered = len(selected) * len(unselected)
        elif isinstance(self._space, ContinuousPolytopeDecisionSpace) and not self._space.A.size:
            method = "box-coordinate-extremes"
            for coordinate in range(self.dimension):
                for endpoint in (self._space.lower[coordinate], self._space.upper[coordinate]):
                    alternative = observed.copy()
                    alternative[coordinate] = endpoint
                    normals.append(query * (observed - alternative))
            alternatives_considered = 2 * self.dimension
        else:
            try:
                alternatives = list(self.enumerate_decisions())
            except CapabilityError:
                exact = False
                method = "forward-oracle-cuts"
                alternatives = self._oracle_alternatives(
                    current_theta,
                    query,
                    rng,
                    max(1, int(alternative_budget)),
                )
            alternatives_considered = len(alternatives)
            normals.extend(query * (observed - alternative) for alternative in alternatives)

        matrix = self._deduplicate(normals)
        if matrix.size == 0:
            matrix = np.empty((0, self.dimension))
        return ConsistencyNormalBatch(
            matrix,
            exact=exact,
            method=method,
            alternatives_considered=alternatives_considered,
        )
