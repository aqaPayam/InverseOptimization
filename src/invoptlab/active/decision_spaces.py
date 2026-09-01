from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations, product
from math import comb
from typing import Iterable, Sequence

import numpy as np

from ..exceptions import CapabilityError, SolverError, ValidationError
from .config import DecisionSpaceConfig, DecisionSpaceKind


Array = np.ndarray


def _as_vector(value: Sequence[float] | Array, dimension: int, name: str) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != dimension or not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} must be finite and have dimension {dimension}")
    return result


def _sample_probabilities(log_weights: Array, rng: np.random.Generator) -> int:
    shifted = np.asarray(log_weights, dtype=float) - float(np.max(log_weights))
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return int(rng.choice(probabilities.size, p=probabilities))


class DecisionSpace(ABC):
    kind: DecisionSpaceKind
    is_discrete: bool

    def __init__(self, dimension: int, max_enumeration: int = 65_536):
        self.dimension = int(dimension)
        self.max_enumeration = int(max_enumeration)

    @abstractmethod
    def min_decision(
        self,
        cost: Array,
        rng: np.random.Generator,
        *,
        tie_breaking: str = "lexicographic",
    ) -> Array:
        ...

    @abstractmethod
    def sample_gibbs(
        self,
        cost: Array,
        temperature: float,
        rng: np.random.Generator,
        *,
        burn_in: int = 40,
        steps: int = 20,
    ) -> Array:
        ...

    @abstractmethod
    def contains(self, decision: Array, tolerance: float = 1e-8) -> bool:
        ...

    @abstractmethod
    def project(self, decision: Array) -> Array:
        ...

    @abstractmethod
    def sample_feasible(self, rng: np.random.Generator) -> Array:
        ...

    def enumerate_decisions(self) -> list[Array]:
        raise CapabilityError(f"{type(self).__name__} does not support exact enumeration")

    def sample_local(
        self,
        decision: Array,
        sigma: float,
        rng: np.random.Generator,
        *,
        distance: str = "euclidean",
        burn_in: int = 40,
        steps: int = 20,
    ) -> Array:
        if sigma <= 0:
            return np.asarray(decision).copy()
        if not self.is_discrete:
            return self.project(np.asarray(decision, dtype=float) + rng.normal(scale=sigma, size=self.dimension))
        try:
            decisions = self.enumerate_decisions()
        except CapabilityError:
            return self.project(np.asarray(decision, dtype=float) + rng.normal(scale=sigma, size=self.dimension))
        reference = np.asarray(decision)
        if distance == "hamming":
            distances = np.asarray([np.sum(candidate != reference) for candidate in decisions], dtype=float)
        else:
            distances = np.asarray([np.linalg.norm(candidate - reference) for candidate in decisions], dtype=float)
        index = _sample_probabilities(-(distances**2) / (2 * sigma**2), rng)
        return decisions[index].copy()

    def reference_energy_gap(self, cost: Array, rng: np.random.Generator) -> float:
        try:
            decisions = self.enumerate_decisions()
        except CapabilityError:
            low = self.min_decision(cost, rng)
            samples = [self.sample_feasible(rng) for _ in range(32)]
            values = sorted(float(np.dot(cost, item)) for item in [low, *samples])
        else:
            values = sorted(float(np.dot(cost, item)) for item in decisions)
        if len(values) < 2:
            return 0.0
        best = values[0]
        return next((value - best for value in values[1:] if value > best + 1e-12), 0.0)


class IndependentBinaryDecisionSpace(DecisionSpace):
    kind = DecisionSpaceKind.INDEPENDENT_BINARY
    is_discrete = True

    def min_decision(self, cost, rng, *, tie_breaking="lexicographic") -> Array:
        cost = _as_vector(cost, self.dimension, "cost")
        decision = (cost < 0).astype(int)
        ties = np.flatnonzero(np.abs(cost) <= 1e-12)
        if tie_breaking == "random" and ties.size:
            decision[ties] = rng.integers(0, 2, size=ties.size)
        return decision

    def sample_gibbs(self, cost, temperature, rng, *, burn_in=40, steps=20) -> Array:
        del burn_in, steps
        if temperature <= 0:
            return self.min_decision(cost, rng)
        cost = _as_vector(cost, self.dimension, "cost")
        scaled = np.clip(cost / temperature, -700, 700)
        probabilities = 1.0 / (1.0 + np.exp(scaled))
        return (rng.random(self.dimension) < probabilities).astype(int)

    def contains(self, decision, tolerance=1e-8) -> bool:
        value = np.asarray(decision, dtype=float).reshape(-1)
        return value.size == self.dimension and bool(
            np.all((np.abs(value) <= tolerance) | (np.abs(value - 1) <= tolerance))
        )

    def project(self, decision) -> Array:
        return (np.asarray(decision, dtype=float).reshape(-1) >= 0.5).astype(int)

    def sample_feasible(self, rng) -> Array:
        return rng.integers(0, 2, size=self.dimension)

    def enumerate_decisions(self) -> list[Array]:
        if 2**self.dimension > self.max_enumeration:
            raise CapabilityError("independent binary space is too large to enumerate")
        return [np.asarray(bits, dtype=int) for bits in product((0, 1), repeat=self.dimension)]

    def sample_local(self, decision, sigma, rng, **kwargs) -> Array:
        del kwargs
        if sigma <= 0:
            return np.asarray(decision, dtype=int).copy()
        log_odds = -1.0 / (2 * sigma**2)
        flip_probability = float(np.exp(log_odds) / (1 + np.exp(log_odds)))
        flips = rng.random(self.dimension) < flip_probability
        return np.logical_xor(np.asarray(decision, dtype=bool), flips).astype(int)

    def reference_energy_gap(self, cost, rng) -> float:
        del rng
        positive = np.abs(_as_vector(cost, self.dimension, "cost"))
        positive = positive[positive > 1e-12]
        return float(np.min(positive)) if positive.size else 0.0


class FixedCardinalityDecisionSpace(DecisionSpace):
    kind = DecisionSpaceKind.FIXED_CARDINALITY
    is_discrete = True

    def __init__(self, dimension: int, cardinality: int, max_enumeration: int = 65_536):
        super().__init__(dimension, max_enumeration)
        if not 1 <= cardinality <= dimension:
            raise ValidationError("cardinality must lie between one and dimension")
        self.cardinality = int(cardinality)

    def min_decision(self, cost, rng, *, tie_breaking="lexicographic") -> Array:
        cost = _as_vector(cost, self.dimension, "cost")
        adjusted = cost.copy()
        if tie_breaking == "random":
            adjusted += rng.uniform(-1e-11, 1e-11, size=self.dimension)
        order = np.lexsort((np.arange(self.dimension), adjusted))
        decision = np.zeros(self.dimension, dtype=int)
        decision[order[: self.cardinality]] = 1
        return decision

    def sample_gibbs(self, cost, temperature, rng, *, burn_in=40, steps=20) -> Array:
        del burn_in, steps
        if temperature <= 0:
            return self.min_decision(cost, rng)
        cost = _as_vector(cost, self.dimension, "cost")
        logits = -cost / temperature
        logits -= np.max(logits)
        d, k = self.dimension, self.cardinality
        log_dp = np.full((d + 1, k + 1), -np.inf)
        log_dp[:, 0] = 0.0
        for index in range(1, d + 1):
            for count in range(1, min(index, k) + 1):
                log_dp[index, count] = np.logaddexp(
                    log_dp[index - 1, count],
                    logits[index - 1] + log_dp[index - 1, count - 1],
                )
        decision = np.zeros(d, dtype=int)
        remaining = k
        for index in range(d, 0, -1):
            if remaining == 0:
                break
            if index == remaining:
                decision[:index] = 1
                break
            log_include = logits[index - 1] + log_dp[index - 1, remaining - 1]
            probability = float(np.exp(log_include - log_dp[index, remaining]))
            if rng.random() < np.clip(probability, 0.0, 1.0):
                decision[index - 1] = 1
                remaining -= 1
        return decision

    def contains(self, decision, tolerance=1e-8) -> bool:
        value = np.asarray(decision, dtype=float).reshape(-1)
        binary = np.all((np.abs(value) <= tolerance) | (np.abs(value - 1) <= tolerance))
        return value.size == self.dimension and bool(binary and abs(value.sum() - self.cardinality) <= tolerance)

    def project(self, decision) -> Array:
        value = _as_vector(decision, self.dimension, "decision")
        order = np.argsort(-value, kind="stable")
        result = np.zeros(self.dimension, dtype=int)
        result[order[: self.cardinality]] = 1
        return result

    def sample_feasible(self, rng) -> Array:
        result = np.zeros(self.dimension, dtype=int)
        result[rng.choice(self.dimension, size=self.cardinality, replace=False)] = 1
        return result

    def enumerate_decisions(self) -> list[Array]:
        if comb(self.dimension, self.cardinality) > self.max_enumeration:
            raise CapabilityError("fixed-cardinality space is too large to enumerate")
        values = []
        for selected in combinations(range(self.dimension), self.cardinality):
            decision = np.zeros(self.dimension, dtype=int)
            decision[list(selected)] = 1
            values.append(decision)
        return values

    def sample_local(self, decision, sigma, rng, **kwargs) -> Array:
        del kwargs
        reference = np.asarray(decision, dtype=int)
        if sigma <= 0 or self.cardinality == self.dimension:
            return reference.copy()
        current = reference.copy()
        iterations = max(10, 4 * self.dimension)
        for _ in range(iterations):
            ones = np.flatnonzero(current)
            zeros = np.flatnonzero(1 - current)
            remove = int(rng.choice(ones))
            add = int(rng.choice(zeros))
            proposal = current.copy()
            proposal[remove], proposal[add] = 0, 1
            old_distance = float(np.sum(current != reference))
            new_distance = float(np.sum(proposal != reference))
            log_ratio = -(new_distance - old_distance) / (2 * sigma**2)
            if np.log(rng.random()) < min(0.0, log_ratio):
                current = proposal
        return current

    def reference_energy_gap(self, cost, rng) -> float:
        del rng
        sorted_cost = np.sort(_as_vector(cost, self.dimension, "cost"))
        if self.cardinality == self.dimension:
            return 0.0
        return float(max(0.0, sorted_cost[self.cardinality] - sorted_cost[self.cardinality - 1]))


class ContinuousPolytopeDecisionSpace(DecisionSpace):
    kind = DecisionSpaceKind.CONTINUOUS_POLYTOPE
    is_discrete = False

    def __init__(
        self,
        dimension: int,
        *,
        A: Sequence[Sequence[float]] | None = None,
        b: Sequence[float] | None = None,
        lower: Sequence[float] | None = None,
        upper: Sequence[float] | None = None,
        max_enumeration: int = 65_536,
    ):
        super().__init__(dimension, max_enumeration)
        self.lower = np.full(dimension, -1.0) if lower is None else _as_vector(lower, dimension, "lower")
        self.upper = np.full(dimension, 1.0) if upper is None else _as_vector(upper, dimension, "upper")
        if np.any(self.lower >= self.upper):
            raise ValidationError("polytope lower bounds must be below upper bounds")
        if A is None:
            self.A = np.empty((0, dimension))
            self.b = np.empty(0)
        else:
            self.A = np.asarray(A, dtype=float)
            self.b = np.asarray(b, dtype=float).reshape(-1) if b is not None else np.empty(0)
            if self.A.ndim != 2 or self.A.shape[1] != dimension or self.A.shape[0] != self.b.size:
                raise ValidationError("A and b have incompatible polytope shapes")
        self._start = self._find_feasible_point()

    def _find_feasible_point(self) -> Array:
        try:
            from scipy.optimize import linprog
        except ImportError as exc:
            raise SolverError("continuous polytopes require scipy") from exc
        result = linprog(
            np.zeros(self.dimension),
            A_ub=None if not self.A.size else self.A,
            b_ub=None if not self.b.size else self.b,
            bounds=list(zip(self.lower, self.upper)),
            method="highs",
        )
        if not result.success:
            raise ValidationError(f"polytope is empty or invalid: {result.message}")
        center = np.clip((self.lower + self.upper) / 2, self.lower, self.upper)
        return center if self.contains(center) else np.asarray(result.x, dtype=float)

    def min_decision(self, cost, rng, *, tie_breaking="lexicographic") -> Array:
        del rng, tie_breaking
        from scipy.optimize import linprog

        result = linprog(
            _as_vector(cost, self.dimension, "cost"),
            A_ub=None if not self.A.size else self.A,
            b_ub=None if not self.b.size else self.b,
            bounds=list(zip(self.lower, self.upper)),
            method="highs",
        )
        if not result.success:
            raise SolverError(f"polytope optimization failed: {result.message}")
        return np.asarray(result.x, dtype=float)

    def _line_interval(self, point: Array, direction: Array) -> tuple[float, float]:
        lower, upper = -np.inf, np.inf
        if self.A.size:
            denominators = self.A @ direction
            slacks = self.b - self.A @ point
            if np.any((np.abs(denominators) <= 1e-14) & (slacks < -1e-10)):
                raise SolverError("hit-and-run point became infeasible")
            for denominator, slack in zip(denominators, slacks):
                if denominator > 1e-14:
                    upper = min(upper, float(slack / denominator))
                elif denominator < -1e-14:
                    lower = max(lower, float(slack / denominator))
        for index, component in enumerate(direction):
            if component > 1e-14:
                lower = max(lower, float((self.lower[index] - point[index]) / component))
                upper = min(upper, float((self.upper[index] - point[index]) / component))
            elif component < -1e-14:
                lower = max(lower, float((self.upper[index] - point[index]) / component))
                upper = min(upper, float((self.lower[index] - point[index]) / component))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper + 1e-12:
            raise SolverError("polytope must be bounded for hit-and-run sampling")
        return lower, upper

    @staticmethod
    def _truncated_exponential(
        lower: float,
        upper: float,
        rate: float,
        rng: np.random.Generator,
    ) -> float:
        length = max(0.0, upper - lower)
        if length <= 1e-15:
            return lower
        if abs(rate) <= 1e-10:
            return float(rng.uniform(lower, upper))
        if rate < 0:
            return upper - ContinuousPolytopeDecisionSpace._truncated_exponential(
                0.0, length, -rate, rng
            )
        denominator = -np.expm1(-rate * length)
        distance = -np.log1p(-rng.random() * denominator) / rate
        return float(lower + min(distance, length))

    def _hit_and_run(self, cost, temperature, rng, iterations) -> Array:
        point = self._start.copy()
        for _ in range(max(1, iterations)):
            direction = rng.normal(size=self.dimension)
            norm = np.linalg.norm(direction)
            if norm <= 1e-15:
                continue
            direction /= norm
            lower, upper = self._line_interval(point, direction)
            rate = 0.0 if cost is None else float(np.dot(cost, direction) / temperature)
            alpha = self._truncated_exponential(lower, upper, rate, rng)
            point = point + alpha * direction
        return np.clip(point, self.lower, self.upper)

    def sample_gibbs(self, cost, temperature, rng, *, burn_in=40, steps=20) -> Array:
        if temperature <= 0:
            return self.min_decision(cost, rng)
        return self._hit_and_run(
            _as_vector(cost, self.dimension, "cost"), temperature, rng, burn_in + steps
        )

    def contains(self, decision, tolerance=1e-8) -> bool:
        value = np.asarray(decision, dtype=float).reshape(-1)
        if value.size != self.dimension or np.any(value < self.lower - tolerance) or np.any(value > self.upper + tolerance):
            return False
        return not self.A.size or bool(np.all(self.A @ value <= self.b + tolerance))

    def project(self, decision) -> Array:
        target = _as_vector(decision, self.dimension, "decision")
        clipped = np.clip(target, self.lower, self.upper)
        if self.contains(clipped):
            return clipped
        from scipy.optimize import minimize

        constraints = [] if not self.A.size else [
            {"type": "ineq", "fun": lambda x: self.b - self.A @ x}
        ]
        result = minimize(
            lambda x: 0.5 * float(np.dot(x - target, x - target)),
            self._start,
            method="SLSQP",
            bounds=list(zip(self.lower, self.upper)),
            constraints=constraints,
        )
        if not result.success:
            raise SolverError(f"polytope projection failed: {result.message}")
        return np.asarray(result.x, dtype=float)

    def sample_feasible(self, rng) -> Array:
        return self._hit_and_run(None, 1.0, rng, max(20, 4 * self.dimension))

    def reference_energy_gap(self, cost, rng) -> float:
        cost = _as_vector(cost, self.dimension, "cost")
        minimum = float(np.dot(cost, self.min_decision(cost, rng)))
        maximum = -float(np.dot(-cost, self.min_decision(-cost, rng)))
        return max(0.0, (maximum - minimum) / max(1, self.dimension))


def generate_dag_edges(dimension: int, rng: np.random.Generator) -> tuple[list[tuple[int, int]], int, int]:
    nodes = 2
    while nodes * (nodes - 1) // 2 < dimension:
        nodes += 1
    if dimension < nodes - 1:
        nodes = dimension + 1
    source, sink = 0, nodes - 1
    chain = [(node, node + 1) for node in range(nodes - 1)]
    candidates = [
        (left, right)
        for left in range(nodes)
        for right in range(left + 1, nodes)
        if (left, right) not in chain
    ]
    rng.shuffle(candidates)
    edges = chain + candidates[: max(0, dimension - len(chain))]
    if len(edges) != dimension:
        raise ValidationError("could not generate a structured DAG with the requested dimension")
    return edges, source, sink


class DAGPathDecisionSpace(DecisionSpace):
    kind = DecisionSpaceKind.STRUCTURED
    is_discrete = True

    def __init__(
        self,
        edges: Sequence[Sequence[int]],
        source: int,
        sink: int,
        *,
        max_enumeration: int = 65_536,
    ):
        parsed = [(int(edge[0]), int(edge[1])) for edge in edges]
        super().__init__(len(parsed), max_enumeration)
        if not parsed or any(left >= right for left, right in parsed):
            raise ValidationError("DAG edges must be ordered pairs (u, v) with u < v")
        self.edges = parsed
        self.source = int(source)
        self.sink = int(sink)
        self.nodes = sorted({item for edge in parsed for item in edge})
        self.outgoing: dict[int, list[tuple[int, int]]] = {node: [] for node in self.nodes}
        self.incoming: dict[int, list[tuple[int, int]]] = {node: [] for node in self.nodes}
        for index, (left, right) in enumerate(parsed):
            self.outgoing[left].append((index, right))
            self.incoming[right].append((index, left))
        if self.source not in self.outgoing or self.sink not in self.incoming:
            raise ValidationError("source and sink must be present in the DAG")
        self.min_decision(np.zeros(self.dimension), np.random.default_rng(0))

    def _shortest_path(self, cost: Array, forbidden: set[int] | None = None) -> tuple[Array, float]:
        forbidden = forbidden or set()
        distances = {node: np.inf for node in self.nodes}
        predecessor: dict[int, tuple[int, int]] = {}
        distances[self.source] = 0.0
        for node in self.nodes:
            if not np.isfinite(distances[node]):
                continue
            for edge_index, target in self.outgoing.get(node, []):
                if edge_index in forbidden:
                    continue
                candidate = distances[node] + float(cost[edge_index])
                if candidate < distances[target] - 1e-12:
                    distances[target] = candidate
                    predecessor[target] = (edge_index, node)
        if not np.isfinite(distances.get(self.sink, np.inf)):
            raise SolverError("DAG has no feasible source-to-sink path")
        decision = np.zeros(self.dimension, dtype=int)
        node = self.sink
        while node != self.source:
            edge_index, previous = predecessor[node]
            decision[edge_index] = 1
            node = previous
        return decision, float(distances[self.sink])

    def min_decision(self, cost, rng, *, tie_breaking="lexicographic") -> Array:
        cost = _as_vector(cost, self.dimension, "cost")
        if tie_breaking == "random":
            cost = cost + rng.uniform(-1e-11, 1e-11, size=self.dimension)
        return self._shortest_path(cost)[0]

    def sample_gibbs(self, cost, temperature, rng, *, burn_in=40, steps=20) -> Array:
        del burn_in, steps
        if temperature <= 0:
            return self.min_decision(cost, rng)
        cost = _as_vector(cost, self.dimension, "cost")
        log_partition = {node: -np.inf for node in self.nodes}
        log_partition[self.sink] = 0.0
        for node in reversed(self.nodes):
            outgoing = self.outgoing.get(node, [])
            terms = [
                -cost[edge_index] / temperature + log_partition[target]
                for edge_index, target in outgoing
                if np.isfinite(log_partition[target])
            ]
            if terms:
                total = terms[0]
                for term in terms[1:]:
                    total = float(np.logaddexp(total, term))
                log_partition[node] = total
        if not np.isfinite(log_partition[self.source]):
            raise SolverError("DAG has no feasible source-to-sink path")
        decision = np.zeros(self.dimension, dtype=int)
        node = self.source
        while node != self.sink:
            options = [
                (edge_index, target)
                for edge_index, target in self.outgoing.get(node, [])
                if np.isfinite(log_partition[target])
            ]
            log_weights = np.asarray([
                -cost[edge_index] / temperature + log_partition[target]
                for edge_index, target in options
            ])
            edge_index, node = options[_sample_probabilities(log_weights, rng)]
            decision[edge_index] = 1
        return decision

    def contains(self, decision, tolerance=1e-8) -> bool:
        value = np.asarray(decision, dtype=float).reshape(-1)
        if value.size != self.dimension or not np.all(
            (np.abs(value) <= tolerance) | (np.abs(value - 1) <= tolerance)
        ):
            return False
        balance = {node: 0.0 for node in self.nodes}
        for selected, (left, right) in zip(value, self.edges):
            balance[left] += selected
            balance[right] -= selected
        return all(
            abs(balance[node] - (1.0 if node == self.source else -1.0 if node == self.sink else 0.0)) <= tolerance
            for node in self.nodes
        )

    def project(self, decision) -> Array:
        target = _as_vector(decision, self.dimension, "decision")
        return self.min_decision(1.0 - 2.0 * target, np.random.default_rng(0))

    def sample_feasible(self, rng) -> Array:
        return self.sample_gibbs(np.zeros(self.dimension), 1.0, rng)

    def enumerate_decisions(self) -> list[Array]:
        paths: list[Array] = []

        def visit(node: int, selected: list[int]) -> None:
            if len(paths) > self.max_enumeration:
                return
            if node == self.sink:
                decision = np.zeros(self.dimension, dtype=int)
                decision[selected] = 1
                paths.append(decision)
                return
            for edge_index, target in self.outgoing.get(node, []):
                visit(target, [*selected, edge_index])

        visit(self.source, [])
        if len(paths) > self.max_enumeration:
            raise CapabilityError("DAG path space is too large to enumerate")
        return paths

    def sample_local(self, decision, sigma, rng, **kwargs) -> Array:
        del kwargs
        reference = np.asarray(decision, dtype=float)
        if sigma <= 0:
            return reference.astype(int)
        local_cost = 1.0 - 2.0 * reference
        return self.sample_gibbs(local_cost, 2 * sigma**2, rng)

    def reference_energy_gap(self, cost, rng) -> float:
        del rng
        cost = _as_vector(cost, self.dimension, "cost")
        best, best_value = self._shortest_path(cost)
        alternatives = []
        for edge_index in np.flatnonzero(best):
            try:
                _, value = self._shortest_path(cost, {int(edge_index)})
            except SolverError:
                continue
            alternatives.append(value)
        return max(0.0, min(alternatives) - best_value) if alternatives else 0.0


class StructuredBinaryDecisionSpace(DecisionSpace):
    kind = DecisionSpaceKind.STRUCTURED
    is_discrete = True

    def __init__(
        self,
        dimension: int,
        *,
        A_eq: Sequence[Sequence[float]] | None = None,
        b_eq: Sequence[float] | None = None,
        C_ub: Sequence[Sequence[float]] | None = None,
        r_ub: Sequence[float] | None = None,
        max_enumeration: int = 65_536,
    ):
        super().__init__(dimension, max_enumeration)
        self.A_eq = np.empty((0, dimension)) if A_eq is None else np.asarray(A_eq, dtype=float)
        self.b_eq = np.empty(0) if b_eq is None else np.asarray(b_eq, dtype=float).reshape(-1)
        self.C_ub = np.empty((0, dimension)) if C_ub is None else np.asarray(C_ub, dtype=float)
        self.r_ub = np.empty(0) if r_ub is None else np.asarray(r_ub, dtype=float).reshape(-1)
        if self.A_eq.shape != (self.b_eq.size, dimension):
            raise ValidationError("A_eq and b_eq have incompatible shapes")
        if self.C_ub.shape != (self.r_ub.size, dimension):
            raise ValidationError("C_ub and r_ub have incompatible shapes")
        self.min_decision(np.zeros(dimension), np.random.default_rng(0))

    def _solve(self, cost: Array, extra_constraints: list | None = None) -> Array:
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
        except ImportError as exc:
            raise SolverError("structured binary spaces require scipy.optimize.milp") from exc
        constraints = []
        if self.A_eq.size:
            constraints.append(LinearConstraint(self.A_eq, self.b_eq, self.b_eq))
        if self.C_ub.size:
            constraints.append(LinearConstraint(self.C_ub, -np.inf, self.r_ub))
        constraints.extend(extra_constraints or [])
        result = milp(
            c=cost,
            integrality=np.ones(self.dimension),
            bounds=Bounds(np.zeros(self.dimension), np.ones(self.dimension)),
            constraints=constraints,
        )
        if not result.success:
            raise SolverError(f"structured binary optimization failed: {result.message}")
        return np.rint(result.x).astype(int)

    def min_decision(self, cost, rng, *, tie_breaking="lexicographic") -> Array:
        cost = _as_vector(cost, self.dimension, "cost")
        if tie_breaking == "random":
            cost = cost + rng.uniform(-1e-11, 1e-11, self.dimension)
        return self._solve(cost)

    def sample_gibbs(self, cost, temperature, rng, *, burn_in=40, steps=20) -> Array:
        del burn_in, steps
        if temperature <= 0:
            return self.min_decision(cost, rng)
        decisions = self.enumerate_decisions()
        values = np.asarray([np.dot(cost, decision) for decision in decisions], dtype=float)
        return decisions[_sample_probabilities(-values / temperature, rng)].copy()

    def contains(self, decision, tolerance=1e-8) -> bool:
        value = np.asarray(decision, dtype=float).reshape(-1)
        if value.size != self.dimension or not np.all(
            (np.abs(value) <= tolerance) | (np.abs(value - 1) <= tolerance)
        ):
            return False
        if self.A_eq.size and not np.allclose(self.A_eq @ value, self.b_eq, atol=tolerance):
            return False
        return not self.C_ub.size or bool(np.all(self.C_ub @ value <= self.r_ub + tolerance))

    def project(self, decision) -> Array:
        target = _as_vector(decision, self.dimension, "decision")
        return self._solve(1.0 - 2.0 * target)

    def sample_feasible(self, rng) -> Array:
        return self._solve(rng.normal(size=self.dimension))

    def enumerate_decisions(self) -> list[Array]:
        if 2**self.dimension > self.max_enumeration:
            raise CapabilityError("structured binary space is too large for exact Gibbs enumeration")
        values = [
            np.asarray(bits, dtype=int)
            for bits in product((0, 1), repeat=self.dimension)
            if self.contains(np.asarray(bits, dtype=int))
        ]
        if not values:
            raise SolverError("structured binary space has no feasible decisions")
        return values

    def reference_energy_gap(self, cost, rng) -> float:
        del rng
        from scipy.optimize import LinearConstraint

        cost = _as_vector(cost, self.dimension, "cost")
        best = self._solve(cost)
        # Exclude the incumbent: sum_{j:x=0} z_j + sum_{j:x=1}(1-z_j) >= 1.
        coefficients = np.where(best == 0, 1.0, -1.0)[None, :]
        lower = 1.0 - float(best.sum())
        try:
            second = self._solve(cost, [LinearConstraint(coefficients, lower, np.inf)])
        except SolverError:
            return 0.0
        return max(0.0, float(np.dot(cost, second - best)))


def make_decision_space(
    config: DecisionSpaceConfig,
    dimension: int,
    rng: np.random.Generator,
) -> DecisionSpace:
    if config.kind == DecisionSpaceKind.INDEPENDENT_BINARY:
        return IndependentBinaryDecisionSpace(dimension, config.max_enumeration)
    if config.kind == DecisionSpaceKind.FIXED_CARDINALITY:
        cardinality = config.cardinality or max(1, dimension // 5)
        return FixedCardinalityDecisionSpace(dimension, cardinality, config.max_enumeration)
    if config.kind == DecisionSpaceKind.CONTINUOUS_POLYTOPE:
        return ContinuousPolytopeDecisionSpace(
            dimension,
            A=config.A,
            b=config.b,
            lower=config.lower,
            upper=config.upper,
            max_enumeration=config.max_enumeration,
        )
    if config.edges is not None:
        edges = [(int(edge[0]), int(edge[1])) for edge in config.edges]
        source = config.source if config.source is not None else min(item for edge in edges for item in edge)
        sink = config.sink if config.sink is not None else max(item for edge in edges for item in edge)
        if len(edges) != dimension:
            raise ValidationError("the number of structured edges must equal dimension")
        return DAGPathDecisionSpace(edges, source, sink, max_enumeration=config.max_enumeration)
    if config.A_eq is not None or config.C_ub is not None:
        return StructuredBinaryDecisionSpace(
            dimension,
            A_eq=config.A_eq,
            b_eq=config.b_eq,
            C_ub=config.C_ub,
            r_ub=config.r_ub,
            max_enumeration=config.max_enumeration,
        )
    edges, source, sink = generate_dag_edges(dimension, rng)
    return DAGPathDecisionSpace(edges, source, sink, max_enumeration=config.max_enumeration)
