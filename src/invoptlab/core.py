from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

import numpy as np

from .capabilities import Capability
from .exceptions import SolverError, ValidationError

Array = np.ndarray


def as_array(value: Any, *, name: str = "value") -> Array:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be numeric and array-like") from exc
    if not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} contains NaN or infinite values")
    return result


@dataclass(slots=True)
class Observation:
    context: Any
    decision: Any
    clean_decision: Any | None = None
    true_theta: Array | None = None
    timestamp: float | int | None = None
    weight: float = 1.0
    expert_id: str | None = None
    noise: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weight < 0 or not np.isfinite(self.weight):
            raise ValidationError("Observation weight must be finite and nonnegative")
        if self.true_theta is not None:
            self.true_theta = as_array(self.true_theta, name="true_theta").reshape(-1)


@dataclass
class InverseDataset:
    observations: list[Observation]
    name: str = "dataset"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValidationError("A dataset must contain at least one observation")

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self.observations)

    def __getitem__(self, index: int | slice) -> Observation | "InverseDataset":
        if isinstance(index, slice):
            return InverseDataset(self.observations[index], self.name, dict(self.metadata))
        return self.observations[index]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]], name: str = "dataset") -> "InverseDataset":
        return cls([Observation(**dict(record)) for record in records], name=name)

    def chronological(self) -> "InverseDataset":
        if all(obs.timestamp is None for obs in self.observations):
            return InverseDataset(list(self.observations), self.name, dict(self.metadata))
        indexed = list(enumerate(self.observations))
        indexed.sort(key=lambda pair: (float("inf") if pair[1].timestamp is None else pair[1].timestamp, pair[0]))
        return InverseDataset([obs for _, obs in indexed], self.name, dict(self.metadata))

    def split(
        self, train: float = 0.7, validation: float = 0.15, *, seed: int = 0, shuffle: bool = True
    ) -> tuple["InverseDataset", "InverseDataset", "InverseDataset"]:
        if train <= 0 or validation < 0 or train + validation >= 1:
            raise ValidationError("Require train > 0, validation >= 0, and train + validation < 1")
        indices = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        n_train = max(1, int(round(train * len(self))))
        n_val = int(round(validation * len(self)))
        n_train = min(n_train, len(self) - n_val - 1)

        def subset(values: Array, suffix: str) -> "InverseDataset":
            return InverseDataset([self.observations[int(i)] for i in values], f"{self.name}-{suffix}")

        return (
            subset(indices[:n_train], "train"),
            subset(indices[n_train : n_train + n_val], "validation"),
            subset(indices[n_train + n_val :], "test"),
        )

    @property
    def fingerprint(self) -> str:
        payload = []
        for obs in self.observations:
            payload.append(
                repr((obs.context, obs.decision, obs.clean_decision, obs.timestamp, obs.weight, obs.expert_id))
            )
        return sha256("\n".join(payload).encode("utf-8")).hexdigest()


class Objective(Protocol):
    parameter_dimension: int

    def value(self, theta: Array, context: Any, decision: Any) -> float: ...


@dataclass(slots=True)
class CallableObjective:
    function: Callable[[Array, Any, Any], float]
    parameter_dimension: int
    gradient: Callable[[Array, Any, Any], Array] | None = None

    def value(self, theta: Array, context: Any, decision: Any) -> float:
        return float(self.function(theta, context, decision))


@dataclass(slots=True)
class LinearObjective:
    feature_map: Callable[[Any, Any], Any]
    parameter_dimension: int

    def features(self, context: Any, decision: Any) -> Array:
        phi = as_array(self.feature_map(context, decision), name="feature vector").reshape(-1)
        if phi.size != self.parameter_dimension:
            raise ValidationError(
                f"Expected {self.parameter_dimension} features, received {phi.size}"
            )
        # A feature map may return a view into the user's context array. Loss
        # gradients perform in-place arithmetic, so return an owned copy and
        # guarantee that fitting cannot mutate the dataset.
        return phi.copy()

    def value(self, theta: Array, context: Any, decision: Any) -> float:
        return float(np.dot(as_array(theta).reshape(-1), self.features(context, decision)))

    def parameter_gradient(self, context: Any, decision: Any) -> Array:
        return self.features(context, decision)


@dataclass(slots=True)
class ParameterSpace:
    dimension: int
    kind: str = "l2_ball"
    lower: Array | None = None
    upper: Array | None = None
    radius: float = 1.0

    def __post_init__(self) -> None:
        supported = {"l2_ball", "simplex", "box"}
        if self.dimension < 1 or self.kind not in supported or self.radius <= 0:
            raise ValidationError(f"Invalid parameter space; supported kinds are {sorted(supported)}")
        if self.kind == "box":
            self.lower = np.full(self.dimension, -1.0) if self.lower is None else as_array(self.lower).reshape(-1)
            self.upper = np.full(self.dimension, 1.0) if self.upper is None else as_array(self.upper).reshape(-1)
            if self.lower.size != self.dimension or self.upper.size != self.dimension:
                raise ValidationError("Box bounds must match parameter dimension")
            if np.any(self.lower >= self.upper):
                raise ValidationError("Each lower bound must be below its upper bound")

    def contains(self, theta: Any, tolerance: float = 1e-8) -> bool:
        value = as_array(theta).reshape(-1)
        if value.size != self.dimension:
            return False
        if self.kind == "l2_ball":
            return bool(np.linalg.norm(value) <= self.radius + tolerance)
        if self.kind == "simplex":
            return bool(np.all(value >= -tolerance) and abs(value.sum() - self.radius) <= tolerance)
        return bool(np.all(value >= self.lower - tolerance) and np.all(value <= self.upper + tolerance))

    def project(self, theta: Any) -> Array:
        value = as_array(theta).reshape(-1)
        if value.size != self.dimension:
            raise ValidationError("Parameter dimension mismatch")
        if self.kind == "l2_ball":
            norm = np.linalg.norm(value)
            return value if norm <= self.radius else value * (self.radius / norm)
        if self.kind == "box":
            return np.clip(value, self.lower, self.upper)
        # Euclidean projection onto {x >= 0, sum(x) = radius}.
        sorted_value = np.sort(value)[::-1]
        cumulative = np.cumsum(sorted_value) - self.radius
        indices = np.arange(1, self.dimension + 1)
        active = np.nonzero(sorted_value - cumulative / indices > 0)[0]
        threshold = cumulative[active[-1]] / float(active[-1] + 1)
        return np.maximum(value - threshold, 0.0)

    def center(self) -> Array:
        if self.kind == "simplex":
            return np.full(self.dimension, self.radius / self.dimension)
        if self.kind == "box":
            return (self.lower + self.upper) / 2.0
        return np.zeros(self.dimension)

    def sample(self, count: int, *, seed: int = 0, boundary: bool = False) -> Array:
        rng = np.random.default_rng(seed)
        if self.kind == "simplex":
            return rng.dirichlet(np.ones(self.dimension), size=count) * self.radius
        if self.kind == "box":
            return rng.uniform(self.lower, self.upper, size=(count, self.dimension))
        directions = rng.normal(size=(count, self.dimension))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-15)
        if boundary:
            return self.radius * directions
        radii = rng.random(count) ** (1.0 / self.dimension)
        return self.radius * directions * radii[:, None]


@dataclass(slots=True)
class ForwardSolution:
    decision: Any
    value: float
    status: str = "optimal"
    metadata: dict[str, Any] = field(default_factory=dict)


class DecisionOracle(Protocol):
    capabilities: set[Capability]

    def solve(self, objective: Objective, theta: Array, context: Any) -> ForwardSolution: ...

    def enumerate(self, context: Any) -> Sequence[Any]: ...


@dataclass
class EnumerationOracle:
    feasible_decisions: Callable[[Any], Sequence[Any]]
    tie_tolerance: float = 1e-10
    capabilities: set[Capability] = field(
        default_factory=lambda: {
            Capability.FINITE_FEASIBLE_SET,
            Capability.SUPPORTS_ENUMERATION,
            Capability.SUPPORTS_SEPARATION,
        }
    )

    def enumerate(self, context: Any) -> Sequence[Any]:
        decisions = list(self.feasible_decisions(context))
        if not decisions:
            raise SolverError("The feasible-decision oracle returned an empty set")
        return decisions

    def solve(self, objective: Objective, theta: Array, context: Any) -> ForwardSolution:
        decisions = self.enumerate(context)
        values = np.asarray([objective.value(theta, context, decision) for decision in decisions])
        index = int(np.argmin(values))
        optimum = float(values[index])
        ties = [i for i, value in enumerate(values) if value <= optimum + self.tie_tolerance]
        return ForwardSolution(decisions[index], optimum, metadata={"tie_count": len(ties), "ties": ties})

    def loss_augmented_solve(
        self,
        objective: Objective,
        theta: Array,
        context: Any,
        observed_decision: Any,
        distance: Callable[[Any, Any], float],
    ) -> ForwardSolution:
        decisions = self.enumerate(context)
        values = np.asarray(
            [objective.value(theta, context, decision) - distance(observed_decision, decision) for decision in decisions]
        )
        index = int(np.argmin(values))
        return ForwardSolution(decisions[index], float(values[index]))


@dataclass
class ForwardProblem:
    objective: Objective
    parameter_space: ParameterSpace
    oracle: DecisionOracle
    name: str = "forward-problem"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.objective.parameter_dimension != self.parameter_space.dimension:
            raise ValidationError("Objective and parameter-space dimensions do not match")

    @property
    def capabilities(self) -> set[Capability]:
        caps = set(self.oracle.capabilities)
        if isinstance(self.objective, LinearObjective):
            caps.add(Capability.LINEAR_IN_THETA)
            caps.add(Capability.DIFFERENTIABLE_IN_THETA)
        elif isinstance(self.objective, CallableObjective) and self.objective.gradient is not None:
            caps.add(Capability.DIFFERENTIABLE_IN_THETA)
        return caps

    def solve(self, theta: Any, context: Any) -> ForwardSolution:
        parameter = as_array(theta, name="theta").reshape(-1)
        if parameter.size != self.parameter_space.dimension:
            raise ValidationError("Parameter dimension mismatch")
        return self.oracle.solve(self.objective, parameter, context)

    def validate_dataset(self, dataset: InverseDataset, *, check_feasibility: bool = True) -> list[str]:
        warnings: list[str] = []
        for index, obs in enumerate(dataset):
            if check_feasibility and Capability.SUPPORTS_ENUMERATION in self.capabilities:
                alternatives = self.oracle.enumerate(obs.context)
                if not any(np.array_equal(np.asarray(obs.decision), np.asarray(x)) for x in alternatives):
                    raise ValidationError(f"Observation {index} is not feasible for its context")
            if obs.true_theta is not None and obs.true_theta.size != self.parameter_space.dimension:
                raise ValidationError(f"Observation {index} has a mismatched true_theta dimension")
        if any(obs.timestamp is None for obs in dataset) and any(obs.timestamp is not None for obs in dataset):
            warnings.append("Only part of the dataset has timestamps")
        return warnings


@dataclass
class StepRecord:
    step: int
    theta: Array
    loss: float | None = None
    radius: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EstimatorHistory:
    steps: list[StepRecord] = field(default_factory=list)

    def append(self, record: StepRecord) -> None:
        self.steps.append(record)

    @property
    def parameters(self) -> Array:
        if not self.steps:
            return np.empty((0, 0))
        return np.vstack([step.theta for step in self.steps])
