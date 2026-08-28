from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .capabilities import Capability, require_capabilities
from .core import ForwardProblem, InverseDataset, LinearObjective, ParameterSpace


@dataclass(slots=True)
class ConstraintRecord:
    normal: np.ndarray
    observation_index: int
    alternative: Any
    context: Any
    source: str = "enumerated"


@dataclass
class ConsistencyConstraints:
    dimension: int
    records: list[ConstraintRecord] = field(default_factory=list)

    @property
    def matrix(self) -> np.ndarray:
        if not self.records:
            return np.empty((0, self.dimension))
        return np.vstack([record.normal for record in self.records])

    @property
    def normalized_matrix(self) -> np.ndarray:
        matrix = self.matrix
        if not matrix.size:
            return matrix
        norms = np.linalg.norm(matrix, axis=1)
        return matrix[norms > 1e-12] / norms[norms > 1e-12, None]

    def feasible(self, theta: np.ndarray, tolerance: float = 1e-8) -> bool:
        return bool(not self.records or np.max(self.matrix @ theta) <= tolerance)

    def violations(self, theta: np.ndarray) -> np.ndarray:
        if not self.records:
            return np.empty(0)
        return np.maximum(self.matrix @ theta, 0.0)

    def slacks(self, theta: np.ndarray, normalized: bool = True) -> np.ndarray:
        matrix = self.normalized_matrix if normalized else self.matrix
        return -(matrix @ theta)

    def prefix(self, observations: int) -> "ConsistencyConstraints":
        return ConsistencyConstraints(
            self.dimension,
            [record for record in self.records if record.observation_index < observations],
        )

    def deduplicated(self, decimals: int = 10) -> "ConsistencyConstraints":
        seen: set[tuple[float, ...]] = set()
        result = []
        for record in self.records:
            norm = np.linalg.norm(record.normal)
            if norm <= 1e-12:
                continue
            key = tuple(np.round(record.normal / norm, decimals=decimals))
            if key not in seen:
                seen.add(key)
                result.append(record)
        return ConsistencyConstraints(self.dimension, result)


def build_consistency_constraints(
    problem: ForwardProblem, dataset: InverseDataset, *, deduplicate: bool = True
) -> ConsistencyConstraints:
    require_capabilities(
        problem.capabilities,
        {Capability.LINEAR_IN_THETA, Capability.SUPPORTS_ENUMERATION},
        "Consistency geometry",
    )
    objective = problem.objective
    assert isinstance(objective, LinearObjective)
    records: list[ConstraintRecord] = []
    for observation_index, observation in enumerate(dataset):
        observed_features = objective.features(observation.context, observation.decision)
        for alternative in problem.oracle.enumerate(observation.context):
            normal = observed_features - objective.features(observation.context, alternative)
            if np.linalg.norm(normal) <= 1e-12:
                continue
            records.append(
                ConstraintRecord(normal, observation_index, alternative, observation.context)
            )
    result = ConsistencyConstraints(problem.parameter_space.dimension, records)
    return result.deduplicated() if deduplicate else result


def clip_polygon_halfspace(polygon: np.ndarray, normal: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    if polygon.size == 0:
        return polygon
    output: list[np.ndarray] = []
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        current_value = float(np.dot(normal, current))
        previous_value = float(np.dot(normal, previous))
        current_inside = current_value <= tolerance
        previous_inside = previous_value <= tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-15:
                fraction = previous_value / denominator
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
    return np.asarray(output, dtype=float) if output else np.empty((0, 2))


def parameter_boundary_2d(space: ParameterSpace, resolution: int = 720) -> np.ndarray:
    if space.dimension != 2:
        raise ValueError("A 2D boundary requires parameter dimension 2")
    if space.kind == "l2_ball":
        angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        return space.radius * np.column_stack([np.cos(angles), np.sin(angles)])
    if space.kind == "box":
        lo, hi = space.lower, space.upper
        return np.asarray([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    # A simplex in R2 is a line segment. Give it a tiny width for robust plotting.
    epsilon = space.radius * 1e-4
    return np.asarray([[0, space.radius], [epsilon, space.radius - epsilon], [space.radius, 0], [space.radius - epsilon, epsilon]])


def feasible_polygon_2d(space: ParameterSpace, constraints: ConsistencyConstraints) -> np.ndarray:
    polygon = parameter_boundary_2d(space)
    for normal in constraints.normalized_matrix:
        polygon = clip_polygon_halfspace(polygon, normal)
        if polygon.size == 0:
            break
    return polygon


def sample_feasible_region(
    space: ParameterSpace,
    constraints: ConsistencyConstraints,
    *,
    count: int = 10_000,
    seed: int = 0,
    boundary: bool = False,
) -> np.ndarray:
    samples = space.sample(count, seed=seed, boundary=boundary)
    if not constraints.records:
        return samples
    mask = np.all(constraints.matrix @ samples.T <= 1e-9, axis=0)
    return samples[mask]


def geometry_statistics(
    space: ParameterSpace,
    constraints: ConsistencyConstraints,
    *,
    theta: np.ndarray | None = None,
    samples: int = 20_000,
    seed: int = 0,
) -> dict[str, float]:
    candidates = space.sample(samples, seed=seed, boundary=True)
    if constraints.records:
        feasible_mask = np.all(constraints.matrix @ candidates.T <= 1e-9, axis=0)
    else:
        feasible_mask = np.ones(samples, dtype=bool)
    feasible = candidates[feasible_mask]
    result = {
        "constraint_count": float(len(constraints.records)),
        "feasible_direction_fraction": float(feasible_mask.mean()),
        "sampled_feasible_count": float(feasible.shape[0]),
    }
    if feasible.shape[0] > 1:
        normalized = feasible / np.maximum(np.linalg.norm(feasible, axis=1, keepdims=True), 1e-15)
        # Approximate diameter without forming a potentially huge pairwise matrix.
        anchors = normalized[np.linspace(0, normalized.shape[0] - 1, min(100, normalized.shape[0]), dtype=int)]
        dots = np.clip(anchors @ normalized.T, -1.0, 1.0)
        result["angular_diameter"] = float(np.max(np.arccos(dots)))
    else:
        result["angular_diameter"] = 0.0
    if theta is not None and constraints.records:
        slacks = constraints.slacks(theta)
        result.update(
            minimum_margin=float(np.min(slacks)),
            mean_margin=float(np.mean(slacks)),
            violation_rate=float(np.mean(slacks < -1e-9)),
            maximum_violation=float(max(0.0, -np.min(slacks))),
        )
    return result


@dataclass(slots=True)
class GeometrySnapshot:
    step: int
    constraints: ConsistencyConstraints
    theta: np.ndarray | None
    incenter: np.ndarray | None
    inradius: float | None
    statistics: dict[str, float]


def build_geometry_history(
    problem: ForwardProblem,
    dataset: InverseDataset,
    theta_history: Iterable[np.ndarray] | None = None,
    *,
    sample_count: int = 5_000,
    seed: int = 0,
) -> list[GeometrySnapshot]:
    all_constraints = build_consistency_constraints(problem, dataset, deduplicate=False)
    parameters = [] if theta_history is None else list(theta_history)
    snapshots: list[GeometrySnapshot] = []
    for step in range(1, len(dataset) + 1):
        constraints = all_constraints.prefix(step).deduplicated()
        theta = parameters[step - 1] if step <= len(parameters) else None
        snapshots.append(
            GeometrySnapshot(
                step=step,
                constraints=constraints,
                theta=theta,
                incenter=None,
                inradius=None,
                statistics=geometry_statistics(
                    problem.parameter_space,
                    constraints,
                    theta=theta,
                    samples=sample_count,
                    seed=seed + step,
                ),
            )
        )
    return snapshots
