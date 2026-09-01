from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..exceptions import ValidationError
from .config import QuerySpaceConfig, QuerySpaceKind
from .decision_spaces import DecisionSpace


Array = np.ndarray


def normalize_rows(values: Array) -> Array:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValidationError("query candidates must be a matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 1e-15):
        raise ValidationError("query candidates cannot contain zero vectors")
    return matrix / norms[:, None]


def random_unit(count: int, dimension: int, rng: np.random.Generator) -> Array:
    return normalize_rows(rng.normal(size=(count, dimension)))


def _orthonormal_basis(dimension: int, rank: int, rng: np.random.Generator) -> Array:
    matrix = rng.normal(size=(dimension, rank))
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return basis[:, :rank]


def _signature(decision: Array) -> tuple[Any, ...]:
    value = np.asarray(decision)
    if np.issubdtype(value.dtype, np.integer) or np.allclose(value, np.rint(value), atol=1e-8):
        return tuple(np.rint(value).astype(int).tolist())
    return tuple(np.round(value.astype(float), 6).tolist())


@dataclass(slots=True)
class QuerySpace:
    kind: QuerySpaceKind
    candidates: Array
    metadata: dict[str, Any] = field(default_factory=dict)
    allow_repeated_queries: bool = True

    def __post_init__(self) -> None:
        self.candidates = normalize_rows(self.candidates)
        self._validate_geometry()

    @property
    def dimension(self) -> int:
        return int(self.candidates.shape[1])

    @property
    def size(self) -> int:
        return int(self.candidates.shape[0])

    def _validate_geometry(self) -> None:
        if not np.allclose(np.linalg.norm(self.candidates, axis=1), 1.0, atol=1e-8):
            raise ValidationError("every query must have unit L2 norm")

    def index_of(self, query: Array, tolerance: float = 1e-7) -> int | None:
        value = np.asarray(query, dtype=float).reshape(-1)
        if value.size != self.dimension:
            return None
        distances = np.linalg.norm(self.candidates - value, axis=1)
        index = int(np.argmin(distances))
        return index if distances[index] <= tolerance else None

    def contains(self, query: Array, tolerance: float = 1e-7) -> bool:
        return self.index_of(query, tolerance) is not None

    def sample(self, rng: np.random.Generator) -> Array:
        return self.candidates[int(rng.integers(self.size))].copy()

    def public_metadata(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind.value, "size": self.size}
        for key, item in self.metadata.items():
            if isinstance(item, np.ndarray):
                value[key] = item.tolist()
            elif isinstance(item, (str, int, float, bool, list, tuple, dict)):
                value[key] = item
        return value


def balanced_queries(count: int, dimension: int, rng: np.random.Generator) -> Array:
    blocks = []
    while sum(block.shape[0] for block in blocks) < count:
        orthogonal, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        signs = rng.choice((-1.0, 1.0), size=dimension)
        blocks.append((orthogonal * signs[None, :]).T)
    return normalize_rows(np.vstack(blocks)[:count])


def clustered_queries(
    count: int,
    dimension: int,
    clusters: int,
    radius: float,
    rng: np.random.Generator,
) -> tuple[Array, dict[str, Any]]:
    centers = random_unit(clusters, dimension, rng)
    assignments = np.arange(count) % clusters
    rng.shuffle(assignments)
    perturbations = rng.normal(size=(count, dimension))
    perturbation_norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
    magnitudes = rng.uniform(0.0, radius, size=(count, 1))
    perturbations = perturbations / np.maximum(perturbation_norms, 1e-15) * magnitudes
    candidates = normalize_rows(centers[assignments] + perturbations)
    return candidates, {"centers": centers, "cluster_assignments": assignments}


def low_rank_queries(
    count: int,
    dimension: int,
    rank: int,
    rng: np.random.Generator,
) -> tuple[Array, dict[str, Any]]:
    if not 1 <= rank < dimension:
        raise ValidationError("low-rank query rank must lie between one and dimension minus one")
    basis = _orthonormal_basis(dimension, rank, rng)
    coordinates = rng.normal(size=(count, rank))
    candidates = normalize_rows(coordinates @ basis.T)
    return candidates, {"basis": basis, "rank": rank}


def sparse_queries(
    count: int,
    dimension: int,
    sparsity: int,
    rng: np.random.Generator,
) -> tuple[Array, dict[str, Any]]:
    if not 1 <= sparsity <= dimension:
        raise ValidationError("query sparsity must lie between one and dimension")
    candidates = np.zeros((count, dimension))
    supports = []
    for row in range(count):
        support = rng.choice(dimension, size=sparsity, replace=False)
        candidates[row, support] = rng.normal(size=sparsity)
        supports.append(support.tolist())
    return normalize_rows(candidates), {"sparsity": sparsity, "supports": supports}


def rare_informative_queries(
    count: int,
    dimension: int,
    rank: int,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[Array, dict[str, Any]]:
    informative_count = max(1, int(round(count * fraction)))
    common_count = count - informative_count
    basis = _orthonormal_basis(dimension, rank, rng)
    common = normalize_rows(rng.normal(size=(common_count, rank)) @ basis.T)
    # Informative queries include a direction orthogonal to the common subspace.
    raw = rng.normal(size=(informative_count, dimension))
    orthogonal = raw - (raw @ basis) @ basis.T
    norms = np.linalg.norm(orthogonal, axis=1)
    for index in np.flatnonzero(norms <= 1e-12):
        replacement = rng.normal(size=dimension)
        orthogonal[index] = replacement - basis @ (basis.T @ replacement)
    informative = normalize_rows(orthogonal + 0.25 * raw)
    candidates = np.vstack([common, informative])
    informative_mask = np.asarray([False] * common_count + [True] * informative_count)
    order = rng.permutation(count)
    return candidates[order], {
        "common_basis": basis,
        "rank": rank,
        "informative_fraction": informative_count / count,
        "informative_mask": informative_mask[order],
    }


def sharp_boundary_queries(
    count: int,
    dimension: int,
    epsilon: float,
    theta_true: Array,
    decision_space: DecisionSpace,
    rng: np.random.Generator,
    attempts: int,
) -> tuple[Array, dict[str, Any]]:
    candidates: list[Array] = []
    pair_ids: list[int] = []
    successful_pairs = 0
    # Find two queries with different decisions, then bisect their segment to
    # locate a behavioral boundary. This is geometry-agnostic and works for all
    # four decision-space families.
    for _ in range(attempts):
        if len(candidates) >= count:
            break
        left = random_unit(1, dimension, rng)[0]
        right = random_unit(1, dimension, rng)[0]
        x_left = decision_space.min_decision(left * theta_true, rng)
        x_right = decision_space.min_decision(right * theta_true, rng)
        if np.array_equal(x_left, x_right):
            continue
        for _ in range(30):
            midpoint = left + right
            midpoint_norm = np.linalg.norm(midpoint)
            if midpoint_norm <= 1e-12:
                midpoint = normalize_rows((left + 1e-3 * rng.normal(size=dimension))[None, :])[0]
            else:
                midpoint /= midpoint_norm
            x_midpoint = decision_space.min_decision(midpoint * theta_true, rng)
            if np.array_equal(x_midpoint, x_left):
                left = midpoint
            else:
                right = midpoint
                x_right = x_midpoint
            if np.linalg.norm(left - right) <= epsilon:
                break
        if np.linalg.norm(left - right) > epsilon + 1e-8:
            continue
        candidates.extend([left, right])
        pair_ids.extend([successful_pairs, successful_pairs])
        successful_pairs += 1
    if len(candidates) < count:
        fallback = random_unit(count - len(candidates), dimension, rng)
        candidates.extend(list(fallback))
        pair_ids.extend([-1] * fallback.shape[0])
    return np.vstack(candidates[:count]), {
        "boundary_epsilon": epsilon,
        "pair_ids": np.asarray(pair_ids[:count]),
        "successful_boundary_pairs": successful_pairs,
    }


def aliased_queries(
    count: int,
    dimension: int,
    theta_true: Array,
    decision_space: DecisionSpace,
    rng: np.random.Generator,
    attempts: int,
) -> tuple[Array, dict[str, Any]]:
    candidates: list[Array] = []
    pair_ids: list[int] = []
    successful_pairs = 0
    achieved_distances: list[float] = []
    target_distance = 0.5
    for _ in range(attempts):
        if len(candidates) >= count:
            break
        first = random_unit(1, dimension, rng)[0]
        first_signature = _signature(decision_space.min_decision(first * theta_true, rng))
        second = first.copy()
        best_distance = 0.0
        # Explore increasingly large perturbations while retaining behavior.
        # This avoids the exponentially small same-decision collision rate of
        # independent random draws in high-dimensional binary environments.
        for scale in np.linspace(0.15, 2.0, 40):
            proposal = normalize_rows((first + scale * rng.normal(size=dimension))[None, :])[0]
            proposal_signature = _signature(decision_space.min_decision(proposal * theta_true, rng))
            distance = float(np.linalg.norm(first - proposal))
            if proposal_signature == first_signature and distance > best_distance:
                second = proposal
                best_distance = distance
        if best_distance < target_distance:
            continue
        candidates.extend([first, second])
        pair_ids.extend([successful_pairs, successful_pairs])
        achieved_distances.append(best_distance)
        successful_pairs += 1
    if len(candidates) < count:
        fallback = random_unit(count - len(candidates), dimension, rng)
        candidates.extend(list(fallback))
        pair_ids.extend([-1] * fallback.shape[0])
    return np.vstack(candidates[:count]), {
        "pair_ids": np.asarray(pair_ids[:count]),
        "successful_aliased_pairs": successful_pairs,
        "target_pair_distance": target_distance,
        "achieved_pair_distances": np.asarray(achieved_distances),
    }


def make_query_space(
    config: QuerySpaceConfig,
    dimension: int,
    theta_true: Array,
    decision_space: DecisionSpace,
    rng: np.random.Generator,
) -> QuerySpace:
    count = config.candidate_count
    metadata: dict[str, Any] = {}
    if config.kind == QuerySpaceKind.BALANCED:
        candidates = balanced_queries(count, dimension, rng)
    elif config.kind == QuerySpaceKind.CLUSTERED:
        candidates, metadata = clustered_queries(
            count, dimension, min(config.clusters, count), config.cluster_radius, rng
        )
    elif config.kind == QuerySpaceKind.SHARP_BOUNDARY:
        candidates, metadata = sharp_boundary_queries(
            count, dimension, config.boundary_epsilon, theta_true,
            decision_space, rng, config.construction_attempts,
        )
    elif config.kind == QuerySpaceKind.LOW_RANK:
        rank = config.rank or max(1, min(dimension - 1, dimension // 3))
        candidates, metadata = low_rank_queries(count, dimension, rank, rng)
    elif config.kind == QuerySpaceKind.RARE_INFORMATIVE:
        rank = config.rank or max(1, min(dimension - 1, dimension // 3))
        candidates, metadata = rare_informative_queries(
            count, dimension, rank, config.informative_fraction, rng
        )
    elif config.kind == QuerySpaceKind.ALIASED:
        candidates, metadata = aliased_queries(
            count, dimension, theta_true, decision_space, rng, config.construction_attempts
        )
    elif config.kind == QuerySpaceKind.SPARSE:
        sparsity = config.sparsity or max(1, int(round(np.sqrt(dimension))))
        candidates, metadata = sparse_queries(count, dimension, sparsity, rng)
    elif config.kind == QuerySpaceKind.DENSE:
        candidates = random_unit(count, dimension, rng)
        metadata = {"sparsity": dimension}
    else:  # pragma: no cover - exhaustive enum guard
        raise ValidationError(f"unsupported query-space kind: {config.kind}")
    return QuerySpace(
        kind=config.kind,
        candidates=candidates,
        metadata=metadata,
        allow_repeated_queries=config.allow_repeated_queries,
    )
