from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import product
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

import numpy as np

from ..exceptions import ValidationError


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ExpertKind(_StringEnum):
    MIN = "min"
    GIBBS = "gibbs"


class DecisionSpaceKind(_StringEnum):
    INDEPENDENT_BINARY = "independent_binary"
    FIXED_CARDINALITY = "fixed_cardinality"
    CONTINUOUS_POLYTOPE = "continuous_polytope"
    STRUCTURED = "structured"


class QuerySpaceKind(_StringEnum):
    BALANCED = "balanced"
    CLUSTERED = "clustered"
    SHARP_BOUNDARY = "sharp_boundary"
    LOW_RANK = "low_rank"
    RARE_INFORMATIVE = "rare_informative"
    ALIASED = "aliased"
    SPARSE = "sparse"
    DENSE = "dense"


class ObservationNoiseKind(_StringEnum):
    CLEAN = "clean"
    LOCAL = "local"
    OUTLIER = "outlier"
    BIASED = "biased"
    QUERY_DEPENDENT = "query_dependent"
    PARTIAL = "partial"


class ParameterNoiseKind(_StringEnum):
    NONE = "none"
    ISOTROPIC = "isotropic"
    ANISOTROPIC = "anisotropic"
    QUERY_DEPENDENT = "query_dependent"
    PERSISTENT = "persistent"


EnumT = TypeVar("EnumT", bound=Enum)


def coerce_enum(value: EnumT | str, enum_type: type[EnumT], name: str) -> EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValidationError(f"{name} must be one of: {allowed}") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(slots=True)
class ExpertConfig:
    kind: ExpertKind | str = ExpertKind.MIN
    temperature: str | float = "medium"
    reference_gap: float | None = None
    tie_breaking: str = "lexicographic"
    gibbs_burn_in: int = 40
    gibbs_steps: int = 20

    def __post_init__(self) -> None:
        self.kind = coerce_enum(self.kind, ExpertKind, "expert kind")
        if self.tie_breaking not in {"lexicographic", "random"}:
            raise ValidationError("tie_breaking must be lexicographic or random")
        if isinstance(self.temperature, str) and self.temperature not in {"low", "medium", "high"}:
            raise ValidationError("temperature must be low, medium, high, or a positive number")
        if not isinstance(self.temperature, str) and float(self.temperature) <= 0:
            raise ValidationError("temperature must be positive")
        if self.reference_gap is not None and self.reference_gap <= 0:
            raise ValidationError("reference_gap must be positive")
        if self.gibbs_burn_in < 0 or self.gibbs_steps < 1:
            raise ValidationError("Gibbs burn-in and step counts must be nonnegative/positive")

    @property
    def normalized_temperature(self) -> float:
        if isinstance(self.temperature, str):
            return {"low": 0.1, "medium": 0.5, "high": 2.0}[self.temperature]
        return float(self.temperature)


@dataclass(slots=True)
class DecisionSpaceConfig:
    kind: DecisionSpaceKind | str = DecisionSpaceKind.INDEPENDENT_BINARY
    cardinality: int | None = None
    lower: Sequence[float] | None = None
    upper: Sequence[float] | None = None
    A: Sequence[Sequence[float]] | None = None
    b: Sequence[float] | None = None
    A_eq: Sequence[Sequence[float]] | None = None
    b_eq: Sequence[float] | None = None
    C_ub: Sequence[Sequence[float]] | None = None
    r_ub: Sequence[float] | None = None
    edges: Sequence[Sequence[int]] | None = None
    source: int | None = None
    sink: int | None = None
    max_enumeration: int = 65_536
    mcmc_burn_in: int = 40
    mcmc_steps: int = 20

    def __post_init__(self) -> None:
        self.kind = coerce_enum(self.kind, DecisionSpaceKind, "decision-space kind")
        if self.cardinality is not None and self.cardinality < 1:
            raise ValidationError("cardinality must be positive")
        if self.max_enumeration < 2:
            raise ValidationError("max_enumeration must be at least two")
        if self.mcmc_burn_in < 0 or self.mcmc_steps < 1:
            raise ValidationError("MCMC burn-in and step counts must be nonnegative/positive")


@dataclass(slots=True)
class QuerySpaceConfig:
    kind: QuerySpaceKind | str = QuerySpaceKind.BALANCED
    candidate_count: int = 128
    clusters: int = 4
    cluster_radius: float = 0.12
    boundary_epsilon: float = 0.08
    rank: int | None = None
    informative_fraction: float = 0.08
    sparsity: int | None = None
    construction_attempts: int = 2_000
    allow_repeated_queries: bool = True

    def __post_init__(self) -> None:
        self.kind = coerce_enum(self.kind, QuerySpaceKind, "query-space kind")
        if self.candidate_count < 2:
            raise ValidationError("candidate_count must be at least two")
        if self.clusters < 1 or self.cluster_radius < 0 or self.boundary_epsilon <= 0:
            raise ValidationError("invalid clustered or boundary query parameters")
        if not 0 < self.informative_fraction < 1:
            raise ValidationError("informative_fraction must lie in (0, 1)")
        if self.rank is not None and self.rank < 1:
            raise ValidationError("rank must be positive")
        if self.sparsity is not None and self.sparsity < 1:
            raise ValidationError("sparsity must be positive")


@dataclass(slots=True)
class ObservationNoiseConfig:
    kind: ObservationNoiseKind | str = ObservationNoiseKind.CLEAN
    sigma: float = 0.1
    outlier_probability: float = 0.05
    bias: Sequence[float] | None = None
    confusion_matrix: Sequence[Sequence[float]] | None = None
    mask_probability: float = 0.25
    query_profile: str = "first_coordinate"
    minimum_scale: float = 0.02
    maximum_scale: float = 0.20
    distance: str = "euclidean"
    target_decision_change_rate: float | None = None
    calibration_trials: int = 96

    def __post_init__(self) -> None:
        self.kind = coerce_enum(self.kind, ObservationNoiseKind, "observation-noise kind")
        if self.sigma < 0:
            raise ValidationError("sigma must be nonnegative")
        if not 0 <= self.outlier_probability <= 1:
            raise ValidationError("outlier_probability must lie in [0, 1]")
        if not 0 <= self.mask_probability <= 1:
            raise ValidationError("mask_probability must lie in [0, 1]")
        if self.minimum_scale < 0 or self.maximum_scale < self.minimum_scale:
            raise ValidationError("query-dependent noise scales are invalid")
        if self.query_profile not in {"first_coordinate", "absolute_first", "sparsity", "norm"}:
            raise ValidationError("unsupported query noise profile")
        if self.distance not in {"euclidean", "hamming"}:
            raise ValidationError("distance must be euclidean or hamming")
        if (
            self.target_decision_change_rate is not None
            and not 0 < self.target_decision_change_rate < 1
        ):
            raise ValidationError("target_decision_change_rate must lie in (0, 1)")
        if self.calibration_trials < 16:
            raise ValidationError("calibration_trials must be at least 16")
        if (
            self.target_decision_change_rate is not None
            and self.kind not in {ObservationNoiseKind.LOCAL, ObservationNoiseKind.OUTLIER}
        ):
            raise ValidationError(
                "behavioral calibration is supported for local and outlier observation noise"
            )


@dataclass(slots=True)
class ParameterNoiseConfig:
    kind: ParameterNoiseKind | str = ParameterNoiseKind.NONE
    sigma: float = 0.1
    covariance: Sequence[Sequence[float]] | None = None
    query_profile: str = "first_coordinate"
    minimum_scale: float = 0.02
    maximum_scale: float = 0.20
    target_decision_change_rate: float | None = None
    calibration_trials: int = 96

    def __post_init__(self) -> None:
        self.kind = coerce_enum(self.kind, ParameterNoiseKind, "parameter-noise kind")
        if self.sigma < 0:
            raise ValidationError("sigma must be nonnegative")
        if self.minimum_scale < 0 or self.maximum_scale < self.minimum_scale:
            raise ValidationError("query-dependent parameter-noise scales are invalid")
        if self.query_profile not in {"first_coordinate", "absolute_first", "sparsity", "norm"}:
            raise ValidationError("unsupported query parameter-noise profile")
        if (
            self.target_decision_change_rate is not None
            and not 0 < self.target_decision_change_rate < 1
        ):
            raise ValidationError("target_decision_change_rate must lie in (0, 1)")
        if self.calibration_trials < 16:
            raise ValidationError("calibration_trials must be at least 16")
        if (
            self.target_decision_change_rate is not None
            and self.kind != ParameterNoiseKind.ISOTROPIC
        ):
            raise ValidationError(
                "behavioral calibration is supported for isotropic parameter noise"
            )


@dataclass(slots=True)
class ActiveScenarioConfig:
    name: str = "active-scenario"
    dimension: int = 5
    horizon: int = 50
    seed: int = 0
    true_theta: Sequence[float] | None = None
    normalize_true_theta: bool = True
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    decision_space: DecisionSpaceConfig = field(default_factory=DecisionSpaceConfig)
    query_space: QuerySpaceConfig = field(default_factory=QuerySpaceConfig)
    observation_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)
    parameter_noise: ParameterNoiseConfig = field(default_factory=ParameterNoiseConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.horizon < 1:
            raise ValidationError("dimension and horizon must be positive")
        if isinstance(self.expert, Mapping):
            self.expert = ExpertConfig(**self.expert)
        if isinstance(self.decision_space, Mapping):
            self.decision_space = DecisionSpaceConfig(**self.decision_space)
        if isinstance(self.query_space, Mapping):
            self.query_space = QuerySpaceConfig(**self.query_space)
        if isinstance(self.observation_noise, Mapping):
            self.observation_noise = ObservationNoiseConfig(**self.observation_noise)
        if isinstance(self.parameter_noise, Mapping):
            self.parameter_noise = ParameterNoiseConfig(**self.parameter_noise)
        if self.true_theta is not None:
            theta = np.asarray(self.true_theta, dtype=float).reshape(-1)
            if theta.size != self.dimension or not np.all(np.isfinite(theta)):
                raise ValidationError("true_theta must be finite and match dimension")
            if self.normalize_true_theta and np.linalg.norm(theta) <= 1e-15:
                raise ValidationError("true_theta cannot be zero when normalization is enabled")
        if self.decision_space.kind == DecisionSpaceKind.FIXED_CARDINALITY:
            cardinality = self.decision_space.cardinality or max(1, self.dimension // 5)
            if cardinality > self.dimension:
                raise ValidationError("cardinality cannot exceed dimension")
        if self.query_space.rank is not None and self.query_space.rank >= self.dimension:
            raise ValidationError("low-rank query dimension must be smaller than ambient dimension")
        if self.query_space.sparsity is not None and self.query_space.sparsity > self.dimension:
            raise ValidationError("query sparsity cannot exceed dimension")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class ActiveBenchmarkGrid:
    """Lazy Cartesian benchmark definition.

    Scalar component configs are reused; sequences form axes. This intentionally does
    not materialize the potentially very large benchmark until ``scenarios`` is iterated.
    """

    dimensions: Sequence[int] = (5, 20, 50)
    experts: Sequence[ExpertConfig] = (ExpertConfig(kind="min"),)
    decision_spaces: Sequence[DecisionSpaceConfig] = (
        DecisionSpaceConfig(kind="independent_binary"),
    )
    query_spaces: Sequence[QuerySpaceConfig] = (QuerySpaceConfig(kind="balanced"),)
    observation_noises: Sequence[ObservationNoiseConfig] = (
        ObservationNoiseConfig(kind="clean"),
    )
    parameter_noises: Sequence[ParameterNoiseConfig] = (
        ParameterNoiseConfig(kind="none"),
    )
    seeds: Sequence[int] = (0,)
    horizon: int = 50
    name_prefix: str = "active-benchmark"

    @property
    def size(self) -> int:
        lengths = (
            len(self.dimensions), len(self.experts), len(self.decision_spaces),
            len(self.query_spaces), len(self.observation_noises),
            len(self.parameter_noises), len(self.seeds),
        )
        return int(np.prod(lengths, dtype=np.int64))

    def scenarios(self, *, limit: int | None = None) -> Iterator[ActiveScenarioConfig]:
        axes: Iterable[tuple[Any, ...]] = product(
            self.dimensions,
            self.experts,
            self.decision_spaces,
            self.query_spaces,
            self.observation_noises,
            self.parameter_noises,
            self.seeds,
        )
        for index, values in enumerate(axes):
            if limit is not None and index >= limit:
                return
            dimension, expert, decision, query, observation, parameter, seed = values
            name = (
                f"{self.name_prefix}-d{dimension}-{expert.kind.value}-{decision.kind.value}-"
                f"{query.kind.value}-{observation.kind.value}-{parameter.kind.value}-s{seed}"
            )
            yield ActiveScenarioConfig(
                name=name,
                dimension=int(dimension),
                horizon=self.horizon,
                seed=int(seed),
                expert=expert,
                decision_space=decision,
                query_space=query,
                observation_noise=observation,
                parameter_noise=parameter,
            )
