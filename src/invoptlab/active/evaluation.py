from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from ..exceptions import ValidationError
from .config import ActiveScenarioConfig, _jsonable
from .decision_spaces import DecisionSpace, make_decision_space
from .query_spaces import make_query_space
from .types import ActiveBenchmarkResult, ActiveRunResult


Array = np.ndarray


@dataclass(slots=True)
class ActiveEvaluationConfig:
    test_query_count: int = 128
    seed: int = 0
    evaluate_trajectory: bool = False
    zero_regret_tolerance: float = 1e-8
    numerical_tolerance: float = 1e-12
    query_distribution: str = "uniform_unit_sphere"
    learning_regret_threshold: float = 0.01
    learning_angular_threshold_degrees: float = 5.0

    def __post_init__(self) -> None:
        if self.test_query_count < 1:
            raise ValidationError("test_query_count must be positive")
        if self.zero_regret_tolerance < 0 or self.numerical_tolerance <= 0:
            raise ValidationError("evaluation tolerances must be nonnegative/positive")
        if self.query_distribution not in {"uniform_unit_sphere", "scenario"}:
            raise ValidationError("query_distribution must be uniform_unit_sphere or scenario")
        if not 0 <= self.learning_regret_threshold <= 1:
            raise ValidationError("learning_regret_threshold must lie in [0, 1]")
        if not 0 <= self.learning_angular_threshold_degrees <= 180:
            raise ValidationError("learning angular threshold must lie in [0, 180]")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class ActiveEvaluationResult:
    final_angular_error_degrees: float | None
    final_normalized_regret: float | None
    final_zero_regret_rate: float | None
    final_estimate_valid: bool
    final_status: str
    final_failure_reason: str | None
    test_query_count: int
    angular_error_history_degrees: list[float | None] = field(default_factory=list)
    normalized_regret_history: list[float | None] = field(default_factory=list)
    zero_regret_rate_history: list[float | None] = field(default_factory=list)
    maximum_normalized_regret_history: list[float | None] = field(default_factory=list)
    valid_estimate_history: list[bool] = field(default_factory=list)
    estimate_status_history: list[str] = field(default_factory=list)
    mean_angular_error_degrees: float | None = None
    mean_normalized_regret: float | None = None
    first_threshold_step: int | None = None
    stable_threshold_step: int | None = None
    first_zero_regret_step: int | None = None
    stable_zero_regret_step: int | None = None
    first_angular_threshold_step: int | None = None
    stable_angular_threshold_step: int | None = None
    first_joint_threshold_step: int | None = None
    stable_joint_threshold_step: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def angular_error_degrees(
    theta_hat: Array,
    theta_true: Array,
    *,
    tolerance: float = 1e-12,
) -> tuple[float | None, bool]:
    estimate = np.asarray(theta_hat, dtype=float).reshape(-1)
    truth = np.asarray(theta_true, dtype=float).reshape(-1)
    if estimate.size != truth.size or not np.all(np.isfinite(estimate)):
        return None, False
    estimate_norm = float(np.linalg.norm(estimate))
    truth_norm = float(np.linalg.norm(truth))
    if estimate_norm <= tolerance or truth_norm <= tolerance:
        return None, False
    cosine = float(np.dot(estimate, truth) / (estimate_norm * truth_norm))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return angle, True


def sample_uniform_test_queries(
    dimension: int,
    count: int,
    *,
    scenario_seed: int,
    evaluation_seed: int,
) -> Array:
    if dimension < 1 or count < 1:
        raise ValidationError("test-query dimension and count must be positive")
    sequence = np.random.SeedSequence(
        [int(scenario_seed), int(evaluation_seed), 1_927_331]
    )
    rng = np.random.default_rng(sequence)
    queries = rng.normal(size=(count, dimension))
    norms = np.linalg.norm(queries, axis=1)
    while np.any(norms <= 1e-15):  # practically impossible, but keeps the contract exact
        invalid = norms <= 1e-15
        queries[invalid] = rng.normal(size=(int(invalid.sum()), dimension))
        norms = np.linalg.norm(queries, axis=1)
    return queries / norms[:, None]


def sample_scenario_hidden_queries(
    scenario: ActiveScenarioConfig,
    theta_true: Array,
    count: int,
    *,
    evaluation_seed: int,
    distribution: str,
    decision_space: DecisionSpace,
) -> Array:
    if distribution == "uniform_unit_sphere":
        return sample_uniform_test_queries(
            scenario.dimension,
            count,
            scenario_seed=scenario.seed,
            evaluation_seed=evaluation_seed,
        )
    sequence = np.random.SeedSequence(
        [int(scenario.seed), int(evaluation_seed), 6_431_909]
    )
    rng = np.random.default_rng(sequence)
    if scenario.query_space.kind.value == "explicit":
        # A finite scenario distribution means uniform over its configured rows.
        # Truly new held-out queries can instead be supplied to evaluate_active_run.
        values = np.asarray(scenario.query_space.candidates, dtype=float)
        values = values / np.linalg.norm(values, axis=1, keepdims=True)
        return values[rng.integers(len(values), size=count)].copy()
    query_config = replace(scenario.query_space, candidate_count=count)
    return make_query_space(
        query_config,
        scenario.dimension,
        theta_true,
        decision_space,
        rng,
    ).candidates


def _first_and_stable_step(
    values: list[float | None], threshold: float
) -> tuple[int | None, int | None]:
    first = next(
        (
            index + 1
            for index, value in enumerate(values)
            if value is not None and value <= threshold
        ),
        None,
    )
    stable = next(
        (
            index + 1
            for index in range(len(values))
            if all(value is not None and value <= threshold for value in values[index:])
        ),
        None,
    )
    return first, stable


def _estimate_status(
    run: ActiveRunResult,
    index: int,
    estimate: Array,
    tolerance: float,
) -> tuple[str, str | None]:
    return estimate_status(estimate, run.records[index].update_diagnostics, tolerance)


def estimate_status(estimate: Array, diagnostics: dict | None = None,
                    tolerance: float = 1e-12) -> tuple[str, str | None]:
    """Shared validity contract for evaluation AND benchmark stopping."""
    diagnostics = diagnostics or {}
    reported = diagnostics.get("estimate_status")
    reason = diagnostics.get("failure_reason")
    if reported is not None and reported != "valid":
        return str(reported), None if reason is None else str(reason)
    value = np.asarray(estimate, dtype=float).reshape(-1)
    if not np.all(np.isfinite(value)):
        return "invalid_estimate", "the estimate contains non-finite values"
    if np.linalg.norm(value) <= tolerance:
        return "invalid_estimate", "the estimate has zero norm"
    return "valid", None


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def _decision_space_for_evaluation(run: ActiveRunResult) -> DecisionSpace:
    streams = np.random.SeedSequence(run.scenario.seed).spawn(8)
    decision_rng = np.random.default_rng(streams[1])
    return make_decision_space(
        run.scenario.decision_space,
        run.scenario.dimension,
        decision_rng,
    )


def normalized_test_regret(
    theta_hat: Array,
    theta_true: Array,
    test_queries: Array,
    decision_space: DecisionSpace,
    *,
    tolerance: float = 1e-12,
    zero_regret_tolerance: float = 1e-8,
) -> tuple[float, float, Array]:
    estimate = np.asarray(theta_hat, dtype=float).reshape(-1)
    truth = np.asarray(theta_true, dtype=float).reshape(-1)
    queries = np.asarray(test_queries, dtype=float)
    if estimate.size != decision_space.dimension or truth.size != decision_space.dimension:
        raise ValidationError("evaluation parameter dimension mismatch")
    if queries.ndim != 2 or queries.shape[1] != decision_space.dimension:
        raise ValidationError("test query matrix has the wrong dimension")
    rng = np.random.default_rng(0)
    regrets = np.empty(queries.shape[0], dtype=float)
    for index, query in enumerate(queries):
        true_cost = query * truth
        estimated_cost = query * estimate
        true_decision = decision_space.min_decision(
            true_cost,
            rng,
            tie_breaking="lexicographic",
        )
        estimated_decision = decision_space.min_decision(
            estimated_cost,
            rng,
            tie_breaking="lexicographic",
        )
        worst_decision = decision_space.min_decision(
            -true_cost,
            rng,
            tie_breaking="lexicographic",
        )
        best_value = float(np.dot(true_cost, true_decision))
        estimated_value = float(np.dot(true_cost, estimated_decision))
        worst_value = float(np.dot(true_cost, worst_decision))
        regret = max(0.0, estimated_value - best_value)
        objective_range = max(0.0, worst_value - best_value)
        if objective_range <= tolerance:
            regrets[index] = 0.0
        else:
            regrets[index] = float(np.clip(regret / objective_range, 0.0, 1.0))
    return (
        float(np.mean(regrets)),
        float(np.mean(regrets <= zero_regret_tolerance)),
        regrets,
    )


def evaluate_active_run(
    run: ActiveRunResult,
    config: ActiveEvaluationConfig | None = None,
    *, test_queries: Array | None = None,
) -> ActiveEvaluationResult:
    settings = config or ActiveEvaluationConfig()
    if run.error is not None:
        raise ValidationError("a failed active run cannot be evaluated")
    if not run.records:
        raise ValidationError("an active run must contain at least one estimate")
    decision_space = _decision_space_for_evaluation(run)
    explicit_test = test_queries is not None
    if explicit_test:
        test_queries = np.asarray(test_queries, dtype=float).copy()
        if (test_queries.ndim != 2 or test_queries.shape[1] != run.scenario.dimension
                or not len(test_queries) or not np.all(np.isfinite(test_queries))
                or not np.allclose(np.linalg.norm(test_queries, axis=1), 1., atol=1e-8, rtol=0)):
            raise ValidationError("held-out queries must be a nonempty finite unit-norm (N, d) matrix")
    else:
        test_queries = sample_scenario_hidden_queries(
            run.scenario, run.true_theta, settings.test_query_count,
            evaluation_seed=settings.seed, distribution=settings.query_distribution,
            decision_space=decision_space,
        )
    estimates = run.parameter_history
    angles: list[float | None] = []
    valid: list[bool] = []
    statuses: list[str] = []
    failure_reasons: list[str | None] = []
    for index, estimate in enumerate(estimates):
        status, failure_reason = _estimate_status(
            run, index, estimate, settings.numerical_tolerance
        )
        if status == "valid":
            angle, is_valid = angular_error_degrees(
                estimate,
                run.true_theta,
                tolerance=settings.numerical_tolerance,
            )
        else:
            angle, is_valid = None, False
        angles.append(angle)
        valid.append(is_valid)
        statuses.append(status)
        failure_reasons.append(failure_reason)

    evaluated_indices = (
        list(range(len(estimates))) if settings.evaluate_trajectory else [len(estimates) - 1]
    )
    regret_history: list[float | None] = []
    zero_regret_history: list[float | None] = []
    maximum_regret_history: list[float | None] = []
    for index in evaluated_indices:
        if not valid[index]:
            regret_history.append(None)
            zero_regret_history.append(None)
            maximum_regret_history.append(None)
            continue
        estimate = estimates[index]
        mean_regret, zero_rate, regrets = normalized_test_regret(
            estimate,
            run.true_theta,
            test_queries,
            decision_space,
            tolerance=settings.numerical_tolerance,
            zero_regret_tolerance=settings.zero_regret_tolerance,
        )
        regret_history.append(mean_regret)
        zero_regret_history.append(zero_rate)
        maximum_regret_history.append(float(np.max(regrets)))

    first_threshold, stable_threshold = (None, None)
    first_zero, stable_zero = (None, None)
    first_angle, stable_angle = (None, None)
    first_joint, stable_joint = (None, None)
    if settings.evaluate_trajectory:
        first_threshold, stable_threshold = _first_and_stable_step(
            regret_history, settings.learning_regret_threshold
        )
        first_zero, stable_zero = _first_and_stable_step(
            maximum_regret_history, settings.zero_regret_tolerance
        )
        first_angle, stable_angle = _first_and_stable_step(
            angles, settings.learning_angular_threshold_degrees
        )
        # Zero marks a simultaneous hit; invalid estimates break stability.
        joint = [
            None if angle is None or regret is None else float(not (
                angle <= settings.learning_angular_threshold_degrees
                and regret <= settings.learning_regret_threshold
            ))
            for angle, regret in zip(angles, regret_history)
        ]
        first_joint, stable_joint = _first_and_stable_step(joint, 0.0)

    result = ActiveEvaluationResult(
        final_angular_error_degrees=angles[-1],
        final_normalized_regret=regret_history[-1],
        final_zero_regret_rate=zero_regret_history[-1],
        final_estimate_valid=valid[-1],
        final_status=statuses[-1],
        final_failure_reason=failure_reasons[-1],
        test_query_count=len(test_queries),
        angular_error_history_degrees=angles if settings.evaluate_trajectory else [],
        normalized_regret_history=regret_history if settings.evaluate_trajectory else [],
        zero_regret_rate_history=zero_regret_history if settings.evaluate_trajectory else [],
        maximum_normalized_regret_history=(
            maximum_regret_history if settings.evaluate_trajectory else []
        ),
        valid_estimate_history=valid if settings.evaluate_trajectory else [],
        estimate_status_history=statuses if settings.evaluate_trajectory else [],
        mean_angular_error_degrees=(
            _mean_or_none(angles) if settings.evaluate_trajectory else None
        ),
        mean_normalized_regret=(
            _mean_or_none(regret_history) if settings.evaluate_trajectory else None
        ),
        first_threshold_step=first_threshold,
        stable_threshold_step=stable_threshold,
        first_zero_regret_step=first_zero,
        stable_zero_regret_step=stable_zero,
        first_angular_threshold_step=first_angle,
        stable_angular_threshold_step=stable_angle,
        first_joint_threshold_step=first_joint,
        stable_joint_threshold_step=stable_joint,
        metadata={
            "test_query_distribution": "explicit-heldout" if explicit_test else settings.query_distribution,
            "explicit_test_queries": test_queries.tolist() if explicit_test else None,
            "test_queries_hidden_from_algorithm": True,
            "clean_true_parameter_evaluation": True,
            "scenario_seed": run.scenario.seed,
            "evaluation_seed": settings.seed,
            "trajectory_evaluated": settings.evaluate_trajectory,
            "learning_regret_threshold": settings.learning_regret_threshold,
            "learning_angular_threshold_degrees": settings.learning_angular_threshold_degrees,
            "stable_threshold_definition": "at or below threshold at every remaining step through T",
            "independent_from_stopping": (
                None if explicit_test and run.metadata.get("external_stopping_enabled", False)
                else not run.metadata.get("external_stopping_enabled", False)
                or settings.seed != run.metadata.get("stopping_rule", {}).get("seed")
                or settings.query_distribution
                != run.metadata.get("stopping_rule", {}).get("query_distribution")
            ),
        },
    )
    run.evaluation = result.to_dict()
    run.metadata["evaluation_applied"] = True
    run.metadata["algorithm_failed"] = not result.final_estimate_valid
    run.metadata["final_estimate_status"] = result.final_status
    run.metadata["algorithm_failure_reason"] = result.final_failure_reason
    return result


def _algorithm_summary(runs: list[ActiveRunResult]) -> dict[str, Any]:
    evaluated = [run for run in runs if run.evaluation is not None and run.error is None]
    if not evaluated:
        return {"evaluated_runs": 0}
    valid = np.asarray(
        [run.evaluation["final_estimate_valid"] for run in evaluated],
        dtype=bool,
    )
    valid_runs = [run for run, is_valid in zip(evaluated, valid) if is_valid]
    angles = np.asarray(
        [run.evaluation["final_angular_error_degrees"] for run in valid_runs],
        dtype=float,
    )
    regrets = np.asarray(
        [run.evaluation["final_normalized_regret"] for run in valid_runs],
        dtype=float,
    )
    zero_rates = np.asarray(
        [run.evaluation["final_zero_regret_rate"] for run in valid_runs],
        dtype=float,
    )

    def mean(value: Array) -> float | None:
        return float(np.mean(value)) if value.size else None

    def median(value: Array) -> float | None:
        return float(np.median(value)) if value.size else None

    return {
        "evaluated_runs": len(evaluated),
        "successful_estimate_runs": len(valid_runs),
        "failed_estimate_runs": len(evaluated) - len(valid_runs),
        "algorithm_failure_rate": float(1.0 - np.mean(valid)),
        "mean_final_angular_error_degrees": mean(angles),
        "median_final_angular_error_degrees": median(angles),
        "mean_final_normalized_regret": mean(regrets),
        "median_final_normalized_regret": median(regrets),
        "mean_final_zero_regret_rate": mean(zero_rates),
        "valid_final_estimate_rate": float(np.mean(valid)),
        "mean_runtime_seconds": float(np.mean([run.runtime_seconds for run in evaluated])),
    }


def evaluate_active_benchmark(
    benchmark: ActiveBenchmarkResult,
    config: ActiveEvaluationConfig | None = None,
) -> dict[str, Any]:
    settings = config or ActiveEvaluationConfig()
    by_algorithm: dict[str, list[ActiveRunResult]] = {}
    for run in benchmark.successful_runs:
        evaluate_active_run(run, settings)
        by_algorithm.setdefault(run.algorithm_name, []).append(run)
    summary = {
        "config": settings.to_dict(),
        "algorithms": {
            name: _algorithm_summary(runs) for name, runs in by_algorithm.items()
        },
        "evaluated_run_count": sum(len(runs) for runs in by_algorithm.values()),
        "failed_run_count": len(benchmark.failed_runs),
    }
    benchmark.metadata["evaluation_applied"] = True
    benchmark.metadata["evaluation"] = summary
    return _jsonable(summary)
