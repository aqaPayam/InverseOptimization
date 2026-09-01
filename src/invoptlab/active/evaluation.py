from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..exceptions import ValidationError
from .config import _jsonable
from .decision_spaces import DecisionSpace, make_decision_space
from .types import ActiveBenchmarkResult, ActiveRunResult


Array = np.ndarray


@dataclass(slots=True)
class ActiveEvaluationConfig:
    test_query_count: int = 128
    seed: int = 0
    evaluate_trajectory: bool = False
    zero_regret_tolerance: float = 1e-8
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.test_query_count < 1:
            raise ValidationError("test_query_count must be positive")
        if self.zero_regret_tolerance < 0 or self.numerical_tolerance <= 0:
            raise ValidationError("evaluation tolerances must be nonnegative/positive")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class ActiveEvaluationResult:
    final_angular_error_degrees: float
    final_normalized_regret: float
    final_zero_regret_rate: float
    final_estimate_valid: bool
    test_query_count: int
    angular_error_history_degrees: list[float] = field(default_factory=list)
    normalized_regret_history: list[float] = field(default_factory=list)
    zero_regret_rate_history: list[float] = field(default_factory=list)
    valid_estimate_history: list[bool] = field(default_factory=list)
    mean_angular_error_degrees: float | None = None
    mean_normalized_regret: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def angular_error_degrees(
    theta_hat: Array,
    theta_true: Array,
    *,
    tolerance: float = 1e-12,
) -> tuple[float, bool]:
    estimate = np.asarray(theta_hat, dtype=float).reshape(-1)
    truth = np.asarray(theta_true, dtype=float).reshape(-1)
    if estimate.size != truth.size or not np.all(np.isfinite(estimate)):
        return 180.0, False
    estimate_norm = float(np.linalg.norm(estimate))
    truth_norm = float(np.linalg.norm(truth))
    if estimate_norm <= tolerance or truth_norm <= tolerance:
        return 180.0, False
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
) -> ActiveEvaluationResult:
    settings = config or ActiveEvaluationConfig()
    if run.error is not None:
        raise ValidationError("a failed active run cannot be evaluated")
    if not run.records:
        raise ValidationError("an active run must contain at least one estimate")
    test_queries = sample_uniform_test_queries(
        run.scenario.dimension,
        settings.test_query_count,
        scenario_seed=run.scenario.seed,
        evaluation_seed=settings.seed,
    )
    decision_space = _decision_space_for_evaluation(run)
    estimates = run.parameter_history
    angles: list[float] = []
    valid: list[bool] = []
    for estimate in estimates:
        angle, is_valid = angular_error_degrees(
            estimate,
            run.true_theta,
            tolerance=settings.numerical_tolerance,
        )
        angles.append(angle)
        valid.append(is_valid)

    evaluated_estimates = estimates if settings.evaluate_trajectory else estimates[-1:]
    regret_history: list[float] = []
    zero_regret_history: list[float] = []
    for estimate in evaluated_estimates:
        mean_regret, zero_rate, _ = normalized_test_regret(
            estimate,
            run.true_theta,
            test_queries,
            decision_space,
            tolerance=settings.numerical_tolerance,
            zero_regret_tolerance=settings.zero_regret_tolerance,
        )
        regret_history.append(mean_regret)
        zero_regret_history.append(zero_rate)

    result = ActiveEvaluationResult(
        final_angular_error_degrees=angles[-1],
        final_normalized_regret=regret_history[-1],
        final_zero_regret_rate=zero_regret_history[-1],
        final_estimate_valid=valid[-1],
        test_query_count=settings.test_query_count,
        angular_error_history_degrees=angles if settings.evaluate_trajectory else [],
        normalized_regret_history=regret_history if settings.evaluate_trajectory else [],
        zero_regret_rate_history=zero_regret_history if settings.evaluate_trajectory else [],
        valid_estimate_history=valid if settings.evaluate_trajectory else [],
        mean_angular_error_degrees=(
            float(np.mean(angles)) if settings.evaluate_trajectory else None
        ),
        mean_normalized_regret=(
            float(np.mean(regret_history)) if settings.evaluate_trajectory else None
        ),
        metadata={
            "test_query_distribution": "uniform-unit-sphere",
            "test_queries_hidden_from_algorithm": True,
            "clean_true_parameter_evaluation": True,
            "scenario_seed": run.scenario.seed,
            "evaluation_seed": settings.seed,
            "trajectory_evaluated": settings.evaluate_trajectory,
        },
    )
    run.evaluation = result.to_dict()
    run.metadata["evaluation_applied"] = True
    return result


def _algorithm_summary(runs: list[ActiveRunResult]) -> dict[str, Any]:
    evaluated = [run for run in runs if run.evaluation is not None and run.error is None]
    if not evaluated:
        return {"evaluated_runs": 0}
    angles = np.asarray(
        [run.evaluation["final_angular_error_degrees"] for run in evaluated],
        dtype=float,
    )
    regrets = np.asarray(
        [run.evaluation["final_normalized_regret"] for run in evaluated],
        dtype=float,
    )
    zero_rates = np.asarray(
        [run.evaluation["final_zero_regret_rate"] for run in evaluated],
        dtype=float,
    )
    valid = np.asarray(
        [run.evaluation["final_estimate_valid"] for run in evaluated],
        dtype=bool,
    )
    return {
        "evaluated_runs": len(evaluated),
        "mean_final_angular_error_degrees": float(np.mean(angles)),
        "median_final_angular_error_degrees": float(np.median(angles)),
        "mean_final_normalized_regret": float(np.mean(regrets)),
        "median_final_normalized_regret": float(np.median(regrets)),
        "mean_final_zero_regret_rate": float(np.mean(zero_rates)),
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
