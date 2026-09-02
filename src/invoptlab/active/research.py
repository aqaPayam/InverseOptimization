from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import ValidationError
from .algorithms import ActiveAlgorithm
from .config import (
    ActiveScenarioConfig,
    DecisionSpaceConfig,
    ExpertConfig,
    ObservationNoiseConfig,
    ParameterNoiseConfig,
    QuerySpaceConfig,
    _jsonable,
)
from .evaluation import ActiveEvaluationConfig, evaluate_active_run
from .runner import ActiveBenchmarkSuite, AlgorithmFactory
from .stopping import RegretStoppingConfig
from .types import ActiveBenchmarkResult, ActiveRunResult


@dataclass(slots=True)
class ActiveResearchConfig:
    """Protocol for the compact, scientifically challenging benchmark."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    horizon: int = 40
    candidate_count: int = 48
    validation_query_count: int = 64
    test_query_count: int = 128
    validation_seed: int = 40_001
    test_seed: int = 90_001
    consecutive_validation_successes: int = 3
    zero_regret_tolerance: float = 1e-8
    learning_regret_threshold: float = 0.01

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValidationError("research protocol requires at least one seed")
        if self.horizon < 1 or self.candidate_count < 8:
            raise ValidationError("research horizon/candidate count is too small")
        if self.validation_query_count < 1 or self.test_query_count < 1:
            raise ValidationError("validation and test query counts must be positive")
        if self.validation_seed == self.test_seed:
            raise ValidationError("validation and final test seeds must be different")
        if self.consecutive_validation_successes < 1:
            raise ValidationError("consecutive validation successes must be positive")
        if not 0 <= self.learning_regret_threshold <= 1:
            raise ValidationError("learning regret threshold must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _theta(dimension: int) -> list[float]:
    # Close magnitudes intentionally produce competitive combinations, while signs
    # remain mixed so the easy independent-binary sanity case is still meaningful.
    negative = np.linspace(-1.0, -0.9, dimension // 2, endpoint=True)
    positive = np.linspace(0.9, 1.0, dimension - dimension // 2, endpoint=True)
    return np.concatenate([negative, positive]).tolist()


def _scenario(
    family: str,
    seed: int,
    protocol: ActiveResearchConfig,
    *,
    dimension: int = 10,
    decision_space: DecisionSpaceConfig | None = None,
    query_space: QuerySpaceConfig | None = None,
    expert: ExpertConfig | None = None,
    observation_noise: ObservationNoiseConfig | None = None,
    parameter_noise: ParameterNoiseConfig | None = None,
    stochastic: bool = False,
    difficulty: str = "hard",
    rationale: str,
) -> ActiveScenarioConfig:
    return ActiveScenarioConfig(
        name=f"research-{family}-s{seed}",
        dimension=dimension,
        horizon=protocol.horizon,
        seed=seed,
        true_theta=_theta(dimension),
        expert=expert or ExpertConfig(kind="min"),
        decision_space=decision_space or DecisionSpaceConfig(kind="fixed_cardinality", cardinality=dimension // 2),
        query_space=query_space or QuerySpaceConfig(
            kind="balanced", candidate_count=protocol.candidate_count
        ),
        observation_noise=observation_noise or ObservationNoiseConfig(kind="clean"),
        parameter_noise=parameter_noise or ParameterNoiseConfig(kind="none"),
        metadata={
            "research_family": family,
            "difficulty": difficulty,
            "stochastic": stochastic,
            "rationale": rationale,
            "objective_family": "linear-in-parameter",
            "role": "sanity-check" if difficulty == "sanity" else "research",
        },
    )


def build_active_research_scenarios(
    config: ActiveResearchConfig | None = None,
) -> list[ActiveScenarioConfig]:
    """Return 12 intentional families times the requested random seeds.

    This is not a Cartesian grid. Each family isolates a meaningful source of
    difficulty so failures remain scientifically interpretable.
    """

    protocol = config or ActiveResearchConfig()
    scenarios: list[ActiveScenarioConfig] = []
    dag_edges = [
        (0, 1), (0, 2),
        (1, 3), (1, 4), (2, 3), (2, 4),
        (3, 5), (4, 5),
    ]
    knapsack_weights = [2, 3, 4, 5, 1, 2, 4, 3, 5, 1]
    simplex_A = [np.ones(8).tolist(), (-np.ones(8)).tolist()]
    for seed in protocol.seeds:
        scenarios.extend(
            [
                _scenario(
                    "easy-independent",
                    seed,
                    protocol,
                    decision_space=DecisionSpaceConfig(kind="independent_binary"),
                    query_space=QuerySpaceConfig(
                        kind="dense", candidate_count=protocol.candidate_count
                    ),
                    difficulty="sanity",
                    rationale="Separable easy control; verifies implementation only.",
                ),
                _scenario(
                    "cardinality-coupled",
                    seed,
                    protocol,
                    rationale="Selecting exactly half the items reveals rankings, not coordinate signs.",
                ),
                _scenario(
                    "knapsack-coupled",
                    seed,
                    protocol,
                    decision_space=DecisionSpaceConfig(
                        kind="structured",
                        C_ub=[knapsack_weights],
                        r_ub=[15],
                        max_enumeration=65_536,
                    ),
                    rationale="A resource budget couples items with unequal weights.",
                ),
                _scenario(
                    "dag-path-coupled",
                    seed,
                    protocol,
                    dimension=len(dag_edges),
                    decision_space=DecisionSpaceConfig(
                        kind="structured", edges=dag_edges, source=0, sink=5
                    ),
                    rationale="Each observation compares complete source-to-sink paths.",
                ),
                _scenario(
                    "continuous-simplex",
                    seed,
                    protocol,
                    dimension=8,
                    decision_space=DecisionSpaceConfig(
                        kind="continuous_polytope",
                        lower=[0.0] * 8,
                        upper=[1.0] * 8,
                        A=simplex_A,
                        b=[1.0, -1.0],
                    ),
                    rationale="A simplex equality creates a genuinely coupled continuous decision.",
                ),
                _scenario(
                    "sparse-queries",
                    seed,
                    protocol,
                    query_space=QuerySpaceConfig(
                        kind="sparse",
                        sparsity=2,
                        candidate_count=protocol.candidate_count,
                    ),
                    rationale="Each interaction probes only two parameter coordinates.",
                ),
                _scenario(
                    "rare-informative",
                    seed,
                    protocol,
                    query_space=QuerySpaceConfig(
                        kind="rare_informative",
                        rank=2,
                        informative_fraction=0.1,
                        candidate_count=protocol.candidate_count,
                    ),
                    rationale="Most allowed queries lie in a low-dimensional subspace.",
                ),
                _scenario(
                    "small-margin-boundary",
                    seed,
                    protocol,
                    query_space=QuerySpaceConfig(
                        kind="sharp_boundary",
                        boundary_epsilon=0.025,
                        candidate_count=protocol.candidate_count,
                    ),
                    rationale="Queries are paired across nearby decision boundaries.",
                ),
                _scenario(
                    "partial-feedback",
                    seed,
                    protocol,
                    observation_noise=ObservationNoiseConfig(
                        kind="partial", mask_probability=0.5
                    ),
                    stochastic=True,
                    rationale="Half of the response coordinates are hidden on average.",
                ),
                _scenario(
                    "behavioral-observation-noise",
                    seed,
                    protocol,
                    observation_noise=ObservationNoiseConfig(
                        kind="local",
                        target_decision_change_rate=0.15,
                        calibration_trials=64,
                    ),
                    stochastic=True,
                    rationale="Noise strength is calibrated to alter about 15% of decisions.",
                ),
                _scenario(
                    "behavioral-parameter-noise",
                    seed,
                    protocol,
                    parameter_noise=ParameterNoiseConfig(
                        kind="isotropic",
                        target_decision_change_rate=0.15,
                        calibration_trials=64,
                    ),
                    stochastic=True,
                    rationale="Parameter perturbations alter about 15% of expert decisions.",
                ),
                _scenario(
                    "gibbs-expert",
                    seed,
                    protocol,
                    expert=ExpertConfig(kind="gibbs", temperature="medium"),
                    stochastic=True,
                    rationale="The expert samples plausible suboptimal coupled decisions.",
                ),
            ]
        )
    return scenarios


def _mean_std(values: Sequence[float | None]) -> dict[str, float | int | None]:
    array = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )
    if not array.size:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


def summarize_active_research(result: ActiveBenchmarkResult) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[ActiveRunResult]] = {}
    for run in result.successful_runs:
        family = str(run.scenario.metadata.get("research_family", run.scenario.name))
        grouped.setdefault((run.algorithm_name, family), []).append(run)

    groups: list[dict[str, Any]] = []
    for (algorithm, family), runs in sorted(grouped.items()):
        evaluated = [run for run in runs if run.evaluation is not None]
        threshold_steps = [
            run.evaluation["first_threshold_step"]
            for run in evaluated
            if run.evaluation.get("first_threshold_step") is not None
        ]
        stable_steps = [
            run.evaluation["stable_threshold_step"]
            for run in evaluated
            if run.evaluation.get("stable_threshold_step") is not None
        ]
        valid_evaluated = [
            run for run in evaluated if run.evaluation.get("final_estimate_valid", False)
        ]
        failed_statuses: dict[str, int] = {}
        for run in evaluated:
            status = str(run.evaluation.get("final_status", "unknown"))
            if status != "valid":
                failed_statuses[status] = failed_statuses.get(status, 0) + 1
        groups.append(
            {
                "algorithm": algorithm,
                "family": family,
                "run_count": len(runs),
                "stochastic": bool(runs[0].scenario.metadata.get("stochastic", False)),
                "valid_estimate_rate": (
                    len(valid_evaluated) / len(evaluated) if evaluated else None
                ),
                "algorithm_failure_rate": (
                    1.0 - len(valid_evaluated) / len(evaluated) if evaluated else None
                ),
                "failure_statuses": failed_statuses,
                "final_normalized_regret": _mean_std(
                    [run.evaluation["final_normalized_regret"] for run in valid_evaluated]
                ),
                "final_angular_error_degrees": _mean_std(
                    [run.evaluation["final_angular_error_degrees"] for run in valid_evaluated]
                ),
                "final_zero_regret_rate": _mean_std(
                    [run.evaluation["final_zero_regret_rate"] for run in valid_evaluated]
                ),
                "runtime_seconds": _mean_std([run.runtime_seconds for run in runs]),
                "threshold_reached_rate": (
                    len(threshold_steps) / len(evaluated) if evaluated else None
                ),
                "first_threshold_step": _mean_std(threshold_steps),
                "stable_threshold_reached_rate": (
                    len(stable_steps) / len(evaluated) if evaluated else None
                ),
                "stable_threshold_step": _mean_std(stable_steps),
            }
        )
    summary = {
        "group_count": len(groups),
        "successful_run_count": len(result.successful_runs),
        "failed_run_count": len(result.failed_runs),
        "groups": groups,
        "composite_score_created": False,
    }
    result.metadata["research_summary"] = summary
    return _jsonable(summary)


def run_active_research_benchmark(
    algorithms: Mapping[str, AlgorithmFactory | ActiveAlgorithm],
    config: ActiveResearchConfig | None = None,
    *,
    fail_fast: bool = True,
    progress: Callable[[int, ActiveScenarioConfig, str], None] | None = None,
) -> tuple[ActiveBenchmarkResult, dict[str, Any]]:
    """Run the protocol with robust clean stopping and fixed-horizon stochastic runs."""

    protocol = config or ActiveResearchConfig()
    all_runs: list[ActiveRunResult] = []
    index = 0
    for scenario in build_active_research_scenarios(protocol):
        stochastic = bool(scenario.metadata.get("stochastic", False))
        stop_config = RegretStoppingConfig(
            enabled=not stochastic,
            test_query_count=protocol.validation_query_count,
            seed=protocol.validation_seed,
            zero_regret_tolerance=protocol.zero_regret_tolerance,
            minimum_steps=1,
            consecutive_successes=protocol.consecutive_validation_successes,
            query_distribution="scenario",
        )

        def report(_inner: int, current: ActiveScenarioConfig, algorithm: str) -> None:
            nonlocal index
            index += 1
            if progress is not None:
                progress(index, current, algorithm)

        batch = ActiveBenchmarkSuite([scenario]).run(
            algorithms,
            fail_fast=fail_fast,
            stopping_config=stop_config,
            progress=report,
        )
        for run in batch.successful_runs:
            evaluate_active_run(
                run,
                ActiveEvaluationConfig(
                    test_query_count=protocol.test_query_count,
                    seed=protocol.test_seed,
                    evaluate_trajectory=True,
                    zero_regret_tolerance=protocol.zero_regret_tolerance,
                    query_distribution="scenario",
                    learning_regret_threshold=protocol.learning_regret_threshold,
                ),
            )
        all_runs.extend(batch.runs)

    result = ActiveBenchmarkResult(
        all_runs,
        metadata={
            "protocol": "compact-active-research-v1",
            "protocol_config": protocol.to_dict(),
            "scenario_family_count": 12,
            "seed_count": len(protocol.seeds),
            "run_count": len(all_runs),
            "validation_and_test_are_independent": True,
            "stochastic_runs_use_fixed_horizon": True,
            "objective": "(s * theta)^T x",
            "composite_score_created": False,
        },
    )
    return result, summarize_active_research(result)


def save_active_research(
    result: ActiveBenchmarkResult,
    summary: Mapping[str, Any],
    directory: str | Path,
) -> Path:
    destination = result.save(directory)
    (destination / "research-summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    return destination
