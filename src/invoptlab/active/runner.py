from __future__ import annotations

import copy
import importlib
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import ValidationError
from .algorithms import ActiveAlgorithm
from .config import (
    ActiveBenchmarkGrid,
    ActiveScenarioConfig,
    DecisionSpaceConfig,
    ExpertConfig,
    ObservationNoiseConfig,
    ParameterNoiseConfig,
    QuerySpaceConfig,
)
from .environment import ActiveInverseEnvironment
from .stopping import RegretStoppingConfig, RegretStoppingRule
from .types import (
    ActiveAction,
    ActiveBenchmarkResult,
    ActiveRunResult,
    ActiveStepRecord,
    AlgorithmObservation,
)


AlgorithmFactory = Callable[[], ActiveAlgorithm]


def _validate_parameter(value: Any, dimension: int, name: str) -> np.ndarray:
    parameter = np.asarray(value, dtype=float).reshape(-1)
    if parameter.size != dimension or not np.all(np.isfinite(parameter)):
        raise ValidationError(f"{name} must be finite and have dimension {dimension}")
    return parameter.copy()


def _validate_action(action: ActiveAction, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(action, ActiveAction):
        raise ValidationError("algorithm.propose must return an ActiveAction")
    query = np.asarray(action.query, dtype=float).reshape(-1)
    if query.size != dimension or not np.all(np.isfinite(query)):
        raise ValidationError(f"algorithm query must be finite and have dimension {dimension}")
    theta = _validate_parameter(action.theta_hat, dimension, "algorithm theta_hat")
    return query, theta


class ActiveBenchmarkRunner:
    def __init__(
        self,
        *,
        respect_stop_requests: bool = False,
        stopping_config: RegretStoppingConfig | None = None,
    ):
        self.respect_stop_requests = bool(respect_stop_requests)
        self.stopping_config = stopping_config or RegretStoppingConfig()

    def run(
        self,
        scenario: ActiveScenarioConfig,
        algorithm: ActiveAlgorithm,
        *,
        algorithm_seed: int | None = None,
    ) -> ActiveRunResult:
        environment = ActiveInverseEnvironment(scenario)
        seed = scenario.seed if algorithm_seed is None else int(algorithm_seed)
        algorithm_rng = np.random.default_rng(np.random.SeedSequence([scenario.seed, seed, 99173]))
        context = environment.algorithm_context(algorithm_seed=seed)
        algorithm.reset(context, algorithm_rng)
        stopping_rule = None
        if self.stopping_config.enabled:
            stopping_rule = RegretStoppingRule(self.stopping_config)
            stopping_rule.reset(scenario, environment.theta_true, environment.decision_space)
        history: list[AlgorithmObservation] = []
        records: list[ActiveStepRecord] = []
        stopped_early = False
        stopping_criterion_met = False
        stopping_reason: str | None = None
        final_stopping_diagnostics: dict[str, Any] = {}
        started = time.perf_counter()
        for _ in range(scenario.horizon):
            action = algorithm.propose(tuple(history))
            query, theta_before = _validate_action(action, scenario.dimension)
            feedback = environment.step(query)
            observation = feedback.public()
            algorithm.observe(observation)
            theta_after = _validate_parameter(
                algorithm.current_estimate(),
                scenario.dimension,
                "algorithm current estimate",
            )
            stopping_check = (
                stopping_rule.check(theta_after, feedback.step)
                if stopping_rule is not None
                else None
            )
            benchmark_stop_requested = bool(
                stopping_check is not None and stopping_check.should_stop
            )
            stopping_diagnostics = (
                stopping_check.to_dict() if stopping_check is not None else {}
            )
            final_stopping_diagnostics = stopping_diagnostics
            history.append(observation)
            records.append(
                ActiveStepRecord(
                    step=feedback.step,
                    query=feedback.query.copy(),
                    theta_hat_before=theta_before,
                    theta_hat_after=theta_after,
                    true_theta=environment.theta_true.copy(),
                    expert_parameter=feedback.expert_parameter.copy(),
                    true_decision=feedback.true_decision.copy(),
                    observed_decision=feedback.observed_decision.copy(),
                    observation_mask=None if feedback.observation_mask is None else feedback.observation_mask.copy(),
                    objective_value=feedback.objective_value,
                    stop_requested=bool(action.stop_requested),
                    benchmark_stop_requested=benchmark_stop_requested,
                    benchmark_stop_reason=(
                        stopping_check.reason if stopping_check is not None else None
                    ),
                    stopping_diagnostics=stopping_diagnostics,
                    action_diagnostics=dict(action.diagnostics),
                    update_diagnostics=dict(algorithm.diagnostics()),
                    expert_metadata=dict(feedback.expert_metadata),
                    parameter_noise_metadata=dict(feedback.parameter_noise_metadata),
                    observation_noise_metadata=dict(feedback.observation_noise_metadata),
                )
            )
            algorithm_stop = bool(action.stop_requested and self.respect_stop_requests)
            if benchmark_stop_requested:
                stopping_criterion_met = True
                stopping_reason = stopping_check.reason
            elif algorithm_stop:
                stopping_reason = "algorithm stop request"
            if benchmark_stop_requested or algorithm_stop:
                stopped_early = feedback.step < scenario.horizon
                break
        runtime = time.perf_counter() - started
        return ActiveRunResult(
            scenario=scenario,
            algorithm_name=getattr(algorithm, "name", type(algorithm).__name__),
            seed=seed,
            true_theta=environment.theta_true.copy(),
            records=records,
            runtime_seconds=float(runtime),
            stopped_early=stopped_early,
            metadata={
                "algorithm_class": type(algorithm).__name__,
                "environment_class": type(environment).__name__,
                "scoring_applied": False,
                "evaluation_applied": False,
                "external_stopping_enabled": self.stopping_config.enabled,
                "stopping_rule": self.stopping_config.to_dict(),
                "stopping_time": len(records),
                "stopping_criterion_met": stopping_criterion_met,
                "stopping_reason": stopping_reason,
                "final_stopping_diagnostics": final_stopping_diagnostics,
                "noise_calibration": environment.noise_calibration,
            },
        )


class ActiveBenchmarkSuite:
    """Run arbitrary algorithm factories on a lazy collection of environments."""

    def __init__(self, scenarios: Iterable[ActiveScenarioConfig]):
        self.scenarios = scenarios

    @classmethod
    def from_grid(cls, grid: ActiveBenchmarkGrid, *, limit: int | None = None):
        return cls(grid.scenarios(limit=limit))

    def run(
        self,
        algorithms: Mapping[str, AlgorithmFactory | ActiveAlgorithm],
        *,
        fail_fast: bool = True,
        respect_stop_requests: bool = False,
        stopping_config: RegretStoppingConfig | None = None,
        progress: Callable[[int, ActiveScenarioConfig, str], None] | None = None,
    ) -> ActiveBenchmarkResult:
        runner = ActiveBenchmarkRunner(
            respect_stop_requests=respect_stop_requests,
            stopping_config=stopping_config,
        )
        runs: list[ActiveRunResult] = []
        scenario_count = 0
        run_index = 0
        for scenario in self.scenarios:
            scenario_count += 1
            for registered_name, factory_or_instance in algorithms.items():
                run_index += 1
                if progress is not None:
                    progress(run_index, scenario, registered_name)
                algorithm = (
                    factory_or_instance()
                    if callable(factory_or_instance) and not isinstance(factory_or_instance, ActiveAlgorithm)
                    else copy.deepcopy(factory_or_instance)
                )
                try:
                    result = runner.run(scenario, algorithm)
                    result.algorithm_name = registered_name
                except Exception as exc:
                    if fail_fast:
                        raise
                    result = ActiveRunResult(
                        scenario=scenario,
                        algorithm_name=registered_name,
                        seed=scenario.seed,
                        true_theta=np.empty(0),
                        records=[],
                        runtime_seconds=0.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                runs.append(result)
        return ActiveBenchmarkResult(
            runs,
            metadata={
                "scenario_count": scenario_count,
                "algorithm_count": len(algorithms),
                "run_count": len(runs),
                "scoring_applied": False,
                "evaluation_applied": False,
                "external_stopping_enabled": runner.stopping_config.enabled,
                "stopping_rule": runner.stopping_config.to_dict(),
            },
        )


def load_active_scenarios(path: str | Path) -> list[ActiveScenarioConfig]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        import json

        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML benchmark files require pyyaml") from exc
        payload = yaml.safe_load(text)
    if isinstance(payload, Mapping) and "grid" in payload:
        raise ValidationError("configuration contains a grid; use load_active_benchmark instead")
    values = payload.get("scenarios", payload) if isinstance(payload, Mapping) else payload
    if isinstance(values, Mapping):
        values = [values]
    if not isinstance(values, list):
        raise ValidationError("active benchmark configuration must contain a scenario list")
    return [ActiveScenarioConfig(**value) for value in values]


def _component_list(values: Any, config_type: type, default: list[Any]) -> tuple[Any, ...]:
    if values is None:
        return tuple(default)
    result = []
    for value in values:
        if isinstance(value, config_type):
            result.append(value)
        elif isinstance(value, str):
            result.append(config_type(kind=value))
        elif isinstance(value, Mapping):
            result.append(config_type(**value))
        else:
            raise ValidationError(f"invalid {config_type.__name__} grid value")
    return tuple(result)


def active_grid_from_dict(payload: Mapping[str, Any]) -> ActiveBenchmarkGrid:
    values = payload.get("grid", payload)
    if not isinstance(values, Mapping):
        raise ValidationError("active benchmark grid must be a mapping")
    return ActiveBenchmarkGrid(
        dimensions=tuple(int(item) for item in values.get("dimensions", (5, 20, 50))),
        experts=_component_list(values.get("experts"), ExpertConfig, [ExpertConfig()]),
        decision_spaces=_component_list(
            values.get("decision_spaces"), DecisionSpaceConfig, [DecisionSpaceConfig()]
        ),
        query_spaces=_component_list(
            values.get("query_spaces"), QuerySpaceConfig, [QuerySpaceConfig()]
        ),
        observation_noises=_component_list(
            values.get("observation_noises"), ObservationNoiseConfig, [ObservationNoiseConfig()]
        ),
        parameter_noises=_component_list(
            values.get("parameter_noises"), ParameterNoiseConfig, [ParameterNoiseConfig()]
        ),
        seeds=tuple(int(item) for item in values.get("seeds", (0,))),
        horizon=int(values.get("horizon", 50)),
        name_prefix=str(values.get("name_prefix", "active-benchmark")),
    )


def load_active_benchmark(
    path: str | Path,
    *,
    limit: int | None = None,
) -> ActiveBenchmarkSuite:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        import json

        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML benchmark files require pyyaml") from exc
        payload = yaml.safe_load(text)
    if isinstance(payload, Mapping) and "grid" in payload:
        return ActiveBenchmarkSuite.from_grid(active_grid_from_dict(payload), limit=limit)
    scenarios = load_active_scenarios(path)
    return ActiveBenchmarkSuite(scenarios[:limit] if limit is not None else scenarios)


def load_algorithm_factory(specification: str) -> AlgorithmFactory:
    """Load ``module:object`` where object is an algorithm class or zero-argument factory."""

    if ":" not in specification:
        raise ValidationError("algorithm specification must use module:object")
    module_name, object_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, object_name)
    if not callable(value):
        raise ValidationError("loaded algorithm object must be callable")

    def factory() -> ActiveAlgorithm:
        algorithm = value()
        if not isinstance(algorithm, ActiveAlgorithm):
            raise ValidationError("algorithm factory must return an ActiveAlgorithm")
        return algorithm

    return factory
