from __future__ import annotations

import copy
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .capabilities import Capability
from .core import EstimatorHistory, ForwardProblem, InverseDataset
from .data import summarize_dataset
from .geometry import GeometrySnapshot, build_consistency_constraints, build_geometry_history
from .metrics import consistency_metrics, evaluate_predictions
from .statistics import summarize_repeated_metrics


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", repr(value))
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 0
    validate_feasibility: bool = True
    compute_geometry: bool = True
    geometry_samples: int = 2_000
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    name: str
    theta: np.ndarray
    metrics: dict[str, float]
    per_observation: list[dict[str, Any]]
    predictions: list[Any]
    history: EstimatorHistory
    geometry_history: list[GeometrySnapshot]
    dataset_summary: dict[str, Any]
    diagnostics: dict[str, Any]
    config: ExperimentConfig
    artifacts: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "theta": self.theta.tolist(),
            "metrics": self.metrics,
            "dataset": self.dataset_summary,
            "diagnostics": self.diagnostics,
        }

    def save(self, directory: str | Path) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "summary.json").write_text(
            json.dumps(_jsonable(self.summary()), indent=2), encoding="utf-8"
        )
        (destination / "config.json").write_text(
            json.dumps(_jsonable(self.config), indent=2), encoding="utf-8"
        )
        try:
            import pandas as pd

            pd.DataFrame(self.per_observation).to_csv(destination / "per_observation.csv", index=False)
            if self.history.steps:
                history_rows = []
                for step in self.history.steps:
                    row = {
                        "step": step.step,
                        "loss": step.loss,
                        "radius": step.radius,
                        **{f"theta_{index}": value for index, value in enumerate(step.theta)},
                        **step.metrics,
                    }
                    history_rows.append(row)
                pd.DataFrame(history_rows).to_csv(destination / "parameter_history.csv", index=False)
            if self.geometry_history:
                pd.DataFrame(
                    [{"step": item.step, **item.statistics} for item in self.geometry_history]
                ).to_csv(destination / "geometry_history.csv", index=False)
        except ImportError:
            pass
        return destination

    def plot_cone(self, problem: ForwardProblem, *, step: int = -1, true_theta: np.ndarray | None = None):
        if not self.geometry_history:
            raise ValueError("No geometry history is available")
        from .visualization import plot_cone_2d, plot_cone_3d

        snapshot = self.geometry_history[step]
        if problem.parameter_space.dimension == 2:
            return plot_cone_2d(problem, snapshot, true_theta=true_theta)
        if problem.parameter_space.dimension == 3:
            return plot_cone_3d(problem, snapshot, true_theta=true_theta)
        raise ValueError("Cone plots are supported only for parameter dimensions 2 and 3")

    def animate_cone(self, problem: ForwardProblem, *, true_theta: np.ndarray | None = None):
        from .visualization import animate_cone_2d

        return animate_cone_2d(problem, self.geometry_history, true_theta=true_theta)

    def plot_parameters(self):
        from .visualization import plot_parameter_history

        return plot_parameter_history(self)

    def plot_regret(self):
        from .visualization import plot_regret

        return plot_regret(self)

    def generate_report(
        self,
        problem: ForwardProblem,
        dataset: InverseDataset,
        destination: str | Path,
        *,
        loss: Any | None = None,
    ) -> Path:
        from .reporting import generate_html_report

        return generate_html_report(self, problem, dataset, destination, loss=loss)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config or ExperimentConfig()

    def run(
        self,
        problem: ForwardProblem,
        dataset: InverseDataset,
        estimator: Any,
    ) -> ExperimentResult:
        warnings = problem.validate_dataset(
            dataset, check_feasibility=self.config.validate_feasibility
        )
        start = time.perf_counter()
        fitted = estimator.fit(problem, dataset)
        elapsed = time.perf_counter() - start
        predictions, rows, metrics = evaluate_predictions(problem, dataset, fitted.theta_)
        geometry_history: list[GeometrySnapshot] = []
        if Capability.LINEAR_IN_THETA in problem.capabilities:
            constraints = build_consistency_constraints(problem, dataset)
            metrics.update(consistency_metrics(constraints, fitted.theta_))
            if self.config.compute_geometry:
                parameter_history = fitted.history_.parameters
                if parameter_history.shape[0] != len(dataset):
                    parameter_history = np.repeat(fitted.theta_[None, :], len(dataset), axis=0)
                geometry_history = build_geometry_history(
                    problem,
                    dataset,
                    parameter_history,
                    sample_count=self.config.geometry_samples,
                    seed=self.config.seed,
                )
                if len(fitted.history_.steps) == len(dataset):
                    for snapshot, record in zip(geometry_history, fitted.history_.steps):
                        snapshot.incenter = record.theta.copy() if record.radius is not None else None
                        snapshot.inradius = record.radius
        metrics["fit_time_seconds"] = float(elapsed)
        diagnostics = {
            "warnings": warnings,
            "estimator": type(estimator).__name__,
            "problem_capabilities": sorted(cap.value for cap in problem.capabilities),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        if hasattr(fitted, "result_"):
            diagnostics["solver_result"] = _jsonable(fitted.result_)
        return ExperimentResult(
            name=self.config.name,
            theta=fitted.theta_.copy(),
            metrics=metrics,
            per_observation=rows,
            predictions=predictions,
            history=fitted.history_,
            geometry_history=geometry_history,
            dataset_summary=summarize_dataset(dataset, problem),
            diagnostics=diagnostics,
            config=self.config,
        )


@dataclass
class SweepResult:
    runs: list[ExperimentResult]
    parameters: list[dict[str, Any]]
    summary: dict[str, dict[str, float]]


def run_sweep(
    problem: ForwardProblem,
    dataset: InverseDataset,
    estimator_factory: Callable[[dict[str, Any]], Any],
    parameter_grid: Iterable[dict[str, Any]],
    *,
    seeds: Iterable[int] = (0,),
    name: str = "sweep",
) -> SweepResult:
    runs: list[ExperimentResult] = []
    parameters: list[dict[str, Any]] = []
    for setting in parameter_grid:
        for seed in seeds:
            config = ExperimentConfig(name=f"{name}-{len(runs):04d}", seed=seed)
            estimator = estimator_factory({**setting, "seed": seed})
            runs.append(ExperimentRunner(config).run(problem, dataset, estimator))
            parameters.append({**setting, "seed": seed})
    return SweepResult(runs, parameters, summarize_repeated_metrics([run.metrics for run in runs]))
