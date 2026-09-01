from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import ActiveScenarioConfig, _jsonable


Array = np.ndarray


@dataclass(slots=True)
class AlgorithmContext:
    dimension: int
    horizon: int
    query_candidates: Array
    seed: int
    scenario_name: str
    public_environment: Mapping[str, Any]


@dataclass(slots=True)
class AlgorithmObservation:
    step: int
    query: Array
    observed_decision: Array
    observation_mask: Array | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActiveAction:
    query: Array
    theta_hat: Array
    stop_requested: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EnvironmentFeedback:
    step: int
    query: Array
    expert_parameter: Array
    true_decision: Array
    observed_decision: Array
    observation_mask: Array | None
    objective_value: float
    expert_metadata: Mapping[str, Any] = field(default_factory=dict)
    parameter_noise_metadata: Mapping[str, Any] = field(default_factory=dict)
    observation_noise_metadata: Mapping[str, Any] = field(default_factory=dict)

    def public(self) -> AlgorithmObservation:
        metadata = {
            "partial": self.observation_mask is not None,
        }
        return AlgorithmObservation(
            step=self.step,
            query=self.query.copy(),
            observed_decision=self.observed_decision.copy(),
            observation_mask=None if self.observation_mask is None else self.observation_mask.copy(),
            metadata=metadata,
        )


@dataclass(slots=True)
class ActiveStepRecord:
    step: int
    query: Array
    theta_hat_before: Array
    theta_hat_after: Array
    true_theta: Array
    expert_parameter: Array
    true_decision: Array
    observed_decision: Array
    observation_mask: Array | None
    objective_value: float
    stop_requested: bool
    action_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    expert_metadata: Mapping[str, Any] = field(default_factory=dict)
    parameter_noise_metadata: Mapping[str, Any] = field(default_factory=dict)
    observation_noise_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_latent: bool = True) -> dict[str, Any]:
        value = _jsonable(asdict(self))
        if not include_latent:
            for key in ("true_theta", "expert_parameter", "true_decision"):
                value.pop(key, None)
        return value


@dataclass
class ActiveRunResult:
    scenario: ActiveScenarioConfig
    algorithm_name: str
    seed: int
    true_theta: Array
    records: list[ActiveStepRecord]
    runtime_seconds: float
    stopped_early: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def queries(self) -> Array:
        return np.vstack([record.query for record in self.records]) if self.records else np.empty((0, self.scenario.dimension))

    @property
    def parameter_history(self) -> Array:
        return np.vstack([record.theta_hat_after for record in self.records]) if self.records else np.empty((0, self.scenario.dimension))

    @property
    def observed_decisions(self) -> Array:
        return np.vstack([record.observed_decision for record in self.records]) if self.records else np.empty((0, self.scenario.dimension))

    def to_dict(self, *, include_latent: bool = True) -> dict[str, Any]:
        payload = {
            "scenario": self.scenario.to_dict(),
            "algorithm_name": self.algorithm_name,
            "seed": self.seed,
            "true_theta": self.true_theta,
            "runtime_seconds": self.runtime_seconds,
            "stopped_early": self.stopped_early,
            "error": self.error,
            "metadata": self.metadata,
            "records": [record.to_dict(include_latent=include_latent) for record in self.records],
        }
        if not include_latent:
            payload.pop("true_theta", None)
        return _jsonable(payload)

    def save_json(self, path: str | Path, *, include_latent: bool = True) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(include_latent=include_latent), indent=2),
            encoding="utf-8",
        )
        return destination

    def to_frame(self, *, include_latent: bool = True):
        import pandas as pd

        rows = []
        for record in self.records:
            row = record.to_dict(include_latent=include_latent)
            flattened: dict[str, Any] = {"step": row.pop("step")}
            for key, value in row.items():
                if isinstance(value, list) and all(isinstance(item, (int, float, bool)) for item in value):
                    flattened.update({f"{key}_{index}": item for index, item in enumerate(value)})
                else:
                    flattened[key] = value
            rows.append(flattened)
        return pd.DataFrame(rows)


@dataclass(slots=True)
class ActiveBenchmarkResult:
    runs: list[ActiveRunResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def successful_runs(self) -> list[ActiveRunResult]:
        return [run for run in self.runs if run.error is None]

    @property
    def failed_runs(self) -> list[ActiveRunResult]:
        return [run for run in self.runs if run.error is not None]

    def save(self, directory: str | Path, *, include_latent: bool = True) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = []
        for index, run in enumerate(self.runs):
            filename = f"{index:05d}-{run.scenario.name}-{run.algorithm_name}.json"
            run.save_json(destination / filename, include_latent=include_latent)
            manifest.append({
                "file": filename,
                "scenario": run.scenario.name,
                "algorithm": run.algorithm_name,
                "seed": run.seed,
                "error": run.error,
            })
        (destination / "manifest.json").write_text(
            json.dumps(_jsonable({"runs": manifest, "metadata": self.metadata}), indent=2),
            encoding="utf-8",
        )
        return destination

