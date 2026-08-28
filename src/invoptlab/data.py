from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .core import ForwardProblem, InverseDataset, Observation, as_array
from .exceptions import ValidationError


def load_csv(
    path: str | Path,
    *,
    context_columns: Sequence[str],
    decision_columns: Sequence[str],
    timestamp_column: str | None = None,
    weight_column: str | None = None,
    name: str | None = None,
) -> InverseDataset:
    import pandas as pd

    frame = pd.read_csv(path)
    required = set(context_columns) | set(decision_columns)
    required |= {item for item in (timestamp_column, weight_column) if item is not None}
    missing = required - set(frame.columns)
    if missing:
        raise ValidationError(f"CSV is missing columns: {sorted(missing)}")
    observations = []
    for _, row in frame.iterrows():
        context = row[list(context_columns)].to_numpy(dtype=float)
        decision = row[list(decision_columns)].to_numpy(dtype=float)
        observations.append(
            Observation(
                context=context,
                decision=decision,
                timestamp=None if timestamp_column is None else row[timestamp_column],
                weight=1.0 if weight_column is None else float(row[weight_column]),
            )
        )
    return InverseDataset(observations, name=name or Path(path).stem)


def save_json(dataset: InverseDataset, path: str | Path) -> None:
    def serializable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return value

    payload = {
        "name": dataset.name,
        "metadata": serializable(dataset.metadata),
        "observations": [serializable(asdict(observation)) for observation in dataset],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> InverseDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return InverseDataset.from_records(payload["observations"], payload.get("name", Path(path).stem))


def generate_dataset(
    problem: ForwardProblem,
    contexts: Iterable[Any],
    true_theta: Any | Callable[[int, Any], Any],
    *,
    noise_model: Any | None = None,
    seed: int = 0,
    name: str = "synthetic",
) -> InverseDataset:
    rng = np.random.default_rng(seed)
    observations: list[Observation] = []
    for index, context in enumerate(contexts):
        theta = true_theta(index, context) if callable(true_theta) else true_theta
        theta = as_array(theta, name="true_theta").reshape(-1)
        clean_solution = problem.solve(theta, context)
        observed = clean_solution.decision
        noise_metadata: dict[str, Any] = {"type": "none"}
        if noise_model is not None:
            observed, noise_metadata = noise_model.apply(
                clean_solution.decision,
                context=context,
                problem=problem,
                theta=theta,
                rng=rng,
            )
        observations.append(
            Observation(
                context=context,
                decision=observed,
                clean_decision=clean_solution.decision,
                true_theta=theta.copy(),
                timestamp=index,
                noise=noise_metadata,
            )
        )
    dataset = InverseDataset(observations, name=name)
    problem.validate_dataset(dataset, check_feasibility=False)
    return dataset


def summarize_dataset(dataset: InverseDataset, problem: ForwardProblem | None = None) -> dict[str, Any]:
    weights = np.asarray([obs.weight for obs in dataset], dtype=float)
    result: dict[str, Any] = {
        "name": dataset.name,
        "size": len(dataset),
        "fingerprint": dataset.fingerprint,
        "weight_sum": float(weights.sum()),
        "timestamped_fraction": float(np.mean([obs.timestamp is not None for obs in dataset])),
        "ground_truth_theta_fraction": float(np.mean([obs.true_theta is not None for obs in dataset])),
        "clean_decision_fraction": float(np.mean([obs.clean_decision is not None for obs in dataset])),
        "experts": sorted({obs.expert_id for obs in dataset if obs.expert_id is not None}),
    }
    noise_types: dict[str, int] = {}
    for obs in dataset:
        key = str(obs.noise.get("type", "unknown"))
        noise_types[key] = noise_types.get(key, 0) + 1
    result["noise_types"] = noise_types
    if problem is not None:
        result["validation_warnings"] = problem.validate_dataset(dataset, check_feasibility=False)
    return result


def kfold_indices(size: int, folds: int = 5, *, seed: int = 0, shuffle: bool = True):
    if folds < 2 or folds > size:
        raise ValidationError("folds must be between 2 and dataset size")
    indices = np.arange(size)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    chunks = np.array_split(indices, folds)
    for index in range(folds):
        validation = chunks[index]
        training = np.concatenate([chunk for j, chunk in enumerate(chunks) if j != index])
        yield training, validation

