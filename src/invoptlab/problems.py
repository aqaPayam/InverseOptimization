from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .core import EnumerationOracle, ForwardProblem, LinearObjective, ParameterSpace
from .data import generate_dataset


def finite_choice_problem(
    parameter_dimension: int,
    feature_map: Callable[[Any, Any], Any],
    feasible_decisions: Callable[[Any], Sequence[Any]],
    *,
    parameter_space: ParameterSpace | None = None,
    name: str = "finite-choice",
) -> ForwardProblem:
    return ForwardProblem(
        LinearObjective(feature_map, parameter_dimension),
        parameter_space or ParameterSpace(parameter_dimension, "l2_ball"),
        EnumerationOracle(feasible_decisions),
        name=name,
    )


def random_choice_experiment(
    *,
    parameter_dimension: int = 2,
    observations: int = 30,
    alternatives: int = 8,
    true_theta: np.ndarray | None = None,
    parameter_space: ParameterSpace | None = None,
    noise_model: Any | None = None,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    contexts = [rng.normal(size=(alternatives, parameter_dimension)) for _ in range(observations)]
    problem = finite_choice_problem(
        parameter_dimension,
        feature_map=lambda context, decision: context[int(decision)],
        feasible_decisions=lambda context: list(range(context.shape[0])),
        parameter_space=parameter_space,
        name="random-finite-choice",
    )
    if true_theta is None:
        truth = rng.normal(size=parameter_dimension)
        truth /= np.linalg.norm(truth)
        if problem.parameter_space.kind == "simplex":
            truth = problem.parameter_space.project(np.abs(truth))
        else:
            truth = problem.parameter_space.project(truth)
    else:
        truth = problem.parameter_space.project(true_theta)
    dataset = generate_dataset(
        problem,
        contexts,
        truth,
        noise_model=noise_model,
        seed=seed + 1,
        name="random-choice-data",
    )
    return problem, dataset, truth


def knapsack_problem(
    feature_dimension: int,
    *,
    parameter_space: ParameterSpace | None = None,
    name: str = "binary-knapsack",
) -> ForwardProblem:
    """Create a small enumerated knapsack model.

    A context is a mapping with ``weights``, ``capacity``, and ``features``.
    The feature matrix has shape (items, feature_dimension). A decision is a
    binary vector and its aggregate feature vector is ``decision @ features``.
    """

    def feasible(context: dict[str, Any]):
        weights = np.asarray(context["weights"], dtype=float)
        capacity = float(context["capacity"])
        return [
            np.asarray(bits, dtype=int)
            for bits in product((0, 1), repeat=weights.size)
            if np.dot(weights, bits) <= capacity + 1e-9
        ]

    def features(context: dict[str, Any], decision: Any):
        return np.asarray(decision, dtype=float) @ np.asarray(context["features"], dtype=float)

    return finite_choice_problem(
        feature_dimension,
        features,
        feasible,
        parameter_space=parameter_space,
        name=name,
    )


def random_knapsack_contexts(
    count: int,
    *,
    items: int = 8,
    feature_dimension: int = 3,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    contexts = []
    for _ in range(count):
        weights = rng.integers(1, 10, size=items)
        contexts.append(
            {
                "weights": weights,
                "capacity": int(np.ceil(0.45 * weights.sum())),
                "features": rng.normal(size=(items, feature_dimension)),
            }
        )
    return contexts

