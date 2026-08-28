from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .estimators import IncenterEstimator, OnlineEstimator, ProjectedSubgradientEstimator
from .experiments import ExperimentConfig, ExperimentResult, ExperimentRunner
from .losses import AugmentedSuboptimalityLoss, SuboptimalityLoss
from .noise import BoltzmannNoise, EpsilonOptimalNoise, RandomFeasibleNoise
from .problems import random_choice_experiment


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML configurations require `pip install pyyaml`") from exc
    result = yaml.safe_load(text)
    if not isinstance(result, dict):
        raise ValueError("The configuration root must be a mapping")
    return result


def noise_from_config(config: dict[str, Any]):
    kind = config.get("type", "none")
    if kind == "none":
        return None
    if kind == "random_feasible":
        return RandomFeasibleNoise(float(config.get("probability", 0.1)))
    if kind == "boltzmann":
        return BoltzmannNoise(float(config.get("temperature", 0.1)))
    if kind == "epsilon_optimal":
        return EpsilonOptimalNoise(float(config.get("epsilon", 0.1)))
    raise ValueError(f"Unsupported configured noise model: {kind!r}")


def estimator_from_config(config: dict[str, Any]):
    kind = config.get("type", "incenter")
    if kind == "incenter":
        return IncenterEstimator(
            tolerance=float(config.get("tolerance", 1e-8)),
            max_iterations=int(config.get("max_iterations", 2_000)),
            sequential_history=bool(config.get("sequential_history", True)),
        )
    if kind in {"asl", "sl", "online_asl", "online_sl"}:
        loss = AugmentedSuboptimalityLoss(
            margin_scale=float(config.get("margin_scale", 1.0))
        ) if "asl" in kind else SuboptimalityLoss()
        estimator = ProjectedSubgradientEstimator(
            loss=loss,
            learning_rate=float(config.get("learning_rate", 0.25)),
            epochs=int(config.get("epochs", 120)),
            regularization=float(config.get("regularization", 1e-3)),
            stochastic=bool(config.get("stochastic", False)),
            mirror_descent=bool(config.get("mirror_descent", False)),
            seed=int(config.get("seed", 0)),
            record_every=int(config.get("record_every", 5)),
        )
        return OnlineEstimator(estimator) if kind.startswith("online_") else estimator
    raise ValueError(f"Unsupported configured estimator: {kind!r}")


def run_configuration(path: str | Path) -> tuple[ExperimentResult, Any, Any, Any]:
    config = load_config(path)
    experiment_values = config.get("experiment", {})
    problem_values = config.get("problem", {})
    if problem_values.get("type", "random_finite_choice") != "random_finite_choice":
        raise ValueError(
            "The generic YAML runner currently constructs the built-in random_finite_choice problem. "
            "Use the Python API for arbitrary user callables."
        )
    seed = int(experiment_values.get("seed", 0))
    problem, dataset, truth = random_choice_experiment(
        parameter_dimension=int(problem_values.get("parameter_dimension", 2)),
        observations=int(problem_values.get("observations", 16)),
        alternatives=int(problem_values.get("alternatives", 5)),
        noise_model=noise_from_config(config.get("noise", {})),
        seed=seed,
    )
    estimator_values = dict(config.get("estimator", {}))
    estimator_values.setdefault("seed", seed)
    estimator = estimator_from_config(estimator_values)
    experiment = ExperimentConfig(
        name=str(experiment_values.get("name", Path(path).stem)),
        seed=seed,
        validate_feasibility=bool(experiment_values.get("validate_feasibility", True)),
        compute_geometry=bool(experiment_values.get("compute_geometry", True)),
        geometry_samples=int(experiment_values.get("geometry_samples", 1_000)),
        tags=dict(experiment_values.get("tags", {})),
    )
    result = ExperimentRunner(experiment).run(problem, dataset, estimator)
    outputs = config.get("outputs", {})
    directory = Path(outputs.get("directory", f"outputs/experiments/{experiment.name}"))
    result.save(directory)
    truth_for_plot = truth if truth is not None else None
    if outputs.get("static_plots", True):
        try:
            from .visualization import save_figure

            if result.geometry_history and problem.parameter_space.dimension == 2:
                path = directory / "consistency_cone.png"
                save_figure(result.plot_cone(problem, true_theta=truth_for_plot), str(path))
                result.artifacts["consistency_cone"] = str(path)
            if result.history.steps:
                path = directory / "parameter_history.png"
                save_figure(result.plot_parameters(), str(path))
                result.artifacts["parameter_history"] = str(path)
            path = directory / "regret.png"
            save_figure(result.plot_regret(), str(path))
            result.artifacts["regret"] = str(path)
        except (ImportError, ValueError):
            pass
    if outputs.get("interactive_plots", True) and result.geometry_history:
        try:
            from .visualization import save_figure

            if problem.parameter_space.dimension == 2:
                path = directory / "cone_evolution.html"
                save_figure(result.animate_cone(problem, true_theta=truth_for_plot), str(path))
                result.artifacts["cone_evolution"] = str(path)
            elif problem.parameter_space.dimension == 3:
                path = directory / "consistency_cone_3d.html"
                save_figure(result.plot_cone(problem, true_theta=truth_for_plot), str(path))
                result.artifacts["consistency_cone_3d"] = str(path)
        except (ImportError, ValueError):
            pass
    if outputs.get("html_report", True):
        loss = getattr(estimator, "loss", getattr(getattr(estimator, "base_estimator", None), "loss", None))
        loss = loss or SuboptimalityLoss()
        result.generate_report(problem, dataset, directory / "report.html", loss=loss)
    result.save(directory)
    return result, problem, dataset, truth
