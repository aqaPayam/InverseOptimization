from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .estimators import IncenterEstimator, ProjectedSubgradientEstimator
from .configuration import run_configuration
from .experiments import ExperimentConfig, ExperimentRunner
from .losses import AugmentedSuboptimalityLoss
from .noise import RandomFeasibleNoise
from .problems import random_choice_experiment


def _demo(arguments: argparse.Namespace) -> int:
    noise = None if arguments.noise <= 0 else RandomFeasibleNoise(arguments.noise)
    problem, dataset, truth = random_choice_experiment(
        parameter_dimension=arguments.dimension,
        observations=arguments.observations,
        alternatives=arguments.alternatives,
        noise_model=noise,
        seed=arguments.seed,
    )
    if arguments.estimator == "incenter":
        estimator = IncenterEstimator(sequential_history=True)
    else:
        estimator = ProjectedSubgradientEstimator(
            loss=AugmentedSuboptimalityLoss(),
            epochs=arguments.epochs,
            seed=arguments.seed,
            record_every=max(1, arguments.epochs // 100),
        )
    runner = ExperimentRunner(ExperimentConfig(name="cli-demo", seed=arguments.seed, geometry_samples=1_000))
    result = runner.run(problem, dataset, estimator)
    output = Path(arguments.output)
    result.save(output)
    try:
        result.generate_report(problem, dataset, output / "report.html", loss=AugmentedSuboptimalityLoss())
    except ImportError:
        pass
    print(json.dumps(result.summary(), indent=2))
    print(f"True theta: {np.array2string(truth, precision=5)}")
    print(f"Artifacts: {output.resolve()}")
    return 0


def _show(arguments: argparse.Namespace) -> int:
    summary = Path(arguments.run) / "summary.json"
    if not summary.exists():
        raise SystemExit(f"No summary.json found in {Path(arguments.run).resolve()}")
    print(summary.read_text(encoding="utf-8"))
    return 0


def _run_config(arguments: argparse.Namespace) -> int:
    result, _, _, truth = run_configuration(arguments.config)
    print(json.dumps(result.summary(), indent=2))
    print(f"True theta: {np.array2string(truth, precision=5)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="invoptlab", description="Inverse-optimization experiment laboratory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run a complete synthetic experiment")
    demo.add_argument("--dimension", type=int, default=2, choices=(2, 3))
    demo.add_argument("--observations", type=int, default=16)
    demo.add_argument("--alternatives", type=int, default=5)
    demo.add_argument("--noise", type=float, default=0.0)
    demo.add_argument("--estimator", choices=("incenter", "asl"), default="incenter")
    demo.add_argument("--epochs", type=int, default=120)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output", default="outputs/experiments/cli-demo")
    demo.set_defaults(function=_demo)
    show = subparsers.add_parser("show", help="Print a saved run summary")
    show.add_argument("run")
    show.set_defaults(function=_show)
    run = subparsers.add_parser("run", help="Run a YAML or JSON experiment configuration")
    run.add_argument("config")
    run.set_defaults(function=_run_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.function(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
