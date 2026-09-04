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
from .active import (
    ActiveBenchmarkRunner,
    ActiveEvaluationConfig,
    ActiveScenarioConfig,
    ActiveResearchConfig,
    QuerySpaceConfig,
    RandomActiveAlgorithm,
    RegretStoppingConfig,
    GeniousPedroAlgorithm,
    UniformOnlineSAMDAlgorithm,
    UniformRandomIncenterAlgorithm,
    NestedLangevinActiveAlgorithm,
    evaluate_active_benchmark,
    load_active_benchmark,
    load_algorithm_factory,
    run_active_research_benchmark,
    save_active_research,
)


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


def _active_smoke(arguments: argparse.Namespace) -> int:
    scenario = ActiveScenarioConfig(
        name="active-smoke",
        dimension=arguments.dimension,
        horizon=arguments.horizon,
        seed=arguments.seed,
        query_space=QuerySpaceConfig(candidate_count=max(8, 2 * arguments.dimension)),
    )
    result = ActiveBenchmarkRunner().run(scenario, RandomActiveAlgorithm())
    destination = result.save_json(arguments.output)
    print(json.dumps({
        "scenario": result.scenario.name,
        "algorithm": result.algorithm_name,
        "steps": len(result.records),
        "trajectory": str(destination.resolve()),
        "evaluation_applied": False,
        "scoring_applied": False,
    }, indent=2))
    return 0


def _active_run(arguments: argparse.Namespace) -> int:
    suite = load_active_benchmark(arguments.config, limit=arguments.limit)
    algorithms = {}
    for specification in arguments.algorithm:
        if specification == "random":
            algorithms["random-smoke-test"] = lambda: RandomActiveAlgorithm()
            continue
        if specification == "nested-langevin":
            algorithms["nested-langevin-disagreement"] = NestedLangevinActiveAlgorithm
            continue
        if specification == "diffusion":
            algorithms["Diffusion"] = NestedLangevinActiveAlgorithm
            continue
        if specification == "uniform-incenter":
            algorithms["uniform-random-sequential-incenter"] = (
                lambda: UniformRandomIncenterAlgorithm()
            )
            continue
        if specification == "genious-pedro":
            algorithms["Genious Pedro"] = GeniousPedroAlgorithm
            continue
        if specification == "uniform-online-samd":
            algorithms["Uniform Online SAMD"] = UniformOnlineSAMDAlgorithm
            continue
        name, separator, import_path = specification.partition("=")
        if not separator:
            import_path = name
            name = import_path.rsplit(":", 1)[-1]
        algorithms[name] = load_algorithm_factory(import_path)
    result = suite.run(
        algorithms,
        fail_fast=not arguments.continue_on_error,
        respect_stop_requests=arguments.respect_stop,
        stopping_config=RegretStoppingConfig(
            enabled=not arguments.no_zero_regret_stop,
            test_query_count=arguments.test_queries,
            seed=arguments.evaluation_seed,
            zero_regret_tolerance=arguments.stop_regret_tolerance,
            minimum_steps=arguments.minimum_stop_time,
        ),
        progress=lambda index, scenario, algorithm: print(
            f"[{index}] {scenario.name} :: {algorithm}", flush=True
        ),
    )
    if arguments.evaluate:
        evaluate_active_benchmark(
            result,
            ActiveEvaluationConfig(
                test_query_count=arguments.test_queries,
                seed=arguments.evaluation_seed,
                evaluate_trajectory=arguments.evaluation_trajectory,
            ),
        )
    destination = result.save(arguments.output)
    print(json.dumps({
        **result.metadata,
        "successful_runs": len(result.successful_runs),
        "failed_runs": len(result.failed_runs),
        "output": str(destination.resolve()),
    }, indent=2))
    return 1 if result.failed_runs else 0


def _active_research(arguments: argparse.Namespace) -> int:
    algorithms = {}
    for specification in arguments.algorithm:
        if specification == "random":
            algorithms["random-smoke-test"] = lambda: RandomActiveAlgorithm()
            continue
        if specification == "nested-langevin":
            algorithms["nested-langevin-disagreement"] = NestedLangevinActiveAlgorithm
            continue
        if specification == "diffusion":
            algorithms["Diffusion"] = NestedLangevinActiveAlgorithm
            continue
        if specification == "uniform-incenter":
            algorithms["uniform-random-sequential-incenter"] = (
                lambda: UniformRandomIncenterAlgorithm()
            )
            continue
        if specification == "genious-pedro":
            algorithms["Genious Pedro"] = GeniousPedroAlgorithm
            continue
        if specification == "uniform-online-samd":
            algorithms["Uniform Online SAMD"] = UniformOnlineSAMDAlgorithm
            continue
        name, separator, import_path = specification.partition("=")
        if not separator:
            import_path = name
            name = import_path.rsplit(":", 1)[-1]
        algorithms[name] = load_algorithm_factory(import_path)
    protocol = ActiveResearchConfig(
        seeds=tuple(arguments.seeds),
        horizon=arguments.horizon,
        candidate_count=arguments.candidates,
        validation_query_count=arguments.validation_queries,
        test_query_count=arguments.test_queries,
        fixed_horizon=arguments.fixed_horizon,
        learning_angular_threshold_degrees=arguments.angular_threshold,
    )
    result, summary = run_active_research_benchmark(
        algorithms,
        protocol,
        fail_fast=not arguments.continue_on_error,
        progress=lambda index, scenario, algorithm: print(
            f"[{index}] {scenario.name} :: {algorithm}", flush=True
        ),
    )
    destination = save_active_research(result, summary, arguments.output)
    print(json.dumps({
        **result.metadata,
        "successful_runs": len(result.successful_runs),
        "failed_runs": len(result.failed_runs),
        "output": str(destination.resolve()),
    }, indent=2))
    return 1 if result.failed_runs else 0


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
    active_smoke = subparsers.add_parser(
        "active-smoke", help="Run a tiny random-algorithm plumbing check"
    )
    active_smoke.add_argument("--dimension", type=int, default=5)
    active_smoke.add_argument("--horizon", type=int, default=3)
    active_smoke.add_argument("--seed", type=int, default=7)
    active_smoke.add_argument("--output", default="outputs/active/smoke.json")
    active_smoke.set_defaults(function=_active_smoke)
    active_run = subparsers.add_parser(
        "active-run", help="Run active benchmark scenarios with plug-in algorithms"
    )
    active_run.add_argument("config")
    active_run.add_argument(
        "--algorithm",
        action="append",
        required=True,
        help="random, uniform-incenter, genious-pedro, uniform-online-samd, diffusion (nested-langevin), or name=python.module:factory",
    )
    active_run.add_argument("--output", default="outputs/active/benchmark")
    active_run.add_argument("--limit", type=int)
    active_run.add_argument("--respect-stop", action="store_true")
    active_run.add_argument("--continue-on-error", action="store_true")
    active_run.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate final angular error and hidden-query normalized regret",
    )
    active_run.add_argument("--test-queries", type=int, default=128)
    active_run.add_argument("--evaluation-seed", type=int, default=0)
    active_run.add_argument(
        "--no-zero-regret-stop",
        action="store_true",
        help="Disable the default external zero-regret stopping rule",
    )
    active_run.add_argument("--stop-regret-tolerance", type=float, default=1e-8)
    active_run.add_argument("--minimum-stop-time", type=int, default=1)
    active_run.add_argument(
        "--evaluation-trajectory",
        action="store_true",
        help="Also evaluate regret after every time step",
    )
    active_run.set_defaults(function=_active_run)
    active_research = subparsers.add_parser(
        "active-research",
        help="Run the compact hard active inverse-optimization research protocol",
    )
    active_research.add_argument(
        "--algorithm",
        action="append",
        required=True,
        help="random, uniform-incenter, genious-pedro, uniform-online-samd, diffusion (nested-langevin), or name=python.module:factory",
    )
    active_research.add_argument("--output", default="outputs/active/research")
    active_research.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    active_research.add_argument("--horizon", type=int, default=40)
    active_research.add_argument("--candidates", type=int, default=64)
    active_research.add_argument("--validation-queries", type=int, default=64)
    active_research.add_argument("--test-queries", type=int, default=128)
    active_research.add_argument("--fixed-horizon", action="store_true",
                                 help="Run all algorithms to T, including clean scenarios")
    active_research.add_argument("--angular-threshold", type=float, default=5.0,
                                 help="Angular-error threshold in degrees for first/stable recovery")
    active_research.add_argument("--continue-on-error", action="store_true")
    active_research.set_defaults(function=_active_research)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.function(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
