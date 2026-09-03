import pytest

from invoptlab.active import (
    ActiveInverseEnvironment,
    ActiveResearchConfig,
    RandomActiveAlgorithm,
    build_active_research_scenarios,
    run_active_research_benchmark,
)


def tiny_protocol(**overrides):
    values = {
        "seeds": (0,),
        "horizon": 2,
        "candidate_count": 8,
        "validation_query_count": 4,
        "test_query_count": 5,
        "consecutive_validation_successes": 2,
    }
    values.update(overrides)
    return ActiveResearchConfig(**values)


def test_curated_suite_has_twelve_interpretable_families_per_seed():
    scenarios = build_active_research_scenarios(tiny_protocol(seeds=(2, 3)))
    assert len(scenarios) == 24
    families = {scenario.metadata["research_family"] for scenario in scenarios}
    assert len(families) == 12
    assert "geometry-cardinality-3d" in families
    assert "knapsack-d20" in families
    assert "continuous-simplex-d10" in families
    assert "knapsack-parameter-noise-moderate-d20" in families
    assert all(scenario.expert.kind.value == "min" for scenario in scenarios)
    assert all(scenario.observation_noise.kind.value == "clean" for scenario in scenarios)
    assert sum(scenario.parameter_noise.kind.value == "isotropic" for scenario in scenarios) == 8

    dimensions = {
        scenario.metadata["research_family"]: scenario.dimension
        for scenario in scenarios
        if scenario.seed == 2
    }
    assert dimensions["geometry-cardinality-3d"] == 3
    assert dimensions["cardinality-balanced-d20"] == 20
    assert dimensions["dag-path-d18"] == 18
    assert dimensions["continuous-simplex-d10"] == 10


def test_behavioral_parameter_noise_is_calibrated_and_saved_privately():
    scenarios = build_active_research_scenarios(tiny_protocol())
    scenario = next(
        item
        for item in scenarios
        if item.metadata["research_family"] == "cardinality-parameter-noise-mild-d20"
    )
    environment = ActiveInverseEnvironment(scenario)
    calibration = environment.noise_calibration["parameter_noise"]
    assert calibration["target_change_rate"] == pytest.approx(0.05)
    assert abs(calibration["achieved_change_rate"] - 0.05) <= 0.05
    assert calibration["effective_strength"] > 0
    context = environment.algorithm_context()
    assert "noise_calibration" not in context.public_environment
    assert "true_theta" not in context.public_environment


def test_research_protocol_runs_stochastic_cases_to_horizon_and_uses_independent_test():
    protocol = tiny_protocol()
    result, summary = run_active_research_benchmark(
        {"random": lambda: RandomActiveAlgorithm()},
        protocol,
        fail_fast=False,
    )
    assert len(result.runs) == 12
    assert not result.failed_runs
    assert summary["group_count"] == 12
    stochastic = [run for run in result.runs if run.scenario.metadata["stochastic"]]
    assert stochastic
    assert all(len(run.records) == protocol.horizon for run in stochastic)
    assert all(run.evaluation["metadata"]["independent_from_stopping"] for run in result.runs)
    assert all(len(run.evaluation["normalized_regret_history"]) == len(run.records) for run in result.runs)
    assert result.metadata["composite_score_created"] is False


def test_fixed_horizon_ignores_early_success_and_algorithm_stop_requests(monkeypatch):
    import numpy as np
    from invoptlab.active import (
        ActiveAction, ActiveScenarioConfig, CallbackActiveAlgorithm, QuerySpaceConfig,
    )
    from invoptlab.active import research

    scenario = ActiveScenarioConfig(
        dimension=2, horizon=3, true_theta=[1., 1.],
        query_space=QuerySpaceConfig(candidate_count=8),
        metadata={"stochastic": False, "research_family": "unit-test-clean"},
    )
    monkeypatch.setattr(research, "build_active_research_scenarios", lambda config: [scenario])
    protocol = tiny_protocol(horizon=3, fixed_horizon=True)
    result, summary = run_active_research_benchmark({
        "constant": lambda: CallbackActiveAlgorithm(
            lambda ctx, history: ActiveAction(ctx.query_candidates[0], np.ones(2), stop_requested=True)
        ),
    }, protocol)
    run = result.runs[0]
    assert len(run.records) == 3
    assert not run.stopped_early and not run.metadata["external_stopping_enabled"]
    assert result.metadata["all_runs_use_fixed_horizon"]
    assert run.evaluation["first_joint_threshold_step"] == 1
    times = summary["groups"][0]["threshold_times"]["first_joint_threshold_step"]
    assert times["mean"] == 1 and times["reached_rate"] == 1
    assert times["not_reached_count"] == 0
