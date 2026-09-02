import pytest

from invoptlab.active import (
    ActiveInverseEnvironment,
    ActiveBenchmarkRunner,
    ActiveEvaluationConfig,
    ActiveResearchConfig,
    RandomActiveAlgorithm,
    RegretStoppingConfig,
    UniformRandomIncenterAlgorithm,
    build_active_research_scenarios,
    evaluate_active_run,
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
    assert "easy-independent" in families
    assert "knapsack-coupled" in families
    assert "continuous-simplex" in families
    assert "behavioral-observation-noise" in families
    assert sum(scenario.metadata["role"] == "sanity-check" for scenario in scenarios) == 2


def test_behavioral_noise_is_calibrated_and_saved_privately():
    scenarios = build_active_research_scenarios(tiny_protocol())
    scenario = next(
        item
        for item in scenarios
        if item.metadata["research_family"] == "behavioral-observation-noise"
    )
    environment = ActiveInverseEnvironment(scenario)
    calibration = environment.noise_calibration["observation_noise"]
    assert calibration["target_change_rate"] == pytest.approx(0.15)
    assert abs(calibration["achieved_change_rate"] - 0.15) <= 0.1
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


def test_incenter_reports_partial_feedback_without_usable_constraints_as_failure():
    protocol = tiny_protocol()
    scenario = next(
        item
        for item in build_active_research_scenarios(protocol)
        if item.metadata["research_family"] == "partial-feedback"
    )
    run = ActiveBenchmarkRunner(
        stopping_config=RegretStoppingConfig(enabled=False)
    ).run(scenario, UniformRandomIncenterAlgorithm(alternative_budget=16))
    evaluation = evaluate_active_run(
        run,
        ActiveEvaluationConfig(test_query_count=5, seed=9, evaluate_trajectory=True),
    )
    assert evaluation.final_status == "insufficient_information"
    assert evaluation.final_angular_error_degrees is None
    assert evaluation.final_normalized_regret is None
