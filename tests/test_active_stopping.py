import numpy as np

from invoptlab.active import (
    ActiveAction,
    ActiveAlgorithm,
    ActiveBenchmarkRunner,
    ActiveEvaluationConfig,
    ActiveScenarioConfig,
    QuerySpaceConfig,
    RegretStoppingConfig,
    RegretStoppingRule,
    evaluate_active_run,
)


class ConstantEstimateAlgorithm(ActiveAlgorithm):
    name = "constant-estimate-test-helper"

    def __init__(self, estimate):
        self.estimate = np.asarray(estimate, dtype=float)

    def reset(self, context, rng):
        del rng
        self.context = context

    def propose(self, history):
        index = len(history) % len(self.context.query_candidates)
        return ActiveAction(
            query=self.context.query_candidates[index],
            theta_hat=self.estimate,
        )

    def observe(self, observation):
        del observation

    def current_estimate(self):
        return self.estimate.copy()


def make_scenario(horizon=4):
    return ActiveScenarioConfig(
        name="external-stopping-unit-test",
        dimension=2,
        horizon=horizon,
        seed=31,
        true_theta=[0.8, -0.6],
        query_space=QuerySpaceConfig(kind="dense", candidate_count=8),
    )


def test_zero_hidden_test_regret_stops_at_first_successful_time():
    config = RegretStoppingConfig(test_query_count=64, seed=9)
    run = ActiveBenchmarkRunner(stopping_config=config).run(
        make_scenario(),
        ConstantEstimateAlgorithm([0.8, -0.6]),
    )
    assert len(run.records) == 1
    assert run.stopped_early is True
    assert run.metadata["stopping_time"] == 1
    assert run.metadata["stopping_criterion_met"] is True
    assert run.metadata["stopping_reason"] == "zero hidden-test regret"
    assert run.records[0].benchmark_stop_requested is True
    assert run.records[0].stopping_diagnostics["mean_normalized_regret"] == 0

    evaluation = evaluate_active_run(
        run,
        ActiveEvaluationConfig(test_query_count=64, seed=9),
    )
    assert evaluation.final_normalized_regret == 0


def test_nonzero_regret_continues_to_fixed_horizon():
    run = ActiveBenchmarkRunner(
        stopping_config=RegretStoppingConfig(test_query_count=64, seed=9)
    ).run(
        make_scenario(horizon=3),
        ConstantEstimateAlgorithm([-0.8, 0.6]),
    )
    assert len(run.records) == 3
    assert run.stopped_early is False
    assert run.metadata["stopping_time"] == 3
    assert run.metadata["stopping_criterion_met"] is False
    assert all(not record.benchmark_stop_requested for record in run.records)
    assert run.records[-1].stopping_diagnostics["mean_normalized_regret"] > 0


def test_minimum_stop_time_is_respected():
    run = ActiveBenchmarkRunner(
        stopping_config=RegretStoppingConfig(
            test_query_count=32,
            seed=2,
            minimum_steps=2,
        )
    ).run(
        make_scenario(horizon=4),
        ConstantEstimateAlgorithm([0.8, -0.6]),
    )
    assert len(run.records) == 2
    assert run.records[0].benchmark_stop_requested is False
    assert run.records[1].benchmark_stop_requested is True


def test_stopping_test_queries_are_reproducible_and_hidden_from_algorithm():
    scenario = make_scenario()
    algorithm = ConstantEstimateAlgorithm([-0.8, 0.6])
    runner = ActiveBenchmarkRunner(
        stopping_config=RegretStoppingConfig(test_query_count=16, seed=7)
    )
    run = runner.run(scenario, algorithm)
    assert "true_theta" not in algorithm.context.public_environment
    assert "test_queries" not in algorithm.context.public_environment

    from invoptlab.active import ActiveInverseEnvironment

    environment = ActiveInverseEnvironment(scenario)
    left = RegretStoppingRule(runner.stopping_config)
    right = RegretStoppingRule(runner.stopping_config)
    left.reset(scenario, environment.theta_true, environment.decision_space)
    right.reset(scenario, environment.theta_true, environment.decision_space)
    np.testing.assert_allclose(left.test_queries, right.test_queries)
    assert run.metadata["external_stopping_enabled"] is True
