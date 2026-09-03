import copy

import numpy as np
import pytest
from invoptlab.exceptions import ValidationError

from invoptlab.active import (
    ActiveBenchmarkResult,
    ActiveEvaluationConfig,
    ActiveRunResult,
    ActiveScenarioConfig,
    ActiveStepRecord,
    IndependentBinaryDecisionSpace,
    angular_error_degrees,
    evaluate_active_benchmark,
    evaluate_active_run,
    normalized_test_regret,
    sample_uniform_test_queries,
)


def test_angular_error_preserves_sign_and_marks_zero_estimate_undefined():
    truth = np.asarray([1.0, 0.0])
    assert angular_error_degrees(truth, truth) == pytest.approx((0.0, True))
    assert angular_error_degrees(-truth, truth) == pytest.approx((180.0, True))
    assert angular_error_degrees([0.0, 1.0], truth) == pytest.approx((90.0, True))
    assert angular_error_degrees([0.0, 0.0], truth) == (None, False)


def test_hidden_uniform_test_queries_are_reproducible_and_unit_norm():
    first = sample_uniform_test_queries(5, 32, scenario_seed=7, evaluation_seed=3)
    second = sample_uniform_test_queries(5, 32, scenario_seed=7, evaluation_seed=3)
    different = sample_uniform_test_queries(5, 32, scenario_seed=7, evaluation_seed=4)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
    assert not np.allclose(first, different)


def test_normalized_regret_is_zero_for_truth_and_one_for_opposite_direction():
    truth = np.ones(2) / np.sqrt(2)
    queries = np.eye(2)
    space = IndependentBinaryDecisionSpace(2)
    correct, correct_zero_rate, _ = normalized_test_regret(
        truth,
        truth,
        queries,
        space,
    )
    opposite, opposite_zero_rate, values = normalized_test_regret(
        -truth,
        truth,
        queries,
        space,
    )
    assert correct == pytest.approx(0.0)
    assert correct_zero_rate == pytest.approx(1.0)
    assert opposite == pytest.approx(1.0)
    assert opposite_zero_rate == pytest.approx(0.0)
    np.testing.assert_allclose(values, 1.0)


def make_run(name="test-algorithm"):
    truth = np.ones(2) / np.sqrt(2)
    scenario = ActiveScenarioConfig(
        name="evaluation-unit-test",
        dimension=2,
        horizon=2,
        seed=12,
    )
    estimates = [np.asarray([1.0, 0.0]), truth]
    records = []
    for step, estimate in enumerate(estimates, start=1):
        records.append(
            ActiveStepRecord(
                step=step,
                query=np.eye(2)[step - 1],
                theta_hat_before=estimate.copy(),
                theta_hat_after=estimate.copy(),
                true_theta=truth.copy(),
                expert_parameter=truth.copy(),
                true_decision=np.zeros(2, dtype=int),
                observed_decision=np.zeros(2, dtype=int),
                observation_mask=None,
                objective_value=0.0,
                stop_requested=False,
            )
        )
    return ActiveRunResult(
        scenario=scenario,
        algorithm_name=name,
        seed=12,
        true_theta=truth,
        records=records,
        runtime_seconds=0.2,
    )


def test_run_evaluation_attaches_final_and_optional_trajectory_metrics():
    run = make_run()
    result = evaluate_active_run(
        run,
        ActiveEvaluationConfig(
            test_query_count=24,
            seed=9,
            evaluate_trajectory=True,
        ),
    )
    assert result.final_angular_error_degrees == pytest.approx(0.0)
    assert result.final_normalized_regret == pytest.approx(0.0)
    assert result.final_zero_regret_rate == pytest.approx(1.0)
    assert result.final_estimate_valid is True
    assert len(result.angular_error_history_degrees) == 2
    assert len(result.normalized_regret_history) == 2
    assert result.metadata["test_queries_hidden_from_algorithm"] is True
    assert run.metadata["evaluation_applied"] is True
    assert run.evaluation == result.to_dict()


def test_failed_estimate_returns_status_and_undefined_metrics():
    run = make_run()
    run.records[-1].theta_hat_after = np.zeros(2)
    run.records[-1].update_diagnostics = {
        "estimate_status": "degenerate_cone",
        "failure_reason": "the consistency cone has no valid nonzero parameter direction",
    }
    result = evaluate_active_run(
        run,
        ActiveEvaluationConfig(test_query_count=12, seed=4, evaluate_trajectory=True),
    )
    assert result.final_estimate_valid is False
    assert result.final_status == "degenerate_cone"
    assert result.final_failure_reason is not None
    assert result.final_angular_error_degrees is None
    assert result.final_normalized_regret is None
    assert result.final_zero_regret_rate is None
    assert result.angular_error_history_degrees[-1] is None
    assert result.normalized_regret_history[-1] is None
    assert run.metadata["algorithm_failed"] is True


def test_benchmark_evaluation_summarizes_algorithms_without_a_composite_score():
    first = make_run("algorithm-a")
    second = copy.deepcopy(first)
    second.algorithm_name = "algorithm-b"
    benchmark = ActiveBenchmarkResult([first, second], metadata={"scoring_applied": False})
    summary = evaluate_active_benchmark(
        benchmark,
        ActiveEvaluationConfig(test_query_count=12, seed=2),
    )
    assert summary["evaluated_run_count"] == 2
    assert set(summary["algorithms"]) == {"algorithm-a", "algorithm-b"}
    assert summary["algorithms"]["algorithm-a"]["mean_final_normalized_regret"] == 0
    assert benchmark.metadata["evaluation_applied"] is True
    assert benchmark.metadata["scoring_applied"] is False


def test_first_angular_and_regret_times_are_distinct_and_joint_is_simultaneous():
    run = make_run()
    run.records[0].theta_hat_after = np.array([1., 0.1])  # Correct signs, wrong direction.
    result = evaluate_active_run(run, ActiveEvaluationConfig(
        test_query_count=12, evaluate_trajectory=True, learning_angular_threshold_degrees=5.,
    ))
    assert result.first_threshold_step == 1
    assert result.stable_threshold_step == 1
    assert result.first_angular_threshold_step == 2
    assert result.stable_angular_threshold_step == 2
    assert result.first_joint_threshold_step == 2
    assert result.stable_joint_threshold_step == 2


def test_invalid_later_estimate_breaks_stable_recovery_without_erasing_first_hit():
    run = make_run()
    run.records[0].theta_hat_after = run.true_theta.copy()
    run.records[1].theta_hat_after = np.zeros(2)
    result = evaluate_active_run(run, ActiveEvaluationConfig(evaluate_trajectory=True))
    assert result.first_joint_threshold_step == 1
    assert result.first_angular_threshold_step == 1
    assert result.stable_joint_threshold_step is None
    assert result.stable_angular_threshold_step is None
    assert result.stable_threshold_step is None


def test_no_recovery_is_none_not_the_horizon_and_final_only_does_not_invent_times():
    run = make_run()
    for record in run.records:
        record.theta_hat_after = -run.true_theta
    result = evaluate_active_run(run, ActiveEvaluationConfig(evaluate_trajectory=True))
    assert result.first_joint_threshold_step is None
    assert result.first_angular_threshold_step is None
    final_only = evaluate_active_run(make_run(), ActiveEvaluationConfig())
    assert final_only.first_joint_threshold_step is None
    assert final_only.first_angular_threshold_step is None


@pytest.mark.parametrize("threshold", [-1., 181., float("nan")])
def test_angular_recovery_threshold_validation(threshold):
    with pytest.raises(ValidationError):
        ActiveEvaluationConfig(learning_angular_threshold_degrees=threshold)
