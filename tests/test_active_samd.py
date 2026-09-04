import numpy as np
import pytest

from invoptlab.active import (
    ActiveBenchmarkRunner,
    ActiveScenarioConfig,
    AlgorithmContext,
    AlgorithmObservation,
    DecisionSpaceConfig,
    IndependentBinaryDecisionSpace,
    OnlineSAMDConfig,
    PublicDecisionProblem,
    QuerySpaceConfig,
    UniformOnlineSAMDAlgorithm,
)
from invoptlab.exceptions import ValidationError


def context() -> AlgorithmContext:
    return AlgorithmContext(
        dimension=2,
        horizon=3,
        query_candidates=np.eye(2),
        decision_problem=PublicDecisionProblem(IndependentBinaryDecisionSpace(2)),
        seed=4,
        scenario_name="samd-unit",
        public_environment={"query_space": {"allow_repeated_queries": True}},
    )


def test_online_samd_configuration_rejects_invalid_values():
    with pytest.raises(ValidationError):
        OnlineSAMDConfig(learning_rate=0)
    with pytest.raises(ValidationError):
        OnlineSAMDConfig(l1_radius=-1)
    with pytest.raises(ValidationError):
        OnlineSAMDConfig(margin_scale=-1)


def test_uniform_online_samd_uses_exact_signed_asl_update():
    algorithm = UniformOnlineSAMDAlgorithm(OnlineSAMDConfig(l1_radius=1.0))
    algorithm.reset(context(), np.random.default_rng(9))
    np.testing.assert_array_equal(algorithm.current_estimate(), np.zeros(2))

    algorithm.observe(AlgorithmObservation(
        step=1,
        query=np.array([1.0, 0.0]),
        observed_decision=np.array([0.0, 0.0]),
    ))

    estimate = algorithm.current_estimate()
    assert estimate[0] > 0
    assert estimate[1] == pytest.approx(0.0, abs=1e-14)
    assert np.linalg.norm(estimate, ord=1) <= 1.0 + 1e-12
    diagnostics = algorithm.diagnostics()
    np.testing.assert_array_equal(diagnostics["competitor"], np.ones(2))
    np.testing.assert_array_equal(diagnostics["subgradient"], [-1.0, 0.0])
    assert diagnostics["loss_augmented_oracle"] == "exact-enumeration"
    assert diagnostics["epsilon"] == 0.0
    assert diagnostics["estimate_status"] == "valid"


def test_online_samd_survives_conflicting_observations_and_queries_reproduce():
    left = UniformOnlineSAMDAlgorithm()
    right = UniformOnlineSAMDAlgorithm()
    indices = []
    for algorithm in (left, right):
        algorithm.reset(context(), np.random.default_rng(12))
        first = algorithm.propose(())
        algorithm.observe(AlgorithmObservation(
            1, np.array([1.0, 0.0]), np.array([0.0, 0.0])
        ))
        second = algorithm.propose(())
        algorithm.observe(AlgorithmObservation(
            2, np.array([1.0, 0.0]), np.array([1.0, 0.0])
        ))
        indices.append((
            first.diagnostics["candidate_index"],
            second.diagnostics["candidate_index"],
        ))
    assert indices[0] == indices[1]
    assert np.all(np.isfinite(left.current_estimate()))
    assert left.diagnostics()["update_count"] == 2
    assert left.diagnostics()["estimate_status"] == "valid"


def test_online_samd_runs_one_update_per_active_observation():
    scenario = ActiveScenarioConfig(
        name="samd-runner-sanity",
        dimension=3,
        horizon=4,
        seed=3,
        true_theta=[-1.0, 0.4, 0.8],
        decision_space=DecisionSpaceConfig(kind="fixed_cardinality", cardinality=1),
        query_space=QuerySpaceConfig(
            kind="explicit",
            candidates=np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ]),
        ),
    )
    result = ActiveBenchmarkRunner().run(scenario, UniformOnlineSAMDAlgorithm())
    assert result.error is None
    assert len(result.records) == 4
    assert all(record.action_diagnostics["query_rule"] == "uniform-random"
               for record in result.records)
    assert [record.update_diagnostics["update_count"] for record in result.records] \
        == [1, 2, 3, 4]
    assert all(np.all(np.isfinite(record.theta_hat_after)) for record in result.records)
    assert all(record.update_diagnostics["split_l1_norm"] <= np.sqrt(3) + 1e-12
               for record in result.records)
