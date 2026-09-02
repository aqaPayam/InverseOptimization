import numpy as np
import pytest

from invoptlab.active import (
    ActiveBenchmarkRunner,
    ActiveScenarioConfig,
    AlgorithmContext,
    AlgorithmObservation,
    ContinuousPolytopeDecisionSpace,
    DAGPathDecisionSpace,
    DecisionSpaceConfig,
    FixedCardinalityDecisionSpace,
    IndependentBinaryDecisionSpace,
    PublicDecisionProblem,
    QuerySpaceConfig,
    RegretStoppingConfig,
    StructuredBinaryDecisionSpace,
    UniformRandomIncenterAlgorithm,
)


def make_context(decision_space, candidates):
    candidates = np.asarray(candidates, dtype=float)
    return AlgorithmContext(
        dimension=candidates.shape[1],
        horizon=10,
        query_candidates=candidates,
        decision_problem=PublicDecisionProblem(decision_space),
        seed=3,
        scenario_name="unit-test",
        public_environment={"query_space": {"allow_repeated_queries": True}},
    )


def test_uniform_incenter_recovers_first_quadrant_incenter_from_public_y():
    root_two = np.sqrt(2.0)
    query = np.asarray([1.0, 1.0]) / root_two
    context = make_context(
        IndependentBinaryDecisionSpace(2),
        [query, np.asarray([1.0, -1.0]) / root_two],
    )
    algorithm = UniformRandomIncenterAlgorithm()
    algorithm.reset(context, np.random.default_rng(11))
    algorithm.observe(
        AlgorithmObservation(
            step=1,
            query=query,
            observed_decision=np.asarray([0, 0]),
        )
    )
    np.testing.assert_allclose(
        algorithm.current_estimate(),
        np.asarray([1.0, 1.0]) / root_two,
        atol=2e-4,
    )
    assert algorithm.incenter_radius_ == pytest.approx(1 / root_two, abs=2e-4)
    assert algorithm.constraints_.shape == (2, 2)
    assert algorithm.constraint_sources_[0]["exact"] is True
    assert algorithm.diagnostics()["estimate_status"] == "valid"
    assert algorithm.diagnostics()["constraint_normals"].shape == (2, 2)


def test_uniform_query_selection_is_reproducible_and_uses_candidate_pool():
    candidates = np.eye(3)
    context = make_context(IndependentBinaryDecisionSpace(3), candidates)
    left = UniformRandomIncenterAlgorithm()
    right = UniformRandomIncenterAlgorithm()
    left.reset(context, np.random.default_rng(17))
    right.reset(context, np.random.default_rng(17))
    left_queries = [left.propose(()).query for _ in range(12)]
    right_queries = [right.propose(()).query for _ in range(12)]
    np.testing.assert_allclose(left_queries, right_queries)
    assert all(any(np.array_equal(query, item) for item in candidates) for query in left_queries)


def test_partial_y_is_skipped_without_using_hidden_x():
    context = make_context(IndependentBinaryDecisionSpace(2), np.eye(2))
    algorithm = UniformRandomIncenterAlgorithm()
    algorithm.reset(context, np.random.default_rng(5))
    initial = algorithm.current_estimate()
    algorithm.observe(
        AlgorithmObservation(
            step=1,
            query=np.asarray([1.0, 0.0]),
            observed_decision=np.asarray([0, 0]),
            observation_mask=np.asarray([1, 0]),
        )
    )
    np.testing.assert_allclose(algorithm.current_estimate(), initial)
    assert algorithm.constraints_.shape == (0, 2)
    assert algorithm.constraint_sources_[0]["skipped_reason"] is not None
    assert algorithm.diagnostics()["estimate_status"] == "insufficient_information"


def test_contradictory_noisy_observations_produce_zero_radius_not_a_crash():
    context = make_context(IndependentBinaryDecisionSpace(1), [[1.0]])
    algorithm = UniformRandomIncenterAlgorithm()
    algorithm.reset(context, np.random.default_rng(8))
    algorithm.observe(
        AlgorithmObservation(step=1, query=np.asarray([1.0]), observed_decision=np.asarray([0]))
    )
    algorithm.observe(
        AlgorithmObservation(step=2, query=np.asarray([1.0]), observed_decision=np.asarray([1]))
    )
    assert algorithm.incenter_radius_ == pytest.approx(0.0, abs=2e-5)
    assert np.linalg.norm(algorithm.current_estimate()) <= 2e-4
    assert algorithm.diagnostics()["estimate_status"] == "degenerate_cone"


def test_coupled_continuous_space_marks_forward_oracle_cuts_as_approximate():
    space = ContinuousPolytopeDecisionSpace(
        2,
        A=[[1.0, 1.0]],
        b=[0.5],
        lower=[-1.0, -1.0],
        upper=[1.0, 1.0],
    )
    public = PublicDecisionProblem(space)
    batch = public.consistency_normals(
        np.ones(2) / np.sqrt(2),
        np.asarray([-1.0, -1.0]),
        None,
        np.ones(2) / np.sqrt(2),
        np.random.default_rng(4),
        alternative_budget=8,
    )
    assert batch.exact is False
    assert batch.method == "forward-oracle-cuts"
    assert batch.normals.shape[1] == 2


@pytest.mark.parametrize(
    ("space", "observed"),
    [
        (IndependentBinaryDecisionSpace(4), np.asarray([0, 1, 0, 1])),
        (FixedCardinalityDecisionSpace(4, 2), np.asarray([1, 1, 0, 0])),
        (ContinuousPolytopeDecisionSpace(4), np.asarray([-1.0, 1.0, -1.0, 1.0])),
        (
            DAGPathDecisionSpace(
                [(0, 1), (1, 3), (0, 2), (2, 3)],
                0,
                3,
            ),
            np.asarray([1, 1, 0, 0]),
        ),
        (
            StructuredBinaryDecisionSpace(
                4,
                A_eq=[[1, 1, 1, 1]],
                b_eq=[2],
            ),
            np.asarray([1, 1, 0, 0]),
        ),
    ],
)
def test_public_constraint_generation_is_exact_for_supported_spaces(space, observed):
    public = PublicDecisionProblem(space)
    batch = public.consistency_normals(
        np.ones(4) / 2,
        observed,
        None,
        np.ones(4) / 2,
        np.random.default_rng(2),
    )
    assert batch.exact is True
    assert batch.skipped_reason is None
    assert batch.normals.shape[1] == 4
    assert batch.normals.shape[0] > 0
    assert "theta" not in public.description()


def test_algorithm_integrates_with_runner_without_latent_inputs():
    scenario = ActiveScenarioConfig(
        name="uniform-incenter-integration-test",
        dimension=5,
        horizon=2,
        seed=6,
        decision_space=DecisionSpaceConfig(kind="independent_binary"),
        query_space=QuerySpaceConfig(kind="balanced", candidate_count=8),
    )
    algorithm = UniformRandomIncenterAlgorithm()
    result = ActiveBenchmarkRunner(
        stopping_config=RegretStoppingConfig(enabled=False)
    ).run(scenario, algorithm)
    assert len(result.records) == 2
    assert algorithm.constraints_.shape[0] > 0
    assert len(algorithm.incenter_history_) == 2
    assert all(
        record.action_diagnostics["query_rule"] == "uniform-random"
        for record in result.records
    )
    assert result.records[-1].update_diagnostics["constraint_count"] > 0
    assert "incenter_radius" in result.records[-1].update_diagnostics
