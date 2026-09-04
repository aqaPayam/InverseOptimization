import json
from dataclasses import replace

import numpy as np
import pytest

from invoptlab.active import (
    ActiveInverseEnvironment, ActiveScenarioConfig, DecisionSpaceConfig,
    GeniousPedroAlgorithm, NestedLangevinConfig, PedroAlgorithm, QuerySpaceConfig,
    ScoreBaseAlgorithm, StructuredBinaryDecisionSpace, UniformRandomIncenterAlgorithm,
    UniformOnlineSAMDAlgorithm, build_pedro_score_scenarios,
    run_four_algorithm_design, run_pedro_score_design, run_three_algorithm_design,
)
from invoptlab.exceptions import ValidationError


def test_two_named_algorithms_have_the_correct_distinct_estimators():
    pedro, score = PedroAlgorithm(), ScoreBaseAlgorithm()
    assert isinstance(pedro, UniformRandomIncenterAlgorithm)
    assert pedro.name == "Pedro algorithm"
    assert score.name == "Score base model"
    assert score.config.point_estimate == "mean" and score.config.theta_samples == 16
    assert score.config.query_policy == "disagreement"
    for change in ({"point_estimate": "first"}, {"query_policy": "uniform"}):
        with pytest.raises(ValidationError):
            ScoreBaseAlgorithm(replace(score.config, **change))


def _genious_context(*, repeats=True):
    root_two = np.sqrt(2.)
    scenario = ActiveScenarioConfig(
        name="genious-formula-unit-test", dimension=2, horizon=3, seed=4,
        true_theta=[.8, .6],
        decision_space=DecisionSpaceConfig(kind="fixed_cardinality", cardinality=1),
        query_space=QuerySpaceConfig(kind="explicit", allow_repeated_queries=repeats,
            candidates=[[1.,0.],[0.,1.],[1/root_two,1/root_two]]))
    return ActiveInverseEnvironment(scenario).algorithm_context()


def test_genious_pedro_uses_exact_minimum_normalized_margin():
    algorithm = GeniousPedroAlgorithm()
    algorithm.reset(_genious_context(), np.random.default_rng(12))
    algorithm._estimate = np.array([.8, .6])
    algorithm.estimate_status_ = "valid"
    algorithm.failure_reason_ = None
    action = algorithm.propose((object(),))
    assert algorithm.name == "Genious Pedro"
    assert action.diagnostics["query_rule"] == "minimum-normalized-decision-margin"
    assert action.diagnostics["candidate_index"] == 2
    np.testing.assert_allclose(
        action.diagnostics["candidate_margins"], [.8, .6, np.sqrt(.02)])
    assert action.diagnostics["selected_margin"] == pytest.approx(np.sqrt(.02))
    np.testing.assert_array_equal(action.diagnostics["predicted_decision"], [0.,1.])
    np.testing.assert_array_equal(action.diagnostics["nearest_alternative"], [1.,0.])


def test_genious_pedro_uniform_initial_and_invalid_fallbacks_are_explicit():
    algorithm = GeniousPedroAlgorithm()
    algorithm.reset(_genious_context(repeats=False), np.random.default_rng(8))
    first = algorithm.propose(())
    first_index = first.diagnostics["candidate_index"]
    assert first.diagnostics["query_rule"] == "uniform-random-fallback"
    assert "D_0 is empty" in first.diagnostics["fallback_reason"]
    assert first.diagnostics["selected_margin"] is None
    assert first_index not in algorithm._available_queries

    algorithm._estimate = np.zeros(2)
    algorithm.estimate_status_ = "degenerate_cone"
    failed = algorithm.propose((object(),))
    assert failed.diagnostics["query_rule"] == "uniform-random-fallback"
    assert "degenerate_cone" in failed.diagnostics["fallback_reason"]
    assert failed.diagnostics["candidate_index"] != first_index
    assert failed.diagnostics["candidate_margins"] is None
    np.testing.assert_array_equal(failed.theta_hat, np.zeros(2))


def test_all_eight_designs_are_paired_noisy_and_theta_independent():
    left = build_pedro_score_scenarios(seed=0)
    right = build_pedro_score_scenarios(seed=1)
    assert len(left) == len(right) == 8
    for a, b in zip(left, right):
        assert a.scenario.dimension in (4, 6) and a.scenario.horizon == 20
        assert a.scenario.expert.kind.value == "min"
        assert a.scenario.observation_noise.kind.value == "clean"
        assert a.scenario.parameter_noise.kind.value != "none"
        np.testing.assert_array_equal(a.scenario.query_space.candidates, b.scenario.query_space.candidates)
        assert not np.array_equal(a.scenario.true_theta, b.scenario.true_theta)
        for distribution, queries in a.test_queries.items():
            assert queries.shape == (120, a.scenario.dimension)
            np.testing.assert_allclose(np.linalg.norm(queries, axis=1), 1.)
            np.testing.assert_array_equal(queries, b.test_queries[distribution])
            assert np.min(np.linalg.norm(np.asarray(a.scenario.query_space.candidates)[:,None]
                                        - queries[None], axis=2)) > 1e-7
        context = ActiveInverseEnvironment(a.scenario).algorithm_context()
        assert "test_queries" not in context.public_environment
        assert "true_theta" not in context.public_environment
    for noisy in (left[6], left[7]):
        np.testing.assert_array_equal(noisy.scenario.true_theta, left[3].scenario.true_theta)
        np.testing.assert_array_equal(noisy.scenario.query_space.candidates,
                                      left[3].scenario.query_space.candidates)
        np.testing.assert_array_equal(noisy.test_queries["ordinary"], left[3].test_queries["ordinary"])
    assert set(left[0].test_queries) == set(left[2].test_queries) == {"ordinary", "balanced"}
    assert np.bincount(left[0].candidate_groups).tolist() == [27,27,27,27,12]
    assert np.bincount(left[2].candidate_groups).tolist() == [48,48,24]


def test_small_knapsack_cached_min_is_exact_and_lexicographic():
    space = StructuredBinaryDecisionSpace(6, C_ub=[[1,2,2,3,3,4]], r_ub=[6])
    rng = np.random.default_rng(11)
    costs = np.vstack([rng.normal(size=(24,6)), np.zeros((1,6))])
    decisions = space.min_decision_batch(costs, rng)
    feasible = np.asarray(space.enumerate_decisions())
    for cost, decision in zip(costs, decisions):
        assert space.contains(decision)
        assert cost @ decision == pytest.approx(np.min(feasible @ cost))
        assert cost @ decision == pytest.approx(cost @ space._solve(cost), abs=1e-8)
        np.testing.assert_array_equal(decision, space.min_decision(cost, rng))
    np.testing.assert_array_equal(decisions[-1], np.zeros(6))
    large = StructuredBinaryDecisionSpace(13, C_ub=[np.ones(13)], r_ub=[5])
    assert large._min_decisions is None


def test_comparison_small_all_families_and_cache_contract(tmp_path):
    cfg = NestedLangevinConfig(theta_samples=4, gibbs_sweeps=1,
        conditional_slice_steps=1, target_slice_steps=2, record_chain_trace=False)
    for design in build_pedro_score_scenarios(horizon=2):
        messages = []
        runs = run_pedro_score_design(design, tmp_path, score_config=cfg, progress=messages.append)
        assert [r["algorithm_name"] for r in runs] == ["Pedro algorithm", "Score base model"]
        for run in runs:
            assert len(run["records"]) == 2
            assert not run["metadata"]["external_stopping_enabled"]
            assert set(run["metadata"]["evaluations_by_distribution"]) == set(design.test_queries)
            json.dumps(run, allow_nan=False)
        for record in runs[0]["records"]:
            assert "incenter_radius" in record["update_diagnostics"]
            assert "theta_samples" not in record["update_diagnostics"]
        for record in runs[1]["records"]:
            np.testing.assert_allclose(record["theta_hat_after"],
                np.mean(record["update_diagnostics"]["theta_samples"], axis=0))
        messages.clear()
        cached = run_pedro_score_design(design, tmp_path, score_config=cfg, progress=messages.append)
        assert all(m.startswith("CACHED") for m in messages)
        assert cached == runs
    # Coupled methods receive the same Gaussian perturbations when sigma is fixed;
    # in the query-dependent case the standardized draws are paired instead.
    for design in build_pedro_score_scenarios(horizon=2):
        runs = run_pedro_score_design(design, tmp_path, score_config=cfg, progress=lambda _:None)
        for a, b in zip(runs[0]["records"], runs[1]["records"]):
            ma, mb = a["parameter_noise_metadata"], b["parameter_noise_metadata"]
            sa, sb = ma.get("sigma", ma.get("scale")), mb.get("sigma", mb.get("scale"))
            np.testing.assert_allclose(np.array(ma["perturbation"])/sa,
                                       np.array(mb["perturbation"])/sb, atol=1e-12)


def test_three_algorithm_comparison_uses_identical_protocol_and_cache(tmp_path):
    cfg = NestedLangevinConfig(theta_samples=4, gibbs_sweeps=1,
        conditional_slice_steps=1, target_slice_steps=2, record_chain_trace=False)
    design = build_pedro_score_scenarios(horizon=2)[0]
    messages = []
    runs = run_three_algorithm_design(
        design, tmp_path, score_config=cfg, progress=messages.append
    )
    assert [run["algorithm_name"] for run in runs] == [
        "Pedro algorithm", "Genious Pedro", "Score base model"
    ]
    assert all(len(run["records"]) == 2 for run in runs)
    assert all(run["scenario"] == runs[0]["scenario"] for run in runs)
    assert all(set(run["metadata"]["evaluations_by_distribution"])
               == set(design.test_queries) for run in runs)
    assert runs[1]["records"][0]["action_diagnostics"]["query_rule"] \
        == "uniform-random-fallback"
    assert runs[1]["records"][1]["action_diagnostics"]["query_rule"] \
        == "minimum-normalized-decision-margin"
    messages.clear()
    cached = run_three_algorithm_design(
        design, tmp_path, score_config=cfg, progress=messages.append
    )
    assert cached == runs
    assert all(message.startswith("CACHED") for message in messages)

    two_runs = run_pedro_score_design(
        design, tmp_path / "two", score_config=cfg, progress=lambda _: None
    )
    for reference, expanded in zip(two_runs, (runs[0], runs[2])):
        assert reference["scenario"] == expanded["scenario"]
        assert reference["evaluation"] == expanded["evaluation"]
        assert reference["metadata"]["evaluations_by_distribution"] \
            == expanded["metadata"]["evaluations_by_distribution"]
        for left, right in zip(reference["records"], expanded["records"]):
            np.testing.assert_array_equal(left["query"], right["query"])
            np.testing.assert_array_equal(
                left["observed_decision"], right["observed_decision"]
            )
            np.testing.assert_array_equal(
                left["theta_hat_after"], right["theta_hat_after"]
            )


def test_four_algorithm_comparison_adds_uniform_online_samd(tmp_path):
    assert UniformOnlineSAMDAlgorithm().name == "Uniform Online SAMD"
    cfg = NestedLangevinConfig(theta_samples=2, gibbs_sweeps=1,
        conditional_slice_steps=1, target_slice_steps=1, record_chain_trace=False)
    design = build_pedro_score_scenarios(horizon=1)[0]
    runs = run_four_algorithm_design(
        design, tmp_path, score_config=cfg, progress=lambda _: None
    )
    assert [run["algorithm_name"] for run in runs] == [
        "Pedro algorithm", "Genious Pedro", "Score base model", "Uniform Online SAMD"
    ]
    samd = runs[-1]
    assert len(samd["records"]) == 1
    assert samd["records"][0]["action_diagnostics"]["query_rule"] == "uniform-random"
    assert samd["records"][0]["update_diagnostics"]["update_count"] == 1
    assert samd["records"][0]["update_diagnostics"]["epsilon"] == 0.0
