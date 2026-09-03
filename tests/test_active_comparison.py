import json
from dataclasses import replace

import numpy as np
import pytest

from invoptlab.active import (
    ActiveInverseEnvironment, NestedLangevinConfig, PedroAlgorithm,
    ScoreBaseAlgorithm, StructuredBinaryDecisionSpace, UniformRandomIncenterAlgorithm,
    build_pedro_score_scenarios, run_pedro_score_design,
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
