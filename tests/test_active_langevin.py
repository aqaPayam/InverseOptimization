import json
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from invoptlab.active import (
    ActiveBenchmarkRunner, ActiveEvaluationConfig, ActiveScenarioConfig,
    AlgorithmContext, AlgorithmObservation, ContinuousPolytopeDecisionSpace,
    DAGPathDecisionSpace, DecisionSpaceConfig, FixedCardinalityDecisionSpace,
    GaussianSmoothedSampler, IndependentBinaryDecisionSpace, InverseLossTarget,
    NestedLangevinActiveAlgorithm, NestedLangevinConfig, PublicDecisionProblem,
    QuerySpaceConfig, RegretStoppingConfig, StructuredBinaryDecisionSpace,
    disagreement_score, evaluate_active_run, load_algorithm_factory,
)
from invoptlab.exceptions import CapabilityError, SolverError, ValidationError
from invoptlab.active.langevin import InnerChainResult


def tiny_config(**kwargs):
    return replace(NestedLangevinConfig(
        sampler="projected_langevin", point_estimate="first", query_tie_breaking="first",
        tau_schedule=(0.2,), inner_step_sizes=(0.01,), outer_step_sizes=(0.02,),
        inner_steps=4, inner_burn_in=1, inner_thinning=2, outer_steps=2,
        theta_samples=3,
    ), **kwargs)


def context(space=None, candidates=None, repeats=True):
    space = space or IndependentBinaryDecisionSpace(2)
    candidates = np.eye(space.dimension) if candidates is None else np.asarray(candidates)
    return AlgorithmContext(space.dimension, 3, candidates, PublicDecisionProblem(space),
                            0, "test", {"query_space": {"allow_repeated_queries": repeats}})


def binary_forward(theta, s):
    return (theta * s < 0).astype(float)


@pytest.mark.parametrize("changes", [
    {"beta": 0}, {"beta": np.nan}, {"bound": np.inf}, {"bound": -1},
    {"theta_samples": 1}, {"theta_samples": 2.5}, {"inner_steps": 0},
    {"inner_burn_in": 4}, {"inner_thinning": 0}, {"outer_steps": -1},
    {"tau_schedule": (0.2, 0.2)}, {"tau_schedule": ()},
    {"inner_step_sizes": (0.1, 0.01)}, {"outer_step_sizes": (-1,)},
    {"parameter_domain": "sphere"}, {"workers": 0}, {"warm_start_renoise_std": -1},
])
def test_invalid_config(changes):
    with pytest.raises(ValidationError):
        tiny_config(**changes)


def test_loss_score_sign_full_sum_and_observed_y_copies():
    query, y = np.array([2., -1.]), np.array([0., 1.])
    obs = AlgorithmObservation(1, query, y)
    target = InverseLossTarget(2, 3., binary_forward, [obs, obs])
    query[:] = 0
    y[:] = 0
    theta = np.array([-0.4, -0.7])
    loss, grad = target.loss_and_subgradient(theta)
    assert loss == pytest.approx(3.)
    np.testing.assert_allclose(grad, [-4., -2.])
    np.testing.assert_allclose(target.target_score(theta), [12., 6.])
    assert target.forward_calls == 4  # Two full-data evaluations, two solves each.
    epsilon = 1e-6
    numeric = [(target.loss(theta + epsilon * e) - target.loss(theta - epsilon * e))
               / (2 * epsilon) for e in np.eye(2)]
    np.testing.assert_allclose(grad, numeric, atol=1e-8)
    assert target.loss(np.zeros(2)) == 0


def test_ties_produce_valid_subgradient_without_random_expert():
    target = InverseLossTarget(1, 1., binary_forward,
                               [AlgorithmObservation(1, np.ones(1), np.ones(1))])
    loss, grad = target.loss_and_subgradient(np.zeros(1))
    for value in (-1., 0., 1.):
        assert target.loss(np.array([value])) >= loss + grad[0] * value


class FixedNoise:
    def __init__(self, values):
        self.values = iter(values)

    def normal(self, *, size, **kwargs):
        return np.full(size, next(self.values))


def test_inner_update_formula_projection_fixed_u_and_retention():
    config = tiny_config(inner_steps=3, inner_burn_in=0, inner_thinning=2)
    target = InverseLossTarget(1, 2., binary_forward,
                               [AlgorithmObservation(1, np.ones(1), np.zeros(1))])
    sampler = GaussianSmoothedSampler(target, config, FixedNoise([1., -1., 0.]))
    u = np.array([2.])
    initial = np.array([-0.5])
    result = sampler.run_inner_chain(u, 0.2, 0.01, initial_z=initial)
    z = initial.copy()
    states = []
    for noise in [1., -1., 0.]:
        score = 2. if z[0] < 0 else 0.
        z = np.clip(z + 0.01 * (score + (u - z) / 0.2) + np.sqrt(0.02) * noise, -1, 1)
        states.append(z.copy())
    np.testing.assert_allclose(result.retained_states, np.vstack([states[0], states[2]]))
    np.testing.assert_allclose(result.mean, np.mean([states[0], states[2]], axis=0))
    np.testing.assert_array_equal(u, [2.])
    np.testing.assert_array_equal(initial, [-0.5])


def test_outer_formula_is_not_projected_and_tau_is_variance():
    sampler = GaussianSmoothedSampler(InverseLossTarget(1, 1., binary_forward),
                                       tiny_config(), FixedNoise([2.]))
    score = sampler.estimate_smoothed_score(np.array([2.]), np.array([1.]), 0.25)
    np.testing.assert_allclose(score, [-4.])
    updated = sampler.outer_update(np.array([2.]), score, 0.1)
    np.testing.assert_allclose(updated, [2 - 0.4 + np.sqrt(0.2) * 2])
    assert updated[0] > sampler.config.bound


def test_extraction_uses_final_updated_u_and_returns_retained_z(monkeypatch):
    config = tiny_config(outer_steps=1)
    sampler = GaussianSmoothedSampler(InverseLossTarget(1, 1., binary_forward),
                                       config, FixedNoise([2., 2.]))
    seen = []

    def inner(u, tau, step_size, **kwargs):
        seen.append(u.copy())
        sampler.inner_update_count += config.inner_steps
        if len(seen) == 1:
            return InnerChainResult(np.array([0.3]), np.array([0.4]), np.array([[0.2], [0.4]]))
        # The last actual state, conditional mean, and last retained state differ.
        return InnerChainResult(np.array([0.6]), np.array([0.9]), np.array([[0.4], [0.8]]))

    monkeypatch.setattr(sampler, "run_inner_chain", inner)
    result = sampler.sample()
    expected_u = 2. + 0.02 * (0.3 - 2.) / 0.2 + np.sqrt(0.04) * 2.
    assert len(seen) == 2
    np.testing.assert_allclose(seen[1], [expected_u])
    np.testing.assert_allclose(result.theta, [0.8])
    np.testing.assert_allclose(result.outer_state, [expected_u])


@pytest.mark.parametrize("domain", ["box", "ball"])
def test_projection_final_latent_extraction_and_exact_solve_count(domain):
    config = tiny_config(parameter_domain=domain, bound=0.1)
    obs = AlgorithmObservation(1, np.ones(2), np.zeros(2))
    target = InverseLossTarget(2, config.beta, binary_forward, [obs])
    sampler = GaussianSmoothedSampler(target, config, np.random.default_rng(1))
    result = sampler.sample()
    expected_scores = (len(config.tau_schedule) * config.outer_steps + 1) * config.inner_steps
    assert target.score_calls == expected_scores
    assert target.forward_calls == expected_scores
    np.testing.assert_array_equal(config.project(result.theta), result.theta)
    assert any(entry["phase"] == "extraction" for entry in result.trace)
    assert result.summary["max_outer_norm"] > config.bound
    assert result.summary["inner_projection_rate"] > 0


def test_inner_gaussian_conditional_moments_statistical_sanity():
    # No data and distant boundaries: conditional is N(u, tau). The Euler
    # variance is tau/(1-delta/(2*tau)); this checks drift and noise scaling.
    config = tiny_config(bound=100., inner_steps=6000, inner_burn_in=1000,
                         inner_thinning=1, record_chain_trace=False)
    sampler = GaussianSmoothedSampler(InverseLossTarget(1, 1., binary_forward),
                                       config, np.random.default_rng(123))
    result = sampler.run_inner_chain(np.array([0.4]), 0.25, 0.01)
    assert abs(result.mean[0] - 0.4) < 0.12
    assert abs(result.retained_states.var() - 0.25 / 0.98) < 0.06


def test_disagreement_equals_pairwise_formula():
    predictions = np.random.default_rng(3).normal(size=(7, 4))
    expected = np.mean([np.sum((a-b)**2) for a, b in combinations(predictions, 2)])
    assert disagreement_score(predictions) == pytest.approx(expected)
    assert disagreement_score(np.ones((3, 2))) == 0
    with pytest.raises(ValidationError):
        disagreement_score(np.ones((1, 2)))


def test_first_sample_estimate_but_all_samples_select_query():
    algorithm = NestedLangevinActiveAlgorithm(tiny_config(theta_samples=2))
    ctx = context(FixedCardinalityDecisionSpace(2, 1), [[1., 0.], [1., 1.]])
    algorithm.reset(ctx, np.random.default_rng(1))
    np.testing.assert_array_equal(algorithm.current_estimate(), algorithm.theta_samples_[0])
    algorithm.theta_samples_ = np.array([[1., 2.], [2., 1.]])
    algorithm._estimate = algorithm.theta_samples_[0].copy()
    action = algorithm.propose(())
    np.testing.assert_array_equal(action.theta_hat, [1., 2.])
    np.testing.assert_array_equal(action.query, [1., 1.])
    np.testing.assert_allclose(action.diagnostics["candidate_scores"], [0., 2.])
    assert action.diagnostics["candidate_forward_calls"] == 4


def test_reproducibility_across_workers_and_no_repeated_queries():
    ctx = context(repeats=False)
    algorithms = [NestedLangevinActiveAlgorithm(tiny_config(workers=w)) for w in (1, 2)]
    for algorithm in algorithms:
        algorithm.reset(ctx, np.random.default_rng(12))
        first = algorithm.propose(())
        second = algorithm.propose(())
        assert not np.array_equal(first.query, second.query)
        with pytest.raises(ValidationError, match="exhausted"):
            algorithm.propose(())
        algorithm.observe(AlgorithmObservation(1, np.ones(2), np.zeros(2)))
    np.testing.assert_array_equal(algorithms[0].theta_samples_, algorithms[1].theta_samples_)
    assert algorithms[0].forward_calls_ == algorithms[1].forward_calls_


def test_partial_infeasible_and_nonfinite_feedback_rejected():
    algorithm = NestedLangevinActiveAlgorithm(tiny_config())
    algorithm.reset(context(), np.random.default_rng(0))
    for obs, error in [
        (AlgorithmObservation(1, np.ones(2), np.zeros(2), np.array([1, 0])), CapabilityError),
        (AlgorithmObservation(1, np.ones(2), np.array([0., 0.5])), ValidationError),
        (AlgorithmObservation(1, np.ones(2), np.array([0., np.nan])), ValidationError),
    ]:
        with pytest.raises(error):
            algorithm.observe(obs)
        assert not algorithm.observations_


def test_instability_is_reported_not_clipped_or_replaced():
    sampler = GaussianSmoothedSampler(InverseLossTarget(1, 1., binary_forward),
                                       tiny_config(max_state_norm=2.), FixedNoise([100.]))
    with pytest.raises(SolverError, match="unstable"):
        sampler.outer_update(np.zeros(1), np.zeros(1), 1.)


@pytest.mark.parametrize("space", [
    IndependentBinaryDecisionSpace(2), FixedCardinalityDecisionSpace(2, 1),
    ContinuousPolytopeDecisionSpace(2, A=[[1, 1]], b=[1], lower=[0, 0], upper=[1, 1]),
    DAGPathDecisionSpace([(0, 1), (1, 3), (0, 2), (2, 3)], 0, 3),
    StructuredBinaryDecisionSpace(2, C_ub=[[1, 2]], r_ub=[2]),
])
def test_supported_forward_spaces_small_smoke(space):
    ctx = context(space)
    algorithm = NestedLangevinActiveAlgorithm(tiny_config(theta_samples=2, outer_steps=1))
    algorithm.reset(ctx, np.random.default_rng(0))
    action = algorithm.propose(())
    y = ctx.decision_problem.minimize(action.query, np.random.default_rng(0))
    algorithm.observe(AlgorithmObservation(1, action.query, y))
    assert np.all(np.isfinite(algorithm.current_estimate()))
    assert algorithm.diagnostics()["round"] == 1


def test_runner_evaluation_and_json_export_integration():
    scenario = ActiveScenarioConfig(
        dimension=2, horizon=2, seed=0, true_theta=[1., -1.],
        decision_space=DecisionSpaceConfig(kind="independent_binary"),
        query_space=QuerySpaceConfig(candidate_count=4),
    )
    algorithm = NestedLangevinActiveAlgorithm(tiny_config())
    run = ActiveBenchmarkRunner(stopping_config=RegretStoppingConfig(enabled=False)).run(
        scenario, algorithm
    )
    evaluation = evaluate_active_run(run, ActiveEvaluationConfig(test_query_count=8,
                                                                 evaluate_trajectory=True))
    assert len(evaluation.normalized_regret_history) == 2
    assert run.runtime_seconds >= run.metadata["algorithm_initialization_seconds"] > 0
    for t, record in enumerate(run.records, 1):
        np.testing.assert_array_equal(record.theta_hat_after,
                                      record.update_diagnostics["theta_samples"][0])
        assert record.action_diagnostics["round"] == t - 1
        assert record.update_diagnostics["round"] == t
    public = run.to_dict(include_latent=False)
    json.dumps(public, allow_nan=False)
    assert "true_theta" not in public
    assert all("true_decision" not in record for record in public["records"])
    assert isinstance(load_algorithm_factory(
        "invoptlab.active:create_nested_langevin_algorithm")(), NestedLangevinActiveAlgorithm)


def test_contradictory_observations_use_soft_loss_not_hard_cone():
    algorithm = NestedLangevinActiveAlgorithm(tiny_config())
    algorithm.reset(context(), np.random.default_rng(5))
    for y in (np.zeros(2), np.ones(2)):
        algorithm.observe(AlgorithmObservation(1, np.ones(2), y))
    assert len(algorithm.observations_) == 2
    assert np.all(np.isfinite(algorithm.theta_samples_))
    assert algorithm.diagnostics()["estimate_status"] == "valid"


def test_clean_trivial_example_recovers_decision_signs():
    scenario = ActiveScenarioConfig(
        dimension=2, horizon=2, seed=0, true_theta=[-1., 1.],
        query_space=QuerySpaceConfig(candidate_count=12),
    )
    run = ActiveBenchmarkRunner(stopping_config=RegretStoppingConfig(enabled=False)).run(
        scenario, NestedLangevinActiveAlgorithm()
    )
    evaluation = evaluate_active_run(run, ActiveEvaluationConfig(test_query_count=16, seed=9103))
    assert evaluation.final_normalized_regret == 0.0
    assert evaluation.final_zero_regret_rate == 1.0
    np.testing.assert_array_equal(np.sign(run.parameter_history[-1]), [-1, 1])
