from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from invoptlab.active import (
    ActiveBenchmarkRunner, ActiveEvaluationConfig, ActiveInverseEnvironment,
    ActiveScenarioConfig, AlgorithmObservation, GaussianGibbsSampler,
    InverseLossTarget, NestedLangevinActiveAlgorithm, NestedLangevinConfig,
    QuerySpaceConfig, RegretStoppingConfig, build_query_sensitive_scenarios,
    evaluate_active_run, sample_scenario_hidden_queries,
    disagreement_score,
    PublicDecisionProblem, IndependentBinaryDecisionSpace, FixedCardinalityDecisionSpace,
    ContinuousPolytopeDecisionSpace, DAGPathDecisionSpace, StructuredBinaryDecisionSpace,
)
from invoptlab.exceptions import SolverError, ValidationError


def binary(theta, s):
    return (theta * s < 0).astype(float)


def config(**kwargs):
    return replace(NestedLangevinConfig(theta_samples=4, gibbs_sweeps=2,
        conditional_slice_steps=2, target_slice_steps=8, record_chain_trace=False), **kwargs)


@pytest.mark.parametrize("field,value", [
    ("sampler", "wrong"), ("query_policy", "wrong"), ("point_estimate", "wrong"),
    ("query_tie_breaking", "wrong"), ("gibbs_sweeps", 0),
    ("conditional_slice_steps", True), ("target_slice_steps", -1),
    ("max_slice_shrinks", 1.5),
    ("radial_refresh", "yes"),
])
def test_config_validation(field, value):
    with pytest.raises(ValidationError):
        config(**{field: value})


@pytest.mark.parametrize("domain", ["box", "ball"])
def test_chord_and_no_data_uniform_target_moments(domain):
    sampler = GaussianGibbsSampler(InverseLossTarget(2, 20., binary),
        config(parameter_domain=domain), np.random.default_rng(34))
    direction = np.array([0.6, 0.8])
    z = np.array([0.2, -0.1])
    lo, hi = sampler.chord(z, direction)
    assert lo < 0 < hi
    if domain == "box":
        assert np.max(np.abs(z + hi * direction)) == pytest.approx(1.)
    else:
        assert np.linalg.norm(z + hi * direction) == pytest.approx(1.)
    states = []
    for i in range(4000):
        z = sampler.slice_step(z)
        if i >= 200:
            states.append(z.copy())
    values = np.asarray(states)
    np.testing.assert_allclose(values.mean(axis=0), 0., atol=0.06)
    np.testing.assert_allclose((values ** 2).mean(axis=0),
                               1/3 if domain == "box" else 1/4, atol=0.04)
    assert np.all(np.max(np.abs(values), axis=1) < 1)


def test_nonsmooth_conditional_matches_quadrature_where_euler_was_biased():
    observations = [AlgorithmObservation(i, np.ones(1), np.zeros(1)) for i in range(8)]
    density = lambda z: np.exp(-160 * max(0., -z) - (z + 0.3)**2 / 0.2)
    reference = quad(lambda z: z * density(z), -1, 1, points=[0])[0] / quad(
        density, -1, 1, points=[0])[0]
    means = []
    for seed in range(64):
        sampler = GaussianGibbsSampler(InverseLossTarget(1, 20., binary, observations),
                                      config(), np.random.default_rng(seed))
        z = np.array([-0.3])
        retained = []
        for step in range(32):
            z = sampler.slice_step(z, u=np.array([-0.3]), tau=0.1)
            if step >= 16 and (step - 16) % 4 == 0:
                retained.append(z[0])
        means.append(np.mean(retained))
    assert reference == pytest.approx(0.16414326644, abs=1e-10)
    assert abs(np.mean(means) - reference) < 0.035


def test_full_sampler_1d_target_not_final_conditional_target():
    observations = [AlgorithmObservation(i, np.ones(1), np.zeros(1)) for i in range(8)]
    # pi(z) ~ 1 for z>0, exp(160z) for z<0: mean about .4969.
    density = lambda z: np.exp(-160 * max(0., -z))
    reference = quad(lambda z: z * density(z), -1, 1, points=[0])[0] / quad(
        density, -1, 1, points=[0])[0]
    draws = [GaussianGibbsSampler(InverseLossTarget(1, 20., binary, observations),
        config(), np.random.default_rng(i)).sample().theta[0] for i in range(256)]
    assert abs(np.mean(draws) - reference) < 0.05
    assert np.all(np.abs(draws) < 1.)


def test_conditional_kernel_preserves_current_z_and_exact_gaussian_variance(monkeypatch):
    cfg = config(tau_schedule=(0.25,), inner_step_sizes=(0.01,), outer_step_sizes=(0.02,),
                 gibbs_sweeps=1, conditional_slice_steps=1, target_slice_steps=1, radial_refresh=False)
    obs = [AlgorithmObservation(1, np.ones(1), np.zeros(1))]
    rng = np.random.default_rng(71)
    sampler = GaussianGibbsSampler(InverseLossTarget(1, 20., binary, obs), cfg, rng)
    expected_u = 0.4 + 0.5 * np.random.default_rng(71).normal(size=1)
    seen = []
    def identity(z, **kwargs):
        seen.append((z.copy(), kwargs))
        return z.copy()
    monkeypatch.setattr(sampler, "slice_step", identity)
    result = sampler.sample(np.array([0.4]))
    np.testing.assert_allclose(seen[0][0], [0.4])
    np.testing.assert_allclose(seen[0][1]["u"], expected_u)
    assert seen[0][1]["tau"] == 0.25
    assert seen[1][1] == {}  # Final refresh targets pi, not r(z|u).
    np.testing.assert_allclose(result.theta, [0.4])


def test_bad_support_and_slice_work_limit_fail_explicitly(monkeypatch):
    sampler = GaussianGibbsSampler(InverseLossTarget(1, 20., binary),
                                  config(max_slice_shrinks=1), np.random.default_rng(1))
    with pytest.raises(ValidationError, match="support"):
        sampler.slice_step(np.array([2.]))
    with pytest.raises(ValidationError, match="support"):
        sampler.radial_step(np.array([2.]))
    monkeypatch.setattr(sampler, "log_density", lambda z, *args: 0. if z[0] == 0. else -np.inf)
    with pytest.raises(SolverError, match="shrink limit"):
        sampler.slice_step(np.zeros(1))


def test_same_estimator_same_feedback_independent_of_query_policy_and_workers():
    env = ActiveInverseEnvironment(build_query_sensitive_scenarios(horizon=2)[0].scenario)
    ctx = env.algorithm_context()
    algorithms = [NestedLangevinActiveAlgorithm(config(query_policy=p, workers=w))
                  for p, w in (("uniform", 1), ("disagreement", 1), ("disagreement", 2))]
    for algorithm in algorithms:
        algorithm.reset(ctx, np.random.default_rng(11))
        for _ in range(3):
            algorithm.propose(())  # Consume selection RNGs differently.
        algorithm.observe(AlgorithmObservation(1, np.array([0.6, 0.8]), np.array([1., 0.])))
        np.testing.assert_array_equal(algorithm.current_estimate(), algorithm.theta_samples_.mean(axis=0))
    for algorithm in algorithms[1:]:
        np.testing.assert_array_equal(algorithm.theta_samples_, algorithms[0].theta_samples_)


def test_vectorized_disagreement_matches_independent_all_sample_formula():
    design = build_query_sensitive_scenarios(horizon=1)[2]
    context = ActiveInverseEnvironment(design.scenario).algorithm_context()
    algorithm = NestedLangevinActiveAlgorithm(config())
    algorithm.reset(context, np.random.default_rng(90))
    indices, scores = algorithm.score_candidates()
    expected = []
    for index in indices:
        predictions = [context.decision_problem.minimize(theta*context.query_candidates[index],
                       np.random.default_rng(0)) for theta in algorithm.theta_samples_]
        expected.append(disagreement_score(np.asarray(predictions)))
    np.testing.assert_allclose(scores, expected, atol=1e-14)
    action = algorithm.propose(())
    assert action.diagnostics["selected_score"] == pytest.approx(max(expected))


@pytest.mark.parametrize("policy", ["uniform", "disagreement"])
def test_corrected_query_ties_and_duplicate_nonrepeat_handling(policy):
    pool = [[1., 0.], [1., 0.], [0., 1.]]
    scenario = ActiveScenarioConfig(dimension=2, horizon=2,
        query_space=QuerySpaceConfig(kind="explicit", candidates=pool, allow_repeated_queries=False))
    context = ActiveInverseEnvironment(scenario).algorithm_context()
    algorithm = NestedLangevinActiveAlgorithm(config(query_policy=policy))
    algorithm.reset(context, np.random.default_rng(15))
    algorithm.theta_samples_[:] = 0.5  # No disagreement at any candidate.
    first, second = algorithm.propose(()), algorithm.propose(())
    assert not np.array_equal(first.query, second.query)
    with pytest.raises(ValidationError, match="exhausted"):
        algorithm.propose(())

    scenario.query_space.allow_repeated_queries = True
    algorithm.reset(ActiveInverseEnvironment(scenario).algorithm_context(), np.random.default_rng(15))
    algorithm.theta_samples_[:] = .5
    indices = {algorithm.propose(()).diagnostics["candidate_index"] for _ in range(30)}
    assert indices == {0, 1, 2}  # Ties are not always resolved to first row.


@pytest.mark.parametrize("bad", [[[0, 0], [1, 0]], [[np.nan, 1], [1, 0]], [[1, 0]]])
def test_explicit_candidates_validation(bad):
    with pytest.raises(ValidationError):
        QuerySpaceConfig(kind="explicit", candidates=bad)


def test_query_designs_hidden_tests_no_theta_dependent_pool_and_small_runner():
    designs = build_query_sensitive_scenarios(horizon=2)
    for design in designs:
        scenario = design.scenario
        env = ActiveInverseEnvironment(scenario)
        other = ActiveInverseEnvironment(replace(scenario, true_theta=np.ones(scenario.dimension)))
        np.testing.assert_array_equal(env.query_space.candidates, other.query_space.candidates)
        assert scenario.parameter_noise.sigma == 0.02
        public = env.algorithm_context().public_environment
        assert "true_theta" not in public and "test_queries" not in public
        assert not np.any(np.linalg.norm(
            env.query_space.candidates[:, None] - design.test_queries[None], axis=2) < 1e-7)
        hidden = sample_scenario_hidden_queries(scenario, env.theta_true, 7,
            evaluation_seed=15, distribution="scenario", decision_space=env.decision_space)
        assert hidden.shape == (7, scenario.dimension)
        run = ActiveBenchmarkRunner(stopping_config=RegretStoppingConfig(enabled=False)).run(
            scenario, NestedLangevinActiveAlgorithm(config()))
        result = evaluate_active_run(run, ActiveEvaluationConfig(test_query_count=1),
                                     test_queries=design.test_queries)
        assert result.test_query_count == 96
        assert result.metadata["test_query_distribution"] == "explicit-heldout"
        assert result.final_estimate_valid
        with pytest.raises(ValidationError, match="held-out"):
            evaluate_active_run(run, test_queries=np.zeros((1, scenario.dimension)))


@pytest.mark.parametrize("space", [IndependentBinaryDecisionSpace(4),
    FixedCardinalityDecisionSpace(4, 2),
    ContinuousPolytopeDecisionSpace(2, A=[[1, 1]], b=[1], lower=[0, 0], upper=[1, 1]),
    DAGPathDecisionSpace([(0, 1), (1, 3), (0, 2), (2, 3)], 0, 3),
    StructuredBinaryDecisionSpace(2, C_ub=[[1, 2]], r_ub=[2]),
])
def test_batched_forward_exactly_matches_scalar_including_ties(space):
    problem = PublicDecisionProblem(space)
    rng = np.random.default_rng(14)
    costs = np.vstack([rng.normal(size=(8, space.dimension)), np.zeros((2, space.dimension))])
    np.testing.assert_array_equal(problem.minimize_batch(costs, rng),
        np.vstack([problem.minimize(cost, rng) for cost in costs]))
    scenario = ActiveScenarioConfig(dimension=space.dimension, horizon=1)
    ctx = ActiveInverseEnvironment(scenario).algorithm_context()
    ctx.decision_problem = problem
    algorithm = NestedLangevinActiveAlgorithm(config(theta_samples=2, gibbs_sweeps=1,
                                                     conditional_slice_steps=1, target_slice_steps=2))
    algorithm.reset(ctx, rng)
    action = algorithm.propose(())
    y = problem.minimize(action.query, rng)
    algorithm.observe(AlgorithmObservation(1, action.query, y))
    assert np.all(np.isfinite(algorithm.current_estimate()))


def test_batched_loss_matches_generic_full_sum_and_gradient():
    problem = PublicDecisionProblem(FixedCardinalityDecisionSpace(3, 1))
    rng = np.random.default_rng(83)
    obs = [AlgorithmObservation(i, s, problem.minimize(s, rng))
           for i, s in enumerate(rng.normal(size=(5, 3)))]
    plain = NestedLangevinActiveAlgorithm._target(problem, 3, 20., obs)
    batch = NestedLangevinActiveAlgorithm._target(problem, 3, 20., obs, use_batch=True)
    for theta in np.vstack([rng.normal(size=(8, 3)), np.zeros((1, 3))]):
        a, ga = plain.loss_and_subgradient(theta)
        b, gb = batch.loss_and_subgradient(theta)
        assert a == pytest.approx(b)
        np.testing.assert_array_equal(ga, gb)
    assert plain.forward_calls == batch.forward_calls == 45


def test_two_dimensional_endpoint_distribution_matches_independent_quadrature():
    problem = PublicDecisionProblem(FixedCardinalityDecisionSpace(2, 1))
    queries = np.array([[0.6, 0.8], [0.8, 0.6]])
    decisions = np.eye(2)
    obs = [AlgorithmObservation(i, s, y) for i, (s, y) in enumerate(zip(queries, decisions))]
    # Independent midpoint integration: don't call the sampler's loss function.
    axis = (np.arange(200) + 0.5) * 2 / 200 - 1
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    loss = sum((grid * s * y).sum(axis=1) - (grid * s).min(axis=1)
               for s, y in zip(queries, decisions))
    weights = np.exp(-20 * loss)
    weights /= weights.sum()
    moments = weights @ np.c_[grid, grid ** 2, grid[:, 0] * grid[:, 1]]
    draws = []
    for seed in range(256):
        target = NestedLangevinActiveAlgorithm._target(problem, 2, 20., obs, use_batch=True)
        draws.append(GaussianGibbsSampler(target, config(gibbs_sweeps=4, target_slice_steps=24),
                                        np.random.default_rng(seed)).sample().theta)
    draws = np.asarray(draws)
    observed = np.c_[draws, draws ** 2, draws[:, 0] * draws[:, 1]].mean(axis=0)
    np.testing.assert_allclose(observed, moments, atol=0.065)


@pytest.mark.parametrize("domain", ["box", "ball"])
def test_radial_refresh_includes_polar_jacobian(domain):
    sampler = GaussianGibbsSampler(InverseLossTarget(4, 20., binary),
                                  config(parameter_domain=domain), np.random.default_rng(200))
    direction = np.ones(4) / 2
    limit = 2 if domain == "box" else 1
    z, radii = direction * .5, []
    for i in range(4000):
        z = sampler.radial_step(z)
        if i >= 200:
            radii.append(np.linalg.norm(z) / limit)
        np.testing.assert_allclose(z / np.linalg.norm(z), direction, atol=1e-12)
    # Conditional radius density is 4*r^3 on [0,1], not uniform.
    assert abs(np.mean(radii) - 4/5) < .025
    assert abs(np.mean(np.square(radii)) - 4/6) < .03
