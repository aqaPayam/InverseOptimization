import numpy as np
import pytest

from invoptlab.active import (
    DecisionSpaceConfig,
    IndependentBinaryDecisionSpace,
    ObservationNoiseConfig,
    ObservationNoiseKind,
    ParameterNoiseConfig,
    ParameterNoiseKind,
    QuerySpaceConfig,
    QuerySpaceKind,
)
from invoptlab.active.noise import make_observation_noise, make_parameter_noise
from invoptlab.active.query_spaces import make_query_space


@pytest.mark.parametrize("kind", list(QuerySpaceKind))
def test_all_query_geometries_construct_unit_candidates(kind):
    dimension = 5
    theta = np.asarray([0.5, -0.2, 0.4, 0.7, -0.1])
    theta /= np.linalg.norm(theta)
    decision_space = IndependentBinaryDecisionSpace(dimension)
    query_space = make_query_space(
        QuerySpaceConfig(
            kind=kind,
            candidate_count=12,
            construction_attempts=400,
        ),
        dimension,
        theta,
        decision_space,
        np.random.default_rng(9),
    )
    assert query_space.candidates.shape == (12, dimension)
    np.testing.assert_allclose(np.linalg.norm(query_space.candidates, axis=1), 1.0)
    if kind == QuerySpaceKind.SHARP_BOUNDARY:
        assert query_space.metadata["successful_boundary_pairs"] > 0
    if kind == QuerySpaceKind.ALIASED:
        assert query_space.metadata["successful_aliased_pairs"] > 0


@pytest.mark.parametrize("kind", list(ParameterNoiseKind))
def test_all_parameter_noise_models_return_unit_parameters(kind):
    dimension = 5
    theta = np.ones(dimension) / np.sqrt(dimension)
    config = ParameterNoiseConfig(kind=kind, sigma=0.05)
    noise = make_parameter_noise(config, dimension)
    rng = np.random.default_rng(12)
    noise.reset(theta, rng)
    first, _ = noise.apply(theta, np.eye(1, dimension, 0).reshape(-1), 1, rng)
    second, _ = noise.apply(theta, np.eye(1, dimension, 1).reshape(-1), 2, rng)
    np.testing.assert_allclose(np.linalg.norm(first), 1.0)
    np.testing.assert_allclose(np.linalg.norm(second), 1.0)
    if kind == ParameterNoiseKind.PERSISTENT:
        np.testing.assert_allclose(first, second)


@pytest.mark.parametrize("kind", list(ObservationNoiseKind))
def test_all_observation_noise_models_return_valid_shapes(kind):
    dimension = 5
    decision_space = IndependentBinaryDecisionSpace(dimension)
    truth = np.asarray([1, 0, 1, 0, 1])
    query = np.ones(dimension) / np.sqrt(dimension)
    noise = make_observation_noise(
        ObservationNoiseConfig(kind=kind, sigma=0.1, outlier_probability=1.0),
        dimension,
    )
    result = noise.apply(truth, query, decision_space, 1, np.random.default_rng(15))
    assert result.decision.shape == truth.shape
    if kind == ObservationNoiseKind.PARTIAL:
        assert result.mask is not None
        np.testing.assert_array_equal(result.decision, result.mask * truth)
    else:
        assert result.mask is None
        assert decision_space.contains(result.decision)

