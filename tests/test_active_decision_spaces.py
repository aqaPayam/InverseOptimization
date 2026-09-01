import numpy as np
import pytest

from invoptlab.active import (
    ContinuousPolytopeDecisionSpace,
    DAGPathDecisionSpace,
    FixedCardinalityDecisionSpace,
    IndependentBinaryDecisionSpace,
    StructuredBinaryDecisionSpace,
)


@pytest.mark.parametrize(
    "space",
    [
        IndependentBinaryDecisionSpace(5),
        FixedCardinalityDecisionSpace(5, 2),
        ContinuousPolytopeDecisionSpace(5),
        DAGPathDecisionSpace([(0, 1), (1, 3), (0, 2), (2, 3), (1, 2)], 0, 3),
        StructuredBinaryDecisionSpace(5, A_eq=[[1, 1, 1, 1, 1]], b_eq=[2]),
    ],
)
def test_decision_spaces_min_gibbs_projection_and_sampling(space):
    rng = np.random.default_rng(4)
    cost = np.asarray([-0.7, 0.2, -0.1, 0.8, 0.3])
    minimum = space.min_decision(cost, rng)
    gibbs = space.sample_gibbs(cost, 0.5, rng, burn_in=5, steps=3)
    projected = space.project(rng.normal(size=5))
    sampled = space.sample_feasible(rng)
    assert space.contains(minimum)
    assert space.contains(gibbs)
    assert space.contains(projected)
    assert space.contains(sampled)
    assert space.reference_energy_gap(cost, rng) >= 0


def test_fixed_cardinality_gibbs_scales_without_enumeration():
    space = FixedCardinalityDecisionSpace(50, 10, max_enumeration=100)
    decision = space.sample_gibbs(
        np.linspace(-1, 1, 50), 0.4, np.random.default_rng(7)
    )
    assert decision.sum() == 10


def test_independent_binary_minimum_is_coordinatewise():
    space = IndependentBinaryDecisionSpace(4)
    decision = space.min_decision(
        np.asarray([-2.0, 1.0, -0.5, 3.0]), np.random.default_rng(0)
    )
    np.testing.assert_array_equal(decision, [1, 0, 1, 0])

