import numpy as np

import invoptlab as io
from invoptlab.data import generate_dataset
from invoptlab.losses import AugmentedSuboptimalityLoss, SuboptimalityLoss


def test_parameter_space_projections():
    simplex = io.ParameterSpace(3, "simplex")
    projected = simplex.project([2.0, -1.0, 0.5])
    assert np.all(projected >= 0)
    assert np.isclose(projected.sum(), 1.0)
    ball = io.ParameterSpace(2, "l2_ball")
    projected = ball.project([3.0, 4.0])
    assert np.isclose(np.linalg.norm(projected), 1.0)


def test_dataset_split_and_fingerprint_are_reproducible():
    _, dataset, _ = io.random_choice_experiment(observations=20, seed=1)
    first = dataset.split(seed=7)
    second = dataset.split(seed=7)
    assert [part.fingerprint for part in first] == [part.fingerprint for part in second]
    assert sum(map(len, first)) == len(dataset)


def test_sl_and_asl_are_nonnegative():
    problem, dataset, theta = io.random_choice_experiment(observations=8, seed=2)
    for loss in (SuboptimalityLoss(), AugmentedSuboptimalityLoss()):
        values = [loss.value_and_subgradient(problem, theta, obs)[0] for obs in dataset]
        assert min(values) >= -1e-12

