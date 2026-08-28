import numpy as np

import invoptlab as io
from invoptlab.geometry import build_consistency_constraints, feasible_polygon_2d


def polygon_area(polygon):
    if polygon.shape[0] < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def test_clean_incenter_is_consistent_and_predictive():
    problem, dataset, truth = io.random_choice_experiment(observations=25, seed=3)
    estimator = io.IncenterEstimator(sequential_history=True).fit(problem, dataset)
    constraints = build_consistency_constraints(problem, dataset)
    assert constraints.feasible(estimator.theta_)
    assert estimator.radius_ >= -1e-7
    assert len(estimator.history_.steps) == len(dataset)
    result = io.ExperimentRunner(io.ExperimentConfig(geometry_samples=1_000)).run(
        problem, dataset, io.IncenterEstimator(sequential_history=True)
    )
    assert result.metrics["observed_decision_accuracy"] >= 0.99
    assert result.metrics["parameter_cosine_similarity"] > 0.9


def test_consistency_region_does_not_expand():
    problem, dataset, _ = io.random_choice_experiment(observations=12, seed=4)
    areas = []
    all_constraints = build_consistency_constraints(problem, dataset, deduplicate=False)
    for step in range(1, len(dataset) + 1):
        polygon = feasible_polygon_2d(problem.parameter_space, all_constraints.prefix(step).deduplicated())
        areas.append(polygon_area(polygon))
    assert all(later <= earlier + 1e-8 for earlier, later in zip(areas, areas[1:]))

