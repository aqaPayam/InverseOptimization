import json

import matplotlib
import numpy as np

matplotlib.use("Agg")

import invoptlab as io
from invoptlab.losses import AugmentedSuboptimalityLoss
from invoptlab.visualization import plot_loss_landscape_2d


def test_noisy_asl_experiment_and_artifacts(tmp_path):
    problem, dataset, _ = io.random_choice_experiment(
        observations=20,
        seed=8,
        noise_model=io.RandomFeasibleNoise(0.15),
    )
    estimator = io.ProjectedSubgradientEstimator(epochs=80, record_every=10, seed=8)
    result = io.ExperimentRunner(io.ExperimentConfig(name="noisy", geometry_samples=500)).run(
        problem, dataset, estimator
    )
    destination = result.save(tmp_path / "run")
    assert (destination / "summary.json").exists()
    assert np.all(np.isfinite(result.theta))
    figure = plot_loss_landscape_2d(
        problem, dataset, AugmentedSuboptimalityLoss(), resolution=15, result=result
    )
    assert len(figure.data) >= 1
    report = result.generate_report(
        problem,
        dataset,
        tmp_path / "report.html",
        loss=AugmentedSuboptimalityLoss(),
    )
    assert report.exists()
    assert "Cone evolution" in report.read_text(encoding="utf-8")


def test_online_estimator_has_one_state_per_observation():
    problem, dataset, _ = io.random_choice_experiment(observations=10, seed=12)
    estimator = io.OnlineEstimator(io.ProjectedSubgradientEstimator(learning_rate=0.2))
    estimator.fit(problem, dataset)
    assert len(estimator.history_.steps) == len(dataset)


def test_gradient_training_does_not_mutate_context_features():
    problem, dataset, _ = io.random_choice_experiment(
        parameter_dimension=2, observations=6, alternatives=4, seed=31
    )
    before = [observation.context.copy() for observation in dataset]
    io.ProjectedSubgradientEstimator(
        loss=io.AugmentedSuboptimalityLoss(), epochs=5, seed=31
    ).fit(problem, dataset)
    for observation, expected in zip(dataset, before):
        np.testing.assert_array_equal(observation.context, expected)
