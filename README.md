# invoptlab

`invoptlab` is an extensible Python laboratory for inverse-optimization experiments. It combines
user-defined forward models and datasets with inverse estimators, clean/noisy data generation,
sequential analysis, consistency-cone geometry, statistical diagnostics, interactive plots, and
reproducible reports.

## Documentation

- [Complete input reference](docs/input_reference.md): every model, data, noise, estimator,
  statistical, and plotting input in one place.
- [User guide](docs/user_guide.md): the core workflow and extension rules.
- [Mathematical background](docs/mathematical_background.md): consistency inequalities, losses,
  identifiability, and regret definitions.
- [Active benchmark](docs/active_benchmark.md): environment axes, algorithm contract, configuration,
  execution, and raw trajectory format.
- [Executed feature tour](notebooks/06_complete_feature_tour.ipynb): a small end-to-end experiment
  whose tables and figures render directly on GitHub.
- [Loss over theta through time](notebooks/07_loss_landscape_over_time.ipynb): SL and ASL evaluated
  over the full 2D parameter domain after every incoming observation.

## What is implemented

- Arbitrary callable objectives and specialized linear-in-parameter feature models.
- Finite enumeration, custom callable, SciPy continuous, and optional CVXPY forward oracles.
- L2-ball, simplex, and box parameter domains.
- CSV/JSON/custom datasets, chronological replay, splitting, fingerprints, and validation.
- Clean synthetic generation plus additive, feasible-action, epsilon-optimal, Boltzmann, binary-flip,
  and contamination noise.
- Exact consistency constraints for finite linear models.
- Normalized incenter and sequential incenter histories.
- Suboptimality, augmented suboptimality, decision-distance, and custom KKT-residual losses.
- Mean, quantile, trimmed-mean, and CVaR risk aggregation.
- Projected subgradient, stochastic, mirror-descent, and online learning.
- Parameter recovery, decision prediction, true/surrogate regret, constraint margins, geometry,
  bootstrap, and influence diagnostics.
- Static 2D geometry, interactive 2D animations, sampled 3D geometry, loss landscapes, parameter
  paths, regret plots, comparisons, and standalone HTML reports.
- Python, notebook, and command-line workflows.
- Algorithm-independent active inverse-optimization environments with minimizing/Gibbs experts,
  four decision-space families, eight query geometries, and configurable parameter/observation
  channels.

## Installation

For the notebooks, plots, configurations, and reports:

```bash
python -m pip install -e ".[plots,notebooks,reports]"
```

Install `.[all]` only if you also want optional CVXPY support and the development dependencies.

## Minimal experiment

```python
import invoptlab as io

problem, dataset, theta_true = io.random_choice_experiment(
    parameter_dimension=2,
    observations=16,
    alternatives=5,
    seed=7,
)

estimator = io.IncenterEstimator(sequential_history=True)
result = io.ExperimentRunner(
    io.ExperimentConfig(name="clean-2d", seed=7)
).run(problem, dataset, estimator)

print(result.summary())
result.plot_cone(problem, true_theta=theta_true)
animation = result.animate_cone(problem, true_theta=theta_true)
animation.show()
result.generate_report(problem, dataset, "outputs/experiments/clean-2d/report.html")
```

## User-defined model

```python
import numpy as np
import invoptlab as io

def decisions(context):
    return context["choices"]

def phi(context, decision):
    return np.asarray(decision)

problem = io.finite_choice_problem(
    parameter_dimension=2,
    feature_map=phi,
    feasible_decisions=decisions,
    parameter_space=io.ParameterSpace(2, "l2_ball"),
)

records = [
    {"context": {"choices": [[0, 1], [1, 0], [1, 1]]}, "decision": [0, 1]},
    {"context": {"choices": [[0, 2], [2, 0], [1, 1]]}, "decision": [0, 2]},
]
dataset = io.InverseDataset.from_records(records)
result = io.ExperimentRunner().run(problem, dataset, io.IncenterEstimator())
```

For a continuous or specialized forward problem, provide `CallableOracle`, `ScipyOracle`, or
`CVXPYOracle`. Geometry is enabled automatically only when the declared capabilities support it.

## Command-line demo

```bash
invoptlab demo --dimension 2 --observations 16 --estimator incenter
invoptlab demo --dimension 2 --noise 0.15 --estimator asl --epochs 120
invoptlab run configs/default.yaml
invoptlab active-smoke --dimension 5 --horizon 3
```

You can also use `python -m invoptlab ...` if the console command is not on your path.

## Active benchmark

The active benchmark asks an algorithm for `(theta_hat_t, s_t)`, returns the resulting noisy or
clean expert observation, and records the complete trajectory. It deliberately does not define
evaluation, scoring, or stopping policy yet. Start with the tiny configuration in
`configs/active_smoke.yaml`; the complete lazy 34,560-scenario grid is in
`configs/active_benchmark.yaml`. Custom algorithms follow the small interface shown in
`examples/active_algorithm_template.py`.

## Notebooks

The `notebooks/` directory contains executable, top-to-bottom examples for quick start, cone
evolution, noisy SL/ASL learning, custom data/models, online learning, and method comparison.
Their defaults are intentionally small and fast. Increase observations, epochs, grid resolution,
or 3D samples only when you deliberately want a larger experiment.

For a single guided example covering nearly every major feature, start with
`notebooks/06_complete_feature_tour.ipynb`. Its first settings cell is the only part you need to
edit for a new synthetic run. The repository version includes verified outputs so its figures are
visible directly on GitHub without executing the notebook.

Launch them with:

```bash
jupyter lab notebooks/01_quickstart_2d.ipynb
```

## Important interpretation

For linear objectives, positive scaling of the parameter does not change the forward decision.
Parameter recovery metrics are therefore direction-aware. Cone plots show the consistency cone
intersected with a bounded parameter domain. True regret is reported only when ground-truth
parameters are available; otherwise results use the explicit label `surrogate_suboptimality`.
