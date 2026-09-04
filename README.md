# invoptlab

`invoptlab` is an extensible Python laboratory for inverse-optimization experiments. It combines
user-defined forward models and datasets with inverse estimators, clean/noisy data generation,
sequential analysis, consistency-cone geometry, statistical diagnostics, interactive plots, and
reproducible reports.

## Documentation

- [Four-algorithm benchmark results](notebooks/17_four_algorithm_eight_scenarios.ipynb):
  the locked eight-scenario protocol expanded with Uniform Online SAMD, containing all 160
  runs, 3,200 observed steps, every-time-step plots/tables, diagnostics, and full outputs.
- [Complete four-algorithm PDF report](output/pdf/Four_Algorithm_Active_Inverse_Optimization_Complete_Results.pdf):
  a shareable 185-page report with the principal convergence figures, comparison tables,
  all 160 run summaries, and the complete 3,200-step appendix.
- [Pedro, Genious Pedro, and Score Base results](notebooks/16_pedro_genious_score_base_eight_scenarios.ipynb):
  the preserved three-algorithm comparison, with 120 complete runs and 2,400 observed steps.
- [Complete three-algorithm PDF report](output/pdf/Pedro_Genious_Pedro_Score_Base_Complete_Results.pdf):
  a shareable 141-page report containing protocol, conclusions, convergence charts, all run
  summaries, and the complete 2,400-step appendix.
- [Pedro versus Score base: eight-scenario results](notebooks/15_pedro_vs_score_base_eight_scenarios.ipynb):
  the preserved historical two-method view. Its scientific Pedro/Score trajectories are
  reproduced exactly in the new three-algorithm run.
- [Pedro/Score base comparison protocol](docs/pedro_score_comparison.md): exact algorithm
  definitions, eight scenario designs, noise, test distributions and reproducibility.
- [Genious Pedro algorithm](docs/genious_pedro.md): Pedro's same sequential incenter estimate
  with a minimum normalized decision-margin query rule.
- [Uniform Online SAMD](docs/uniform_online_samd.md): uniform queries with one signed
  exponentiated ASL update per new observation; the noise-tolerant fourth comparison method.
- [Saved experiment outputs](outputs/README.md): raw trajectories and historical snapshots,
  including all 160 runs of the current four-algorithm comparison.
- [Complete input reference](docs/input_reference.md): every model, data, noise, estimator,
  statistical, and plotting input in one place.
- [User guide](docs/user_guide.md): the core workflow and extension rules.
- [Mathematical background](docs/mathematical_background.md): consistency inequalities, losses,
  identifiability, and regret definitions.
- [Active benchmark](docs/active_benchmark.md): environment axes, algorithm contract, configuration,
  execution, and raw trajectory format.
- [Hard active research protocol](docs/active_research_protocol.md): twelve interpretable coupled
  and information-limited families, behavioral noise, fair stopping, and multi-seed evaluation.
- [Corrected diffusion and legacy v2 algorithm](docs/active_nested_langevin.md): the Gaussian-augmented
  sampler, configurable estimate and query policies, and all configuration inputs.
- [Corrected diffusion validation and query comparison](notebooks/14_corrected_diffusion_validation.ipynb):
  numerical reference checks, three small noisy query-sensitive scenarios, a matched-estimator
  comparison of uniform versus disagreement queries, and plots at every step. **Its uniform
  control is not Pedro's incenter algorithm; use notebook 15 for Pedro-versus-Score-base results.**
- [Executed nested Langevin sanity examples](notebooks/11_active_nested_langevin_sanity.ipynb):
  two tiny 2D examples with loss surfaces, samples and query scores at every step.
- [Random vs Diffusion comparison](notebooks/12_random_vs_diffusion_fixed_horizon.ipynb):
  historical **legacy v2** results on twelve scenarios (including parameter noise), equal fixed query budgets,
  first/sustained recovery times, failures, and full trajectory plots.
- [Diffusion accuracy audit and query designs](notebooks/13_diffusion_audit_and_query_designs.ipynb):
  saved-score verification, a numerical inner-sampler bias check, and three small candidate-pool
  designs where query selection has a clear information advantage.
- [Executed feature tour](notebooks/06_complete_feature_tour.ipynb): a small end-to-end experiment
  whose tables and figures render directly on GitHub.
- [Loss over theta through time](notebooks/07_loss_landscape_over_time.ipynb): SL and ASL evaluated
  over the full 2D parameter domain after every incoming observation.
- [Active uniform-query incenter baseline](notebooks/08_active_uniform_incenter_clean_2d.ipynb): an
  executed 12-step clean 2D run with hidden-query angular-error and normalized-regret evaluation.
- [Curated active benchmark](notebooks/09_active_curated_25_scenarios.ipynb): 25 permanently
  executed one-factor-at-a-time scenarios covering every benchmark axis without the full grid.
- [Hard active research benchmark](notebooks/10_active_hard_research_protocol.ipynb): the compact
  protocol with complete regret, angular-error, correctness, and incenter trajectories.

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
clean expert observation, and records the complete trajectory. An external benchmark rule stops at
the first zero-regret hidden-test estimate, or at the fixed horizon if zero regret is not reached.
The algorithm never sees the stopping data, and no composite score is defined. Start with the tiny configuration in
`configs/active_smoke.yaml`; the complete lazy 34,560-scenario grid is in
`configs/active_benchmark.yaml`. Custom algorithms follow the small interface shown in
`examples/active_algorithm_template.py`.

For scientific comparisons, use the compact hard protocol rather than the complete Cartesian
software grid. It preserves the linear objective but adds coupled decisions, scarce information,
behavior-calibrated noise, robust validation stopping, and a separate final test set:

```bash
python -m invoptlab active-research --algorithm uniform-incenter
python -m invoptlab active-research --algorithm uniform-online-samd
```

The first estimation baseline is available as `--algorithm uniform-incenter`. It chooses queries
uniformly at random and updates a sequential consistency-cone incenter using only observed `Y`.
The third algorithm is available as `--algorithm genious-pedro`. It uses the same incenter but,
after the initial uniform query, chooses the smallest normalized predicted decision margin.
The fourth algorithm is available as `--algorithm uniform-online-samd`. It keeps uniform queries
and replaces the hard cone with one online ASL mirror update per observation.
Add `--evaluate` to calculate final angular error and normalized regret on a shared hidden set of
new uniform test queries. No single composite score is imposed.

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
