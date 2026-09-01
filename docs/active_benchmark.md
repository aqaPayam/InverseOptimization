# Active inverse-optimization benchmark

This benchmark implements the environment layer only. At time `t`, an algorithm returns its
current estimate `theta_hat_t` and a query `s_t`. The environment produces the expert decision,
applies the configured parameter and observation channels, and returns only the observation the
algorithm is allowed to see. The runner stores the complete trajectory for future evaluation.

The benchmark imposes one external stopping rule: after each update, it evaluates the estimate on a
fixed hidden test set and stops at the first time all normalized test regrets are numerically zero.
If this never occurs, the run continues to the configured horizon. The algorithm does not receive
the true parameter, hidden queries, regret values, or stopping calculation. Algorithm-originated
stop requests remain separate and are ignored unless `respect_stop_requests=True` is selected.
Score aggregation, rankings, and statistical comparison are not imposed.

## Interaction contract

The fixed forward objective is

`F_theta(s, x) = (s * theta)^T x`.

An algorithm implements four methods:

```python
from invoptlab.active import ActiveAction, ActiveAlgorithm

class MyAlgorithm(ActiveAlgorithm):
    def reset(self, context, rng): ...
    def propose(self, history):
        return ActiveAction(query=s_t, theta_hat=theta_hat_t)
    def observe(self, observation): ...
    def current_estimate(self): ...
```

`context` contains the dimension, horizon, finite query candidates, the known public feasible
decision problem/forward oracle, seed, scenario name, and public environment description. It never
contains the true parameter. Each public observation contains
`step`, the chosen `query`, `observed_decision`, an optional `observation_mask`, and public metadata.
The runner privately retains `theta_true`, the noisy expert parameter, the latent expert decision,
and channel metadata. See `examples/active_algorithm_template.py` for a complete importable shell.

## Implemented benchmark axes

Experts:

- exact minimizing expert with deterministic or random tie breaking;
- Gibbs expert, with low/medium/high normalized temperatures `0.1`, `0.5`, and `2.0` times a
  scenario-specific reference energy gap, or a user-supplied positive numeric temperature.

Decision spaces:

- independent binary decisions;
- fixed-cardinality binary decisions;
- bounded continuous polytopes, including user-supplied inequalities and equalities;
- structured binary decisions, represented by DAG shortest paths by default, with optional general
  binary equality/inequality constraints.

Query spaces always contain unit-L2-norm candidates and support balanced, clustered,
sharp-boundary, low-rank, rare-informative, aliased, sparse, and dense geometries. The sharp-boundary
and aliased constructors use the actual forward decision map, so their intended behavior is checked
when the scenario is built.

Observation channels support clean, local, outlier, biased/asymmetric, query-dependent, and partial
observations. Parameter channels support none, IID isotropic, anisotropic, query-dependent, and a
persistent session shift. Parameter noise is applied before the expert responds; observation noise
is applied afterward. Independent seeded random streams prevent a change in one channel from
silently changing unrelated environment components.

## Run one tiny check

```bash
python -m invoptlab active-smoke --dimension 5 --horizon 3
```

Or run the explicit small scenario file:

```bash
python -m invoptlab active-run configs/active_smoke.yaml --algorithm random
```

The built-in random algorithm is deliberately meaningless and exists only to test plumbing.

## Uniform-random sequential-incenter baseline

The built-in `uniform-incenter` baseline chooses every permitted query with equal probability. It
uses only the observed pairs `(s_t, y_t)`, converts each complete feasible observation into
consistency half-spaces, and recomputes the normalized hard-cone incenter after every response. The
latent clean decision `x_t` is never given to the algorithm. In a partial-observation scenario, an
observation with missing coordinates is recorded but skipped because ordinary exact hyperplanes
cannot be constructed without those missing values.

```bash
python -m invoptlab active-run configs/active_smoke.yaml --algorithm uniform-incenter
```

The repository also includes the permanently executed
`notebooks/08_active_uniform_incenter_clean_2d.ipynb`, which records a complete clean 2D run,
time-by-time estimates, evaluation tables, and plots directly inside the notebook.

`notebooks/09_active_curated_25_scenarios.ipynb` provides a smaller one-factor-at-a-time benchmark.
It covers every level of every benchmark axis in 25 executed scenarios, with one reference setting
and exactly one changed component per scenario. Its tables and plots are stored in the notebook.

Constraint construction is exact for independent binary, fixed-cardinality, uncoupled box, and
enumerable structured decision spaces. For a coupled continuous or non-enumerable structured
space, the baseline uses a finite set of public forward-oracle cuts and marks them as approximate
in its diagnostics.

## Universal evaluation

Evaluation is optional and remains separate from the environment and algorithm. For each scenario,
the evaluator generates a reproducible hidden set of new queries uniformly on the unit sphere. The
algorithm never receives these queries during learning, and every algorithm evaluated on the same
scenario receives exactly the same test set.

The two primary measurements are the final angular error between `theta_hat_T` and `theta_true`,
and final normalized regret on the hidden test queries. Positive rescaling is removed by the angular
measurement, but reversing the sign is not treated as equivalent. A zero or invalid estimate is
marked invalid and assigned 180 degrees. Normalized regret compares the decision selected by the
estimate with the true optimal decision, evaluates both under the clean base parameter, and divides
by that query's best-to-worst objective range. It therefore lies between zero and one across every
implemented decision-space family. The evaluator also reports the fraction of test queries with
numerically zero regret and runtime. It does not create a composite score.

## External stopping time

Every active run uses the same simple rule by default. At time `t`, after the algorithm observes
`Y_t` and returns its updated estimate, the benchmark evaluates that estimate on its hidden uniform
test queries. The stopping time is the first `t` for which every query has normalized regret no
larger than the configured numerical tolerance. When no such time exists, the stopping time is the
fixed horizon `T`. This is an evaluation protocol, not part of the algorithm, and its hidden inputs
never appear in the algorithm context or observation history.

The saved run records the stopping time, whether the zero-regret criterion was achieved, mean and
maximum hidden-test regret at every executed step, and the stopping reason. By default the rule uses
the same test-query count and seed as optional final evaluation. `--minimum-stop-time` can require a
minimum number of interactions. `--no-zero-regret-stop` disables the rule for controlled fixed-
horizon studies.

```bash
python -m invoptlab active-run configs/active_smoke.yaml --algorithm uniform-incenter --evaluate
```

The default evaluates only the final estimate using 128 hidden queries. The query count and seed
are configurable. Learning curves are optional because evaluating every intermediate estimate can
be expensive:

```bash
python -m invoptlab active-run configs/active_smoke.yaml --algorithm uniform-incenter --evaluate --test-queries 200 --evaluation-seed 4 --evaluation-trajectory
```

## Run custom algorithms

An importable zero-argument factory can be registered by `module:object`:

```bash
python -m invoptlab active-run configs/active_smoke.yaml --algorithm mine=examples.active_algorithm_template:create_algorithm
```

Repeat `--algorithm` to compare several algorithms on exactly the same scenario seeds. For a safe
preview of the complete grid, use `--limit 1`. Omit the limit only when a full benchmark run is
actually intended.

```bash
python -m invoptlab active-run configs/active_benchmark.yaml --algorithm mine=my_package.algorithms:create_algorithm --limit 1
```

## Configuration and scale

`configs/active_smoke.yaml` is a three-step example. `configs/active_benchmark.yaml` is the complete
lazy grid: 3 dimensions × 4 expert regimes × 4 decision spaces × 8 query regimes × 6 observation
channels × 5 parameter channels × 3 seeds = 34,560 scenarios. Loading the file does not materialize
this grid. Scenarios are generated one at a time, and `--limit` can select a prefix for development.

The configuration dataclasses expose all lower-level controls: explicit `true_theta`, dimensions,
horizon, seeds, Gibbs sampling settings, polytope matrices, graph edges, cardinality, candidate
count, query construction parameters, channel strengths, masks, covariances, and query-dependent
profiles. The enum values and defaults are defined in `src/invoptlab/active/config.py`.

## Stored trajectory

Each step stores both `theta_hat_before` (the estimate supplied with `s_t`) and `theta_hat_after`
(the estimate after observing the response), together with the query, observed response, optional
mask, stop request, pre-query action diagnostics, post-update diagnostics, and runtime. For the
uniform-incenter baseline, the post-update fields include its inradius, constraint count,
construction method, and exact/approximate status. Latent exports additionally contain the true
parameter, expert's possibly perturbed parameter, latent expert decision, objective value, and
noise metadata. Use `result.to_dict(include_latent=False)` or
`save_json(..., include_latent=False)` when a public-only export is required.

No composite score is imposed. External zero-regret stopping diagnostics are always stored unless
explicitly disabled. Full angular recovery and final/trajectory evaluation are attached only when
the optional evaluation layer is requested.
