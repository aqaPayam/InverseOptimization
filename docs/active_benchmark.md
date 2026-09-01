# Active inverse-optimization benchmark

This benchmark implements the environment layer only. At time `t`, an algorithm returns its
current estimate `theta_hat_t` and a query `s_t`. The environment produces the expert decision,
applies the configured parameter and observation channels, and returns only the observation the
algorithm is allowed to see. The runner stores the complete trajectory for future evaluation.

Stopping rules, performance measures, score aggregation, rankings, and statistical comparison are
intentionally not imposed yet. An algorithm may return a stop request, but the standard runner
ignores it unless `respect_stop_requests=True` is selected. This keeps the benchmark independent of
the future evaluation protocol.

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

`context` contains the dimension, horizon, finite query candidates, seed, scenario name, and public
environment description. It never contains the true parameter. Each public observation contains
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
mask, stop request, algorithm diagnostics, and runtime. Latent exports additionally contain the true
parameter, expert's possibly perturbed parameter, latent expert decision, objective value, and noise
metadata. Use `result.to_dict(include_latent=False)` or `save_json(..., include_latent=False)` when a
public-only export is required.

No loss, recovery measure, regret, stopping decision, or score is calculated by this phase. The raw
record design preserves the information needed to add those as a separate evaluation layer later.
