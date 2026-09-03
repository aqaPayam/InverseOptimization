# Compact active inverse-optimization research protocol

This protocol is the scientific benchmark. The older 34,560-case Cartesian grid remains useful
for software coverage, but it should not be used by itself to claim that an algorithm learns a
difficult active inverse-optimization problem.

The objective remains `F_theta(S, x) = (S * theta)^T x`. Every family uses the deterministic MIN
expert and clean observations, so `Y = X`. Difficulty is created by coupled feasible decisions,
limited query information, controlled decision boundaries, and behaviorally calibrated parameter
noise. This keeps the hyperplane/cone interpretation while
avoiding the one-observation sign-recovery shortcut of an independent binary decision set.

## The twelve scenario families

1. `geometry-cardinality-3d`: `d=3`, cardinality one, with cone/incenter visualization.
2. `cardinality-balanced-d20`: `d=20`, cardinality ten, balanced queries.
3. `cardinality-small-margin-d20`: `d=20`, cardinality ten, boundary queries.
4. `knapsack-d20`: 20 unequal-weight items under one resource budget.
5. `dag-path-d18`: decisions are source-to-sink paths over 18 edge variables.
6. `continuous-simplex-d10`: a coupled ten-dimensional continuous simplex.
7. `cardinality-sparse-queries-d20`: each query touches only three coordinates.
8. `cardinality-rare-informative-d20`: 90% of queries lie in a rank-three subspace.
9. `cardinality-parameter-noise-mild-d20`: IID noise changes about 5% of MIN decisions.
10. `cardinality-parameter-noise-moderate-d20`: IID noise changes about 15% of MIN decisions.
11. `dag-path-parameter-noise-moderate-d18`: IID noise changes about 15% of MIN paths.
12. `knapsack-parameter-noise-moderate-d20`: IID noise changes about 15% of MIN knapsacks.

The suite is intentional rather than factorial: each family has a specific interpretation. By
default it uses five seeds, 40 interactions, 64 available query candidates, 64 hidden validation
queries, and 128 separate hidden test queries.

## Fair stopping and final testing

The algorithm never sees the true parameter, validation queries, test queries, regret, or stopping
calculation. Clean deterministic scenarios stop only after zero validation regret for three
consecutive interactions. Stochastic scenarios always run to the fixed horizon, which prevents a
lucky response from ending the experiment.

Set `fixed_horizon=True` to disable early stopping for **all** algorithms and
scenarios, including clean cases. Each execution then collects exactly T queries
unless it encounters an execution error. Evaluation thresholds are retrospective
measurements, not stopping commands or feedback to the algorithms.

Validation and final testing use different random seeds. Both draw fresh held-out queries from the
scenario's query family. The final report therefore does not reuse the set that decided when a run
ended.

For every time step, final testing records angular error, mean normalized clean regret, maximum
normalized regret, and zero-regret rate. It also records the first and stable time at which mean
regret is at most 0.01, plus corresponding exact-zero times. Across seeds, the report gives the
mean, sample standard deviation, threshold success rate, stability success rate, and runtime. It
does not manufacture a composite score.

Failure is explicit. If hard consistency leaves no valid nonzero parameter direction, the result is
`degenerate_cone`. If the algorithm obtains no usable constraints, it is
`insufficient_information`. These failed estimates have undefined angular error and regret; they
are never encoded as artificial 90- or 180-degree values. The summary reports their failure rate
and reason separately from successful-estimate performance.

## Running the built-in baseline

```bash
python -m invoptlab active-research --algorithm uniform-incenter
```

For a quick development run:

```bash
python -m invoptlab active-research --algorithm uniform-incenter --seeds 0 1 --horizon 12 --candidates 24 --validation-queries 32 --test-queries 48 --output outputs/active/research-quick
```

Run any importable custom algorithm with the same interaction contract:

```bash
python -m invoptlab active-research --algorithm mine=my_package.algorithms:create_algorithm
```

Repeat `--algorithm` to compare algorithms on exactly the same scenario seeds. Each run is saved as
JSON alongside `manifest.json` and `research-summary.json`.

## Python inputs

`ActiveResearchConfig` accepts:

- `seeds`: repeated experiment seeds;
- `horizon`: maximum number of interactions;
- `candidate_count`: number of permitted active queries;
- `validation_query_count` and `test_query_count`;
- distinct `validation_seed` and `test_seed`;
- `consecutive_validation_successes`;
- `zero_regret_tolerance`;
- `learning_regret_threshold`.
- `learning_angular_threshold_degrees` (default 5 degrees);
- `fixed_horizon` (default false; true forces the full budget for every scenario).

Use `build_active_research_scenarios(config)` to inspect or modify scenarios before running. Use
`run_active_research_benchmark(algorithms, config)` to obtain the complete benchmark result and
multi-seed summary. The protocol accepts any existing `ActiveAlgorithm`; it is not specialized to
the built-in incenter baseline.

## Interpretation

The independent binary problem remains available separately as a plumbing control; it is not one
of these twelve research families. The coupled, information-limited, boundary, and parameter-noise
families are the meaningful comparisons. Angular error and behavioral regret can
disagree because some parameter directions may not affect decisions; both should be reported. The
hidden true-parameter validation rule is an oracle sample-complexity measurement, not a deployable
stopping detector available to an algorithm.

## Fixed-horizon Random vs Diffusion comparison

[Notebook 12](../notebooks/12_random_vs_diffusion_fixed_horizon.ipynb) compares
**uniform Random S + incenter estimation** with **Diffusion**, the new nested
Langevin sampler plus maximum-disagreement query selection. Both run to T=8 on
all 12 existing families, including all four parameter-noise cases. It uses seed
0, 16 candidates, and 32 hidden scenario-distribution test queries. Observation
noise remains disabled; parameter noise is retained exactly as specified by each
scenario. The same seeds give shared candidate pools, true parameters, hidden
queries and noise streams, not identical chosen queries or responses.

The notebook uses a deliberately small Diffusion budget: four samples, three tau
levels, 32 inner steps (16 burn-in, thinning 4), and four outer steps per level.
Its source code, configuration and outputs are saved; no larger or tuned run is
implied. The earlier sanity notebook uses a larger per-round sampling budget.

For every run the evaluator reports first and sustained-through-T threshold times:

- `first_angular_threshold_step`, `stable_angular_threshold_step`: angle <= 5 degrees;
- `first_threshold_step`, `stable_threshold_step`: mean normalized regret <= 0.01;
- `first_joint_threshold_step`, `stable_joint_threshold_step`: both simultaneously;
- `first_zero_regret_step`, `stable_zero_regret_step`: every hidden query has zero regret.

All thresholds are configurable. A sustained time means the threshold holds at
EVERY remaining measured step; a hit at the final step alone is not proof of
future stability. Times start after the first observation. `None` means not
reached by T, never an invented time of T or T+1. Invalid estimates do not count
as hits and break sustained success; a later failure does not erase an earlier
first hit. Group summaries include reached rates and not-reached counts rather
than treating non-recovery as a successful late finish. Final-only evaluations
do not invent threshold times from an unmeasured trajectory.

The CLI supports `--algorithm diffusion` as an alias for the new sampler and
`--fixed-horizon --angular-threshold 5` in `active-research`. Use `uniform-incenter`
for the comparison baseline; the older `random` CLI option is a meaningless
random-estimate plumbing test, NOT the Random + incenter research baseline.
For the notebook's small sampler configuration, use its Python factories rather
than the default CLI sampler settings.

The optional `run_completed` callback in `run_active_research_benchmark` receives
each successfully executed, evaluated run for incremental checkpointing. Stored
algorithm failure statuses are distinct from execution errors.

Report final regret, angle, failures, query-based threshold times and runtime
together. These two methods change BOTH estimation and query selection; their
comparison alone cannot isolate the effect of active queries. A one-seed,
small-budget comparison is exploratory and must not be presented as a universal
ranking or a claim of exact recovery under noise.
