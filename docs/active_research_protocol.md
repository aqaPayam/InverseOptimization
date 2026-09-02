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
