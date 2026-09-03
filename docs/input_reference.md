# Complete input reference

This page collects the inputs accepted by `invoptlab` in one place. The framework treats every
forward problem as a minimization problem. A user normally supplies contexts, feasible decisions,
an objective, and observed decisions; everything else is optional configuration.

## Minimal synthetic experiment

`random_choice_experiment` accepts:

| Input | Type | Default | Meaning |
|---|---|---:|---|
| `parameter_dimension` | positive integer | `2` | Dimension of the unknown parameter |
| `observations` | positive integer | `30` | Number of demonstrated decisions |
| `alternatives` | positive integer | `8` | Feasible choices per context |
| `true_theta` | array or `None` | random | Ground-truth parameter |
| `parameter_space` | `ParameterSpace` or `None` | unit L2 ball | Identifiability domain |
| `noise_model` | noise object or `None` | clean | Demonstration corruption model |
| `seed` | integer | `0` | Reproducibility seed |

## Objective and forward-model inputs

For a linear-in-parameter objective,

\[
F_\theta(s,x)=\theta^\mathsf{T}\phi(s,x),
\]

use `LinearObjective` or `finite_choice_problem`. Supply the parameter dimension and a callable
`phi(context, decision)` that returns one numerical feature vector of that dimension.

For an arbitrary objective, use `CallableObjective(function, parameter_dimension, gradient=None)`.
The function receives `(theta, context, decision)` and returns a scalar. The optional gradient has
the same arguments and returns the derivative with respect to `theta`. When appropriate, inverse
losses fall back to central finite differences if no gradient is supplied.

Contexts and decisions may be arrays, dictionaries, indices, binary vectors, or custom Python
objects, provided the objective and oracle understand them.

Available forward-oracle routes are:

- `EnumerationOracle` for a finite feasible set returned by a context-dependent callable;
- `CallableOracle` to wrap an existing solver;
- `ScipyOracle` for continuous numerical optimization;
- optional `CVXPYOracle` for convex models;
- a user-defined oracle implementing `solve`, with optional enumeration and loss-augmented solve.

Exact finite consistency geometry requires a linear objective and an enumeration-capable oracle.
ASL requires a loss-augmented solver. The framework checks declared capabilities before using
these features.

## Parameter-space inputs

`ParameterSpace(dimension, kind, lower=None, upper=None, radius=1.0)` supports:

| `kind` | Required values | Domain |
|---|---|---|
| `l2_ball` | dimension and radius | `||theta||_2 <= radius` |
| `simplex` | dimension and radius | nonnegative entries summing to radius |
| `box` | same-length lower and upper arrays | component-wise bounds |

Dimensions two and three support direct cone visualization. Higher dimensions retain estimation,
constraint, regret, recovery, bootstrap, and influence diagnostics without direct cone plots.

## Observation and dataset inputs

An `Observation` accepts:

| Field | Required | Meaning |
|---|---:|---|
| `context` | yes | Information defining the forward instance |
| `decision` | yes | Observed demonstrated decision |
| `clean_decision` | no | Decision before synthetic noise |
| `true_theta` | no | Ground truth for recovery and true-regret metrics |
| `timestamp` | no | Ordering for chronological or online analysis |
| `weight` | no | Nonnegative observation importance; default `1` |
| `expert_id` | no | Decision-maker identifier |
| `noise` | no | Noise metadata |
| `metadata` | no | Arbitrary user metadata |

Create datasets with `InverseDataset`, `InverseDataset.from_records`, `load_csv`, `load_json`, or
`generate_dataset`. Synthetic `true_theta` may be a fixed vector or a callable of
`(observation_index, context)`, allowing parameter drift.

Dataset utilities include chronological ordering, deterministic splitting, fingerprints,
summaries, JSON persistence, CSV loading, and K-fold indices.

## Noise-model inputs

| Model | Inputs |
|---|---|
| `NoNoise` | none |
| `RandomFeasibleNoise` | replacement probability |
| `AdditiveNoise` | scale, `gaussian`/`laplace`/`uniform`, optional projection |
| `EpsilonOptimalNoise` | epsilon-optimality tolerance |
| `BoltzmannNoise` | positive temperature |
| `BinaryFlipNoise` | component-wise flip probability |
| `ContaminationNoise` | base model, contamination probability, optional contaminator |

A custom noise object implements `apply(...)` and returns `(noisy_decision, metadata)`.

## Inverse-loss inputs

| Loss | Inputs and interpretation |
|---|---|
| `SuboptimalityLoss` | Standard forward-objective suboptimality (SL) |
| `AugmentedSuboptimalityLoss` | Decision distance and margin scale (ASL) |
| `DecisionDistanceLoss` | User-selected decision-distance callable |
| `KKTResidualLoss` | Callable returning residual value and parameter gradient |

Built-in decision distances include Euclidean, squared Euclidean, and Hamming distance. A custom
loss implements `value_and_subgradient(problem, theta, observation)`.

## Risk inputs

| Risk | Input |
|---|---|
| `MeanRisk` | observation weights |
| `CVaRRisk` | tail fraction in `(0, 1]` |
| `TrimmedMeanRisk` | trim fraction in `[0, 0.5)` |
| `QuantileRisk` | quantile level |

## Estimator inputs

`IncenterEstimator` accepts numerical tolerance, maximum SLSQP iterations, and
`sequential_history`. Enable sequential history to estimate the incenter after every observation.

`ProjectedSubgradientEstimator` accepts:

- inverse loss and risk aggregator;
- learning rate and epochs;
- L2 regularization strength;
- batch or stochastic updates;
- optional mirror descent on a simplex;
- random seed;
- history recording interval;
- optional initial parameter `theta0`.

`OnlineEstimator` accepts a projected-subgradient base estimator and the number of passes per
observation. `ConsistencyEstimator` is the stable normalized-consistency representative based on
the incenter implementation.

## Experiment, statistics, and sweep inputs

`ExperimentConfig` accepts a run name, seed, feasibility-validation switch, geometry switch,
geometry sample count, and string tags. `ExperimentRunner` accepts a problem, dataset, and
estimator.

Additional experiment inputs include:

- train/validation/test fractions and shuffle seed;
- number of K-fold splits;
- hyperparameter-grid dictionaries and repeated seeds;
- bootstrap repetitions, confidence level, and seed;
- leave-one-out estimator factory;
- output directory and report switches in YAML/JSON configurations.

## Plot and report inputs

| Output | Adjustable inputs | Sequential behavior |
|---|---|---|
| 2D cone | step, true theta, hyperplane visibility | one snapshot per observation |
| 2D cone animation | snapshots and true theta | animated over every observation |
| 3D cone | step, sample count, seed | sampled snapshot; repeatable per step |
| Parameter path | estimator history | every recorded observation/epoch |
| Training loss | estimator history | every `record_every` epochs |
| Regret | per-observation results | instantaneous and cumulative |
| Geometry history | sample count and seed | feasible fraction, constraints, inradius |
| 2D loss landscape | resolution and contour/surface mode | repeatable for dataset prefixes |
| Run comparison | selected metric | final summary across methods |
| HTML report | problem, dataset, optional loss | tables plus interactive figures |

The default examples use low resolutions and small datasets. Increase them only when a research
run needs greater numerical or visual detail.

## Active diffusion and explicit query inputs

See [all corrected and legacy sampler inputs](active_nested_langevin.md#all-sampler-inputs).
`NestedLangevinConfig` now defaults to the corrected `gaussian_gibbs` backend, 16 samples,
an ensemble-mean estimate and disagreement queries. `query_policy="uniform"` changes
only query selection, for a matched-estimator comparison.

`QuerySpaceConfig(kind="explicit", candidates=my_matrix)` accepts a finite nonzero N-by-d
candidate matrix. Rows are normalized to unit length; `candidate_count` is inferred.
Duplicates count as repeated rows in the uniform candidate distribution. With repeats
disabled, selecting one row removes all numerically identical rows.

`evaluate_active_run(run, config, test_queries=my_test_matrix)` accepts a separate finite,
nonempty, unit-norm N-by-d held-out design, records it in evaluation output, and uses its
actual row count. It is not passed to the algorithm. Without this override, `scenario`
evaluation on an explicit pool samples rows uniformly WITH replacement (not unseen queries).

`build_query_sensitive_scenarios(seed=0, horizon=12, parameter_sigma=0.02)` returns three
designs, each containing its scenario, separate test queries, candidate-group labels and
description. Evaluation weights the fresh informative ratios / coordinate pairs evenly,
which differs intentionally from the imbalanced training pool.
