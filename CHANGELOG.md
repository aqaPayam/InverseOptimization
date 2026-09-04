# Changelog

## Uniform Online SAMD implementation

- Added the fourth active method: uniform query selection with one online ASL
  exponentiated mirror update per newly received observation.
- Implemented the paper's signed `2d` positive/negative parameter construction,
  enabling the method to estimate the mixed-sign parameters in this benchmark.
- Uses exact finite loss-augmented decisions in the current scenarios
  (`epsilon=0`), records complete update diagnostics, and skips incomplete
  observations without accessing latent values.
- Added `run_four_algorithm_design`, the `uniform-online-samd` CLI name, focused
  unit/integration tests, documentation, and a four-step sanity check.
- Ran the locked four-way comparison: 160 fixed-horizon runs and 3,200 observed
  steps across the unchanged eight scenarios and five paired seeds.
- Added executed notebook 17 with all per-step tables, convergence figures,
  estimator diagnostics, pairwise comparisons, and expandable full outputs.

## Genious Pedro implementation

- Added `GeniousPedroAlgorithm`: Pedro's exact cone incenter with minimum
  normalized decision-margin query selection.
- Uses a uniform first query for the empty dataset and an explicit uniform
  fallback when the incenter is invalid; invalid estimates remain failures.
- Records every candidate margin, the selected margin, predicted optimizer,
  nearest alternative, tie indices, and fallback reason for later analysis.
- Added a tiny five-step sanity script and focused tests.
- Ran the authorized exact three-way comparison separately from the preserved
  two-method snapshot: 120 fixed-horizon runs and 2,400 observed steps.
- Added executed notebook 16 and a visually verified 141-page PDF containing
  convergence plots, all run summaries, and the complete time-step appendix.
- Verified that the Pedro and Score Base scientific trajectories reproduce the
  earlier comparison exactly; only Genious Pedro is added.

All notable changes to this project will be documented here.

## Pedro versus Score base: eight-scenario experiment

- Added unambiguous `PedroAlgorithm` (uniform query, incenter) and `ScoreBaseAlgorithm`
  (disagreement query, mean of parameter samples) names without relabeling historical controls.
- Added eight predeclared 4D/6D scenarios, five paired seeds, full T=20 trajectories,
  and separate ordinary/balanced held-out evaluations for imbalanced candidate pools.
- Added source/configuration-validated per-run checkpoints and a standalone resumable runner.
- Added exact cached enumeration for small structured binary MIN problems (at most 4096
  possible binary states, also honoring the user enumeration limit); larger spaces retain MILP.
  Small-space ties now consistently use lexicographic enumeration, including batch solves.
- Added a self-contained results notebook with full per-step parameter/query/decision outputs,
  five-seed plots, failure counts, recovery times and explicitly conditional valid-only means.

## Corrected diffusion sampling and controlled query comparison

- Replaced the default projected Euler sampler with target-invariant Gaussian-augmented
  Gibbs and full-chord random-direction slice transitions. This is a documented change
  from v2, retaining the same full-sum loss and bounded Lebesgue target, not a new likelihood.
- Preserved the original backend as `sampler="projected_langevin"` and explicitly pinned
  notebooks 11–13 to legacy settings without recomputing or relabeling their saved results.
- Added ensemble-mean estimates (new default, 16 samples), configurable first-sample
  estimates, uniform query ablation, independent query RNGs and randomized score ties.
- Added a conditional-radius slice refresh with the polar Jacobian to improve
  magnitude mixing, validated against uniform radial moments and coupled 2D integration.
- Added exact batched binary/cardinality MIN solves, with generic solver fallback.
- Added explicit candidate pools and separately supplied held-out query designs, with
  three small threshold/rare-information/coordinate-coverage scenarios, all noisy MIN.
- Fixed invalid-estimate stopping: failed/zero estimates never count as zero-regret success.
- Added quadrature and stationary-moment checks, solver-family integration tests, matched
  estimator/RNG tests, and an executed 18-run, T=12, d=2/4 validation notebook (14).

## Nested Langevin active algorithm

- Added the v2 Gaussian-smoothed nested parameter sampler with bounded box/ball support,
  projected inner states, unprojected outer states, and final latent-state extraction.
- Added first-retained-sample estimates and full-ensemble maximum-disagreement queries.
- Added configurable sampling schedules, warm starts, reproducible parallel trajectories,
  forward-solve profiling and inner/outer diagnostics without exposing latent expert data.
- Registered `nested-langevin` in both active CLI workflows, with a complete input guide.
- Included algorithm initialization in benchmark runtime (important for t=0 sampling).
- Added mathematical and integration sanity tests and an executed two-example 2D notebook.
- Added the `diffusion` CLI alias and an optional all-scenario fixed-horizon research mode.
- Added first/sustained angular and joint angle-regret recovery times, with not-reached counts.
- Added incremental run checkpoints and a twelve-family Random + incenter vs Diffusion notebook,
  preserving all parameter-noise cases and both full T=8 trajectories.

## Active research protocol

- Added a compact twelve-family hard benchmark separating its easy sanity control from coupled,
  information-limited, boundary, partial-feedback, and stochastic research cases.
- Added behavioral calibration for local observation and isotropic parameter noise using their
  empirical probability of changing the expert decision.
- Added scenario-distribution held-out queries, independent validation/final-test streams,
  consecutive-success stopping, fixed-horizon stochastic evaluation, first/stable threshold times,
  and multi-seed mean/standard-deviation summaries.
- Made the hard sequential incenter baseline record solver failures without terminating the
  benchmark.
- Replaced artificial angular-error penalties for invalid estimates with explicit
  `degenerate_cone` and `insufficient_information` failures and undefined performance metrics.
- Added the `active-research` command and a permanently executed research notebook.
- Replaced the initial research suite with twelve MIN-only, clean-observation families covering
  3D geometry, cardinality, knapsack, paths, a continuous simplex, limited queries, and calibrated
  parameter noise at dimensions 3, 10, 18, and 20.

## 0.1.0

- Added extensible objectives, parameter spaces, forward oracles, and datasets.
- Added clean and noisy synthetic data generation.
- Added incenter, consistency, projected-subgradient, stochastic, mirror, and online estimators.
- Added SL, ASL, decision-distance, and custom KKT-residual losses.
- Added exact finite consistency constraints and sequential 2D/3D geometry utilities.
- Added recovery, prediction, regret, robustness, bootstrap, and influence diagnostics.
- Added static and interactive figures, HTML reports, CLI/config workflows, and notebooks.
- Added a lightweight executed feature-tour notebook and regression tests.
- Added a sequential notebook visualizing SL and ASL directly over theta after every observation.
- Added an algorithm-independent active inverse-optimization benchmark with minimizing and Gibbs
  experts, four decision-space families, eight query geometries, observation/parameter channels,
  lazy grid execution, public/latent trajectory exports, CLI commands, and a custom-algorithm API.
- Added a uniform-random query baseline with sequential consistency-hyperplane construction and
  normalized incenter estimates based only on public observed decisions.
- Added optional universal active-run evaluation using final angular parameter error and normalized
  clean regret on reproducible hidden uniform test queries, with optional learning curves.
- Added an external benchmark stopping time: stop at the first zero-regret hidden-test estimate or
  continue to the fixed horizon, without exposing evaluation information to the algorithm.
- Added executed notebooks for the clean 2D stopping example and a curated 25-scenario
  one-factor-at-a-time active benchmark with embedded tables and comparison plots.
