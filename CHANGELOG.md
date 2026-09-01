# Changelog

All notable changes to this project will be documented here.

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
