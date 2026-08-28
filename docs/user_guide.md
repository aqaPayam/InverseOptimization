# User guide

## The four objects a user normally supplies

1. A context or signal `s` for each observation.
2. A feasible-decision oracle describing `X(s)`.
3. An objective `F(theta, s, x)` or a feature map `phi(s, x)`.
4. Observed decisions and optional ground truth.

For finite problems linear in the unknown parameter, use `finite_choice_problem`. For arbitrary
solvers, wrap existing code with `CallableOracle`. Continuous problems can use `ScipyOracle` or
the optional `CVXPYOracle`.

## Geometry

The homogeneous consistency cone is unbounded and always contains zero. `invoptlab` therefore
analyzes `C_t` intersected with the configured `ParameterSpace`. Exact halfspaces are constructed
when alternatives can be enumerated. Three-dimensional geometry uses documented sampling when
the exact mesh would be unnecessarily expensive.

## Noise

Noise is applied during data generation and its metadata is stored on each observation. Keeping
both `clean_decision` and `decision` permits noise-correction and true-regret metrics.

## Metrics

Parameter error is direction-aware. Decision metrics distinguish observed and clean targets. The
name `true_regret` is reserved for evaluations with `true_theta`; otherwise the framework reports
`surrogate_suboptimality`.

## Extension rule

New losses implement `value_and_subgradient`. New estimators implement `fit`, expose `theta_` and
`history_`, and optionally implement `predict`. New oracles implement `solve`; enumeration and ASL
methods are optional capabilities.

