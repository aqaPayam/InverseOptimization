# Uniform Online SAMD

`UniformOnlineSAMDAlgorithm` is the noise-tolerant fourth method in the active
comparison. It separates data acquisition from parameter estimation:

1. choose one candidate query uniformly;
2. receive the new public `(s_t, y_t)` observation;
3. solve the ASL loss-augmented decision problem exactly over the public finite
   decision set;
4. take one exponentiated mirror-descent update.

The active environment has the known objective
`F(theta, s, x) = theta^T (s * x)`. The ASL subgradient from competitor `x_t`
is therefore `s_t * (y_t - x_t)`. Signed parameters are represented as
`theta = theta_plus - theta_minus`, so the entropy mirror update operates on a
nonnegative vector of dimension `2d`, as in the paper's L1 construction.

This is the online specialization discussed after Algorithm 1, with a uniform
query rule added by this benchmark. It uses the newly received observation
once; it does not retrain on the full history after every query. Exact finite
enumeration gives epsilon zero in the current comparison scenarios.

## Inputs

- `learning_rate` (default `1.0`): multiplier in `eta_t = learning_rate/sqrt(t)`.
- `l1_radius` (default `None`): signed L1-ball radius. `None` resolves to
  `sqrt(d)`, which contains all unit-L2 directions without using hidden truth.
- `margin_scale` (default `1.0`): multiplier on the L1 decision distance in ASL.
- `normalize_subgradient` (default `True`): divide by the L-infinity dual norm.
- `tolerance` and `exponent_clip`: numerical safeguards only.

The algorithm is available as `--algorithm uniform-online-samd`. The
four-method comparison entry point is `run_four_algorithm_design`.

## Interpretation

Pedro versus Uniform Online SAMD holds uniform query selection fixed and changes
the estimator from a hard consistency cone to soft ASL updates. Pedro versus
Genious Pedro instead holds the hard incenter estimator fixed and changes query
selection. Score Base changes both components.

This method is not the paper's offline Algorithm 1 verbatim because Algorithm 1
starts from a fixed dataset and does not select queries. It is the corresponding
online update plus the benchmark's uniform query policy.
