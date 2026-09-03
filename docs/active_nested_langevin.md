# Corrected diffusion sampling and legacy nested Langevin

The default algorithm now uses **Gaussian-augmented Gibbs with random-direction
slice transitions**. This is a deliberate correction to the numerical method,
not a claim that specification v2 already described these updates.
The full loss, target distribution, public information boundary, and disagreement
objective remain unchanged. The original v2 implementation remains available.

Start with [executed validation notebook 14](../notebooks/14_corrected_diffusion_validation.ipynb):
reference integration, small noisy comparisons, saved tables and plots at every t.

## Why the default changed

[Audit notebook 13](../notebooks/13_diffusion_audit_and_query_designs.ipynb)
found an inner conditional whose numerical reference mean was 0.164143, while
the legacy 32-step estimate averaged 0.403081 over 64 seeds. Increasing to 320
steps at the same step size still gave 0.379776. More iterations did not remove
the fixed-step discretization/boundary bias.

Correcting only the inner update would leave the estimated-score outer Euler
approximation. The new backend therefore replaces both stages. It does not
apply a fictitious Metropolis correction to an intractable smoothed density.

## Unchanged mathematical target and information contract

The forward objective is `F(theta,s,x) = theta @ (s*x)`. The algorithm receives
the public hard-MIN solver, candidates, and complete observed `(S,Y)`. It never
receives hidden theta*, latent X, noisy expert parameters, or evaluation queries.
Partial, nonfinite or infeasible Y is rejected; no hidden values are substituted.

```text
L_t(theta) = sum_i theta @ (s_i * (Y_i - x_star(theta,s_i)))
pi_t(theta) proportional to exp(-beta * L_t(theta)), theta in Theta
Theta = [-bound,bound]^d or the full-dimensional L2 ball
r_tau(z|u) proportional to pi_t(z) * exp(-||z-u||²/(2*tau))
```

This is the full SUM, not an average/minibatch. There is no distance augmentation,
added prior factor or Gibbs expert. The base measure is Lebesgue measure on a
box/ball, not surface measure on a sphere. At t=0, pi is uniform on that support.

**Modeling limitation:** this is a loss-based target, not the likelihood of
Gaussian parameter-noisy MIN responses. Since L_t(0)=0 and the loss is positively
homogeneous, conflicting observations can favor small magnitudes. Correct
sampling does not fix this modeling issue or identify a parameter's scale.

## Corrected default: gaussian_gibbs

For each independent trajectory, start uniformly in the support, or warm-start
from that trajectory's previous endpoint after receiving a new observation.
At t=0, draw the uniform target directly, without approximate MCMC.

For each decreasing variance tau, repeat:

1. Draw the unbounded auxiliary variable `u = z + sqrt(tau) * N(0,I)`.
2. Hold u fixed. Starting from the **current z**, apply several conditional
   random-direction slice transitions targeting r_tau(z|u).
3. Carry the resulting z to the next Gaussian draw. Do not reset z to project(u).

One slice transition draws a log slice height, selects an isotropic random
direction, computes the **entire box/ball chord**, and shrinks that interval
around the current point until a proposal is inside the slice. Proposals are
never clipped. A finite work limit raises a sampling error rather than silently
returning a substitute sample.

After all levels, apply additional nonlocalized slice transitions targeting
pi itself. Each such step also refreshes the radius conditional on direction,
using density proportional to `r^(d-1) * pi(r * direction)` over the full
support radius. The polar volume factor is essential: without it, radial
updates would incorrectly overweight the origin. This extra transition improves
magnitude exploration in narrow cones; it does not impose a sphere constraint.
Return the final z from this refresh as the trajectory's parameter
sample, not the Gaussian u. The recorded outer state is the last auxiliary
state before these final target refresh moves, not a parameter estimate.

### What is guaranteed, and what is not

The augmented joint density is
`P_tau(z,u) proportional to pi(z) * N(u;z,tau I)`.
The Gaussian u|z draw is exact; each inner slice transition leaves z|u invariant.
The composed transition therefore preserves this joint density, with marginal
pi(z). Every fixed-tau marginal kernel preserves the SAME pi, so composing a
finite predetermined tau schedule preserves pi too. The final target slice
kernel also preserves pi.

This argument establishes the intended **invariant distribution**, assuming
correct forward loss evaluation and exact arithmetic. It does NOT say that a
finite inner run is an independent conditional sample or that a short
trajectory from an arbitrary start has mixed. A new observation changes pi, so
warm-started chains still need mixing. In high dimensions, narrow directions can
mix slowly. Recorded diagnostics explicitly say `finite-chain-not-certified`.

Background: [Neal, Slice Sampling](https://arxiv.org/abs/physics/0009028) and
[Lee, Shen & Tian, Restricted Gaussian Oracles](https://arxiv.org/abs/2010.03106).
The implementation is Gibbs-within-MCMC, not an exact independent restricted
Gaussian oracle, and does not claim those papers' complexity guarantees.

## Estimation, query selection, and reproducibility

The new default estimate is the mean of 16 trajectory endpoints, without
normalization. `point_estimate="first"` restores the first-sample convention.
A mean is intended to reduce point-estimate randomness, not guaranteed to reduce
angular error on every dataset. Near-zero means are marked invalid, not replaced
with a fabricated direction.

For every candidate s and ALL M parameter samples, solve the public MIN and use

```text
A(s) = 2/(M-1) * sum_m ||x_m(s) - mean_m x_m(s)||²
```

This equals average pairwise squared decision distance. The default chooses an
argmax, with uniform random selection among exact score ties; this also avoids
always choosing the first row when all scores are zero. Deterministic first-row
ties remain configurable. There is no hidden-theta query scoring.

`query_policy="uniform"` samples an available candidate index uniformly and
does not evaluate disagreement. It changes neither sampling nor estimation.
Separate query and per-trajectory RNG streams ensure identical histories yield
identical samples under either policy. Duplicate rows count as repeated entries
when repeats are enabled; disabling repeats removes all duplicate chosen rows.

`reset()` produces the t=0 ensemble; `propose()` selects a query without
resampling; `observe()` copies Y and resamples exactly once. The returned estimate
after observe is theta_hat_t. All runtime includes initialization.

## All sampler inputs

| Input | Default | Applies to / interpretation |
|---|---|---|
| sampler | gaussian_gibbs | Corrected backend; or projected_langevin for legacy v2 |
| beta | 20 | Full-sum loss inverse temperature |
| parameter_domain | box | Box or full-dimensional ball |
| bound | 1 | Box half-width or ball radius |
| tau_schedule | (0.5, 0.1, 0.02) | Strictly decreasing positive variances, BOTH backends |
| theta_samples | 16 | Independent random trajectories, at least 2 |
| point_estimate | mean | mean or first |
| query_policy | disagreement | disagreement or uniform; same estimator |
| query_tie_breaking | random | random or first among exact maximal scores |
| gibbs_sweeps | 6 | Corrected: auxiliary Gaussian refreshes per tau |
| conditional_slice_steps | 4 | Corrected: slice transitions per fixed-u conditional |
| target_slice_steps | 32 | Corrected: final nonlocalized pi transitions |
| radial_refresh | true | Corrected: one volume-corrected radius update per final target step |
| max_slice_shrinks | 1000 | Corrected: work cap per slice transition; raises on exhaustion |
| warm_start | true | Both: use previous trajectory endpoint after new data |
| workers | 1 | Both: independent per-trajectory threads/RNGs/solver copies |
| record_chain_trace | true | Both: detailed chain trace; false keeps summaries/samples |
| inner_step_sizes | (0.02, 0.005, 0.001) | LEGACY ONLY: one delta per tau |
| outer_step_sizes | (0.1, 0.02, 0.004) | LEGACY ONLY: one eta per tau |
| inner_steps | 64 | LEGACY ONLY: projected steps per conditional |
| inner_burn_in | 32 | LEGACY ONLY: discarded initial steps |
| inner_thinning | 4 | LEGACY ONLY: retain burn+1, burn+1+thin, ... |
| outer_steps | 8 | LEGACY ONLY: Euler updates per tau |
| initialization_std | 1 | LEGACY ONLY: initial outer Gaussian std |
| warm_start_renoise_std | 0 | LEGACY ONLY: optional Gaussian noise on warm starts |
| warm_start_inner | false | LEGACY ONLY: reuse previous conditional state |
| max_state_norm | 1e6 | LEGACY ONLY: explosion guard |

Legacy-only fields remain serialized/validated but are not used by the corrected
sampler. Their sequence lengths need match tau only in legacy mode. Corrected
budgets are small starting points, not universal convergence settings.

## Minimal noisy example / matched-estimator comparison

```python
from dataclasses import replace
from invoptlab.active import (
    ActiveBenchmarkRunner, ActiveEvaluationConfig, RegretStoppingConfig,
    NestedLangevinConfig, NestedLangevinActiveAlgorithm,
    build_query_sensitive_scenarios, evaluate_active_run,
)

design = build_query_sensitive_scenarios(seed=0, horizon=12, parameter_sigma=.02)[0]
base = NestedLangevinConfig(record_chain_trace=False)
runner = ActiveBenchmarkRunner(stopping_config=RegretStoppingConfig(enabled=False))
for policy in ("uniform", "disagreement"):
    run = runner.run(design.scenario,
        NestedLangevinActiveAlgorithm(replace(base, query_policy=policy)))
    result = evaluate_active_run(run,
        ActiveEvaluationConfig(evaluate_trajectory=True),
        test_queries=design.test_queries)
    print(policy, result.final_angular_error_degrees, result.final_normalized_regret)
```

The helper also provides rare-informative 2D and coordinate-coverage 4D scenarios.
All use MIN, clean Y, and Gaussian parameter noise. Test designs are held
separately from the public scenario and cover informative ratios/pairs evenly.
These test weights intentionally differ from the imbalanced training pools.

For arbitrary pools, use `QuerySpaceConfig(kind="explicit", candidates=matrix)`;
nonzero rows are normalized and count is inferred. Supply fresh unit-norm test
queries separately to `evaluate_active_run`. Default `scenario` evaluation of
an explicit finite pool samples its rows WITH replacement, not unseen queries.

Use `RegretStoppingConfig(enabled=True)` for external validation stopping.
Invalid estimates reset the success streak and cannot trigger a successful stop.
Zero regret on a finite test set does not establish exact theta recovery.
Custom explicit final tests do not by themselves certify independence from a
stopping-validation set; that metadata is unknown when stopping was enabled.

CLI aliases `--algorithm nested-langevin` and `--algorithm diffusion` now select
the corrected defaults. Custom settings use a zero-argument module factory.

## Diagnostics and computational cost

Both backends store configuration, pre/post-observation ensembles, parameter
estimate, sample norms/spread, and forward timing/counts. Returned diagnostics
are copies. The corrected backend additionally reports density evaluations,
slice/radial updates, shrinks and mean squared jump. These are work/movement diagnostics,
NOT effective sample size, R-hat, or a convergence certificate.

Uniform selection records no candidate scores and zero candidate MIN calls.
Disagreement adds M times the number of available candidates. Binary/cardinality
MIN is exactly vectorized; other feasible sets use the ordinary public solver.
Forward counts report equivalent individual MIN problems, even for a batch.

Corrected sampling cost depends on slice shrinkage. No data minibatching,
approximate loss surrogate or parameter-dependent prior is introduced.
`InverseLossTarget` retains a generic `(theta,s)->decision` callback and accepts
an optional exact batch callback. Dataset copies and full-sum loss are preserved.

## Legacy v2 reproduction

```python
legacy = NestedLangevinConfig(
    sampler="projected_langevin", point_estimate="first",
    query_tie_breaking="first", theta_samples=4,
)
```

This backend implements the original projected inner subgradient Euler updates,
conditional-mean smoothed-score estimate, unprojected outer Euler updates, and
a fresh final conditional at the final updated u. It returns the last retained
latent z. For M trajectories, L levels, K outer updates, N inner steps and t
observations, it uses `M*(L*K+1)*N*t` sampling MIN problems.

Notebooks 11–13 explicitly select this legacy behavior. Their historical outputs
were not recomputed or relabeled as corrected results. Their saved configuration
displays predate the newly introduced fields.

## Validation scope

Regression tests cover original v2 formulas, subgradient signs/ties, observation
isolation, complete-Y validation, generic and batch forward agreement, all five
solver families, explicit query pools, invalid-estimate stopping, query-policy
RNG isolation, reproducibility across workers and serialization.

Numerical tests cover bounded uniform moments (box and ball), the biased 1D
conditional against quadrature, the full 1D marginal, and coupled 2D moments
against independent midpoint integration. Notebook 14 adds a grid-refinement
and mixing-budget check plus the 18 small, noisy, fixed-T query comparisons.

These checks are deliberately small. Do not infer universal convergence,
noise-model correctness, or guaranteed active-query superiority from them.
