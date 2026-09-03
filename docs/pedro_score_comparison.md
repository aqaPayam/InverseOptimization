# Pedro algorithm versus Score base model

The comparison contains exactly two methods:

| Name | Next S | theta_hat after each observed Y |
|---|---|---|
| Pedro algorithm | Uniform over candidate indices, with replacement | Sequential hard-consistency cone incenter |
| Score base model | Maximum optimizer disagreement over 16 parameter samples | Arithmetic mean of those 16 samples |

This compares COMPLETE algorithms, not query selection alone. Notebook 14 used
a separate uniform-query sample-mean control: those historical results are NOT
Pedro results. The corrected comparison is notebook 15.

## Fixed protocol

- Eight families; dimensions 4 or 6; seeds 0,1,2,3,4; T=20 for both methods.
- 120 fixed unit-norm candidate queries per family; repeats allowed.
- MIN expert; objective F(theta,s,x)=theta dot (s*x); clean observation Y=X.
- Isotropic Gaussian parameter perturbations with sigma=.02, except cases 7/8.
  The perturbed parameter is then unit-normalized as in the existing environment.
  This positive rescaling leaves hard-MIN decisions unchanged.
- The two methods receive identical theta*, candidate pools and standardized Gaussian
  perturbations per paired round. Their different S can produce different decisions;
  case 8 also changes the scale applied to the shared standardized perturbation.
- No early stopping or algorithm stop requests. Invalid Pedro estimates remain
  explicitly invalid while uniform querying continues up to T. No softened replacement.
- Score base uses the corrected Gaussian-augmented slice sampler, beta=20,
  box [-1,1]^d, variances (.5,.1,.02), 6 Gaussian sweeps/variance,
  4 conditional slice steps, 32 final target steps with radial refresh,
  16 samples, mean estimate, warm starts and randomized query-score ties.
- Pedro uses its original incenter settings (tolerance 1e-8, at most 2000
  SLSQP iterations). Every consistency-normal set in these small cases is exact.

All settings are fixed before the main run; no seed selection based on outcomes.
The estimator implementations have different costs: sample efficiency is compared
by expert-query count, while computer time is reported separately.

## Eight scenarios

| # | Family | d | Feasible decisions | Query design / noise |
|---|---|---|---|---|
| 1 | Connecting two groups | 6 | One-hot | Pairs (1,2),(2,3),(4,5),(5,6),(3,4); counts 27,27,27,27,12 |
| 2 | Several boundaries | 4 | One-hot | Reference pairs (1,2),(1,3),(1,4); 40 candidates each |
| 3 | Similar versus varied queries | 6 | Choose exactly 2 | 48 queries near each of two fixed centers, 24 dense queries |
| 4 | Ordinary balanced choice | 4 | One-hot | Dense uniform unit-sphere queries |
| 5 | Ordinary subset selection | 6 | Choose exactly 3 | Dense uniform unit-sphere queries |
| 6 | Small budget selection | 6 | Binary knapsack | Weights [1,2,2,3,3,4], capacity 6; dense queries |
| 7 | Stronger noise | 4 | One-hot | Identical to case 4 except sigma=.08 |
| 8 | Query-dependent noise | 4 | One-hot | Identical to case 4 except sigma(s)=.02+.08*abs(s[0]) |

Pair queries are normalized -q e_i - e_j, with q spanning [.25,2.25].
All other coordinates are exactly zero. The hidden parameters in cases 1/2
have positive coordinates, so the response compares the selected pair. The
comparison graph is connected; we do not ask an algorithm to infer an unavailable
relative scale between disconnected groups.

The two cluster centers in case 3 are normalized versions of
[-1,-.6,-.2,.2,.6,1] and [.3,-.8,.6,-.5,.9,-.2]. Add isotropic Gaussian
coordinate jitter of standard deviation .08, then renormalize. These queries
are correlated, not guaranteed to have identical decisions.

Pool RNGs use SeedSequence([271828, base_family_index]); held-out RNGs use
[314159, base_family_index]. Neither construction uses theta*. Hidden magnitudes
are drawn from U(.8,1.2) with SeedSequence([161803, base_family_index, seed]),
then normalized. Cases 1/2 are positive; remaining cases have alternating signs
[-,+,-,+,...]. Cases 4/7/8 share the same base index, theta*, pool and held-out
queries, isolating the noise change. Thus five seeds also vary theta*, rather
than repeating only one hidden direction.

## Held-out evaluation and failures

At every t, evaluate the post-observation estimate on 120 fixed fresh queries
under the CLEAN theta*. Algorithms never receive these tests.

- Ordinary tests follow each family's generating mixture, using stratified counts.
- Cases 1 and 3 ALSO have a separate balanced test set: 24 queries per pair for
  case 1, and 40 per cluster/diverse group for case 3. Neither test set is used to
  construct the query score or stop learning.
- Tests are fixed across paired methods, time steps and seeds. Their distribution
  is disclosed; do not mix ordinary and balanced regret into one score.
- Angular error measures direction, not unidentifiable positive scale.
- Regret is mean extra true objective cost, normalized by the feasible objective
  range at each test query. It is held-out regret at t, NOT cumulative query regret.
- First/sustained threshold times use 5 degrees and .01 regret. Sustained means
  all remaining observed steps through T; a hit only at T gives no future guarantee.
- Failed estimates have None angle/regret, NEVER arbitrary 90/180-degree penalties.
  Means/std over valid runs are labeled CONDITIONAL and always shown with valid n/5.
  Failure rates and per-seed traces are retained; no claim of superiority is based
  solely on a favorable survivor-only mean.

The loss-based Score base target is not the exact Gaussian-noise likelihood.
Finite sampling budgets remain approximate, even with target-invariant kernels.
This small comparison is evidence about these settings, not a convergence theorem.

The notebook also independently audits the two failed Pedro incenter cases with
linear programs. Both have zero strict interior margin but retain nonzero feasible
directions: the cone is lower-dimensional, not literally empty. The original stored
diagnostic saying "no valid nonzero parameter direction" overstates what the zero
estimate check establishes. The notebook explains this distinction and reports
invalid incenter estimates without replacing them or altering the measured results.

## Reproduce or resume

```powershell
python scripts/run_pedro_score_comparison.py
python scripts/execute_notebooks.py notebooks/15_pedro_vs_score_base_eight_scenarios.ipynb --timeout 1200 --in-place
```

The first command saves 80 raw JSON trajectories in
outputs/active/15_pedro_vs_score_base. The notebook can also run the experiments
itself if these files are absent. Matching checkpoints are reused; the fingerprint
includes all active-module source files, settings, scenarios and hidden tests.
Incomplete writes are never treated as completed runs. Unexpected execution errors
raise explicitly rather than being misreported as an ordinary cone failure.

Small binary knapsack optimization is exact cached enumeration, not a surrogate:
there are only 64 possible binary vectors before filtering feasibility. Problems
above the conservative enumeration cap retain their MILP solver.

The notebook preserves all rendered results and plots for later reading without
rerunning anything. Raw JSON provides full floating-point precision; displayed
parameter/query vectors are rounded for readability.
