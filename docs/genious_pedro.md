# Genious Pedro

Genious Pedro is the third algorithm in the active inverse-optimization laboratory.
Its parameter estimate is exactly Pedro's sequential hard-consistency cone
incenter. Only the next-query rule changes.

After at least one observation and a valid current incenter `theta_hat`, each
candidate query `s` is scored as follows:

1. Predict the public MIN decision under `theta_hat`.
2. For every distinct feasible alternative, compute the normalized objective
   margin. This is the Euclidean distance from `theta_hat` to the corresponding
   decision-boundary hyperplane.
3. Give `s` the smallest such distance.
4. Select a candidate with the globally smallest score, breaking exact score
   ties reproducibly with the algorithm RNG.

The first query is uniform because `D_0` is empty. If Pedro's incenter is later
invalid, query selection falls back to uniform while evaluation keeps the
estimate explicitly invalid. No replacement estimate is fabricated.

The implementation uses only the public candidate pool, public feasible decision
set, and public observations. It never receives the true parameter, parameter
noise realization, hidden test queries, a parameter distribution, or parameter
samples.

The exact score currently requires a finite enumerable decision space. This
covers every scenario in the predeclared eight-scenario comparison. Alternatives
whose feature difference is zero are skipped because they define no parameter
boundary. Candidates without any distinguishing alternative receive no finite
margin and cannot win unless every candidate is uninformative, in which case the
query falls back to uniform.

The saved two-algorithm benchmark and notebook 15 are intentionally unchanged.
They are not relabeled as three-algorithm results. The authorized three-way run
is saved separately in notebook 16 and `outputs/active/16_three_algorithm_comparison`.
It uses the identical scenarios, seeds, horizon, noise, tests, metrics and failure
rules. The original Pedro and Score Base scientific trajectories reproduce exactly.

Across the 40 Genious Pedro runs, 15 final estimates are valid and 5 meet both
the 5-degree and 0.01-regret targets. Some valid subset and knapsack runs are
highly accurate, but boundary-seeking under parameter noise frequently creates
contradictory hard constraints and a degenerate cone. This is reported as a
failure, not replaced by a softened estimate.
