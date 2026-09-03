# Saved experiment outputs

These are versioned snapshots of the synthetic experiments and examples in this
repository. The executed notebooks under `notebooks/` remain the main entry point
for reading explanations, tables, plots, and per-step results without rerunning.

## Current comparison

`active/15_pedro_vs_score_base/` contains all 80 complete JSON trajectories from
eight scenarios, five paired seeds, and two algorithms at T=20, plus the angular
error and regret overview figures. See notebook 15 and
`docs/pedro_score_comparison.md` for settings and interpretation.

Pedro uses uniform queries and a hard-cone incenter. Score base uses disagreement
queries and the mean of 16 parameter samples. Invalid estimates remain explicit
failures, not fabricated angular errors or zero regrets.

## Historical snapshots

- `active/14_corrected_diffusion/` is the historical matched-estimator query-policy
  comparison. Its uniform sample-mean control is NOT the Pedro algorithm.
- `active/12_diffusion_comparison/` preserves the earlier diffusion implementation's
  results. These should not be interpreted as results of the current corrected sampler.
- Other active folders preserve previous examples and smoke checks.
- `experiments/` preserves the earlier general inverse-optimization examples,
  datasets, reports, plots, and tables.

Snapshots are retained as originally generated; not every historical result was
produced by the current source version. Source-aware cache checks in the current
comparison determine whether a checkpoint can be reused. New generated runs and
temporary outputs remain ignored by default; these existing snapshots are
explicitly versioned for reproducibility and sharing.
