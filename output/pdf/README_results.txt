PEDRO VS SCORE BASE - COMPLETE SAVED RESULTS

Open Pedro_vs_Score_Base_Complete_Results.pdf first.
Pages 1-8: main comparisons. Pages 9-16: every scenario and seed.
Pages 17-36: all 1,600 per-step metric records.

per_run_results.csv: 80 rows, final metrics, first/sustained thresholds, runtime.
per_step_results.csv: 1,600 rows, metrics and full-precision theta/S/Y vectors.
Vector-valued CSV cells are JSON arrays. Invalid numeric metrics are blank.
CSV normalized regrets are fractions (0.01 = 1%); PDF tables use percentages.
Balanced-test fields are blank for scenarios without that extra test distribution.
Raw trajectories preserve all original diagnostics and configuration in raw_runs/.
The legacy degenerate-cone failure message is overstrong; see the PDF audit.
No estimate or experimental result was replaced to prepare this report.

Primary objective: F(theta,s,x) = theta dot (s*x). MIN expert, noisy parameter,
no observation noise. T=20; five paired seeds per scenario. Both algorithms run
through T. Thresholds use clean held-out angle <=5 degrees and regret <=0.01.
Reported success times are retrospective, not executable stopping rules.

Source commit: 193b2e91318c063d4a236f928623716d7c2f7519
Repository: https://github.com/aqaPayam/InverseOptimization
