"""Create notebook 17 from the executed three-algorithm presentation template."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "16_pedro_genious_score_base_eight_scenarios.ipynb"
DESTINATION = ROOT / "notebooks" / "17_four_algorithm_eight_scenarios.ipynb"


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"Notebook template text not found: {old[:100]!r}")
        text = text.replace(old, new)
    return text


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cells[0]["source"] = lines(r"""# Four-algorithm active inverse-optimization comparison — eight scenarios

This is the exact previous benchmark expanded from three algorithms to four. Every scenario, seed, horizon, candidate pool, parameter-noise realization, hidden test set, metric, and failure rule is unchanged.

| Algorithm | Next query S | Parameter estimate after observing Y |
|---|---|---|
| **Pedro algorithm** | Uniform random candidate | Incenter of the hard-consistency cone |
| **Genious Pedro** | Minimum normalized predicted decision margin | The same hard-consistency cone incenter |
| **Score base model** | Maximum disagreement across parameter samples | Mean of 16 parameter samples |
| **Uniform Online SAMD** | Uniform random candidate | One signed exponentiated mirror update of augmented suboptimality loss per observation |

Uniform Online SAMD is the online finite-oracle specialization of the paper's SAMD idea. Its loss-augmented competitor is enumerated exactly (epsilon=0), its signed 2d representation permits negative parameter coordinates, and it receives exactly one update per expert query. Its hyperparameters are frozen across all scenarios; it never receives theta-star, latent clean responses, or held-out evaluations.

**Protocol:** eight predeclared 4D/6D families, seeds 0–4, T=20, 120 candidate queries and 120 fresh held-out queries per test distribution. MIN expert, parameter noise, no observation noise (Y=X). All algorithms continue to T, even after zero regret or an invalid estimate. No large-dimensional experiments or training epochs are used.

F(theta,s,x)=theta dot (s*x). The true parameter is fixed within each run and shared by all four algorithms. Seeds vary theta and noise; pools and tests are independent of theta and shared. Standardized Gaussian parameter perturbations are paired across algorithms. The expert's perturbed parameter is normalized, which does not change hard-MIN decisions.

**Failures are part of the result.** Invalid estimates have no reported angular error or regret. We do not replace failure with 90/180 degrees, zero regret, or another estimator. Every valid-only aggregate displays its valid count out of five.

**Read without running:** all tables and plots are saved in this notebook. Each scenario includes every-step statistics, all five seed trajectories, query heatmaps, estimator diagnostics, and expandable full theta/S/Y tables.

**Rerun:** source-aware checkpoints live under `outputs/active/17_four_algorithm_comparison`. There are 160 complete runs: 8 scenarios × 5 seeds × 4 algorithms.

Finite sampler budgets and finite horizons are not convergence proofs. Zero regret on finite tests does not imply exact parameter recovery.
""")

    setup = "".join(cells[1]["source"])
    setup = replace_all(setup, [
        ("NestedLangevinConfig, PublicDecisionProblem, make_decision_space,\n    build_pedro_score_scenarios, run_three_algorithm_design,",
         "NestedLangevinConfig, OnlineSAMDConfig, PublicDecisionProblem, make_decision_space,\n    build_pedro_score_scenarios, run_four_algorithm_design,"),
        ('LABELS = ("Pedro algorithm", "Genious Pedro", "Score base model")',
         'LABELS = ("Pedro algorithm", "Genious Pedro", "Score base model", "Uniform Online SAMD")'),
        ('COLORS = {"Pedro algorithm":"#777777", "Genious Pedro":"#cc3311", "Score base model":"#0077bb"}',
         'COLORS = {"Pedro algorithm":"#777777", "Genious Pedro":"#cc3311", "Score base model":"#0077bb", "Uniform Online SAMD":"#009988"}'),
        ('OUTPUT = ROOT/"outputs"/"active"/"16_three_algorithm_comparison"',
         'SAMD_CFG = OnlineSAMDConfig()\nOUTPUT = ROOT/"outputs"/"active"/"17_four_algorithm_comparison"'),
        ('print("Exactly three algorithms:", LABELS)',
         'print("Exactly four algorithms:", LABELS)'),
        ('print("Planned runs: 8 x 5 x 3 = 120; all run through T=20.")',
         'print("Planned runs: 8 x 5 x 4 = 160; all run through T=20.")'),
    ])
    cells[1]["source"] = lines(setup)

    helpers = "".join(cells[3]["source"])
    helpers = replace_all(helpers, [
        ('short={LABELS[0]:"Pedro", LABELS[1]:"Genious", LABELS[2]:"Score base"}[label]',
         'short={LABELS[0]:"Pedro", LABELS[1]:"Genious", LABELS[2]:"Score base", LABELS[3]:"SAMD"}[label]'),
        ('fig,axes=plt.subplots(3,3,figsize=(15,9),layout="constrained")',
         'fig,axes=plt.subplots(4,3,figsize=(15,12),layout="constrained")'),
        ('            else:\n                values=[r["update_diagnostics"]["ensemble_spread"] for r in run["records"]]\n            axes[row,2].plot(range(1,T+1),values,label=f"seed {run[\'seed\']}")\n        axes[row,2].set(title="Incenter radius (invalid masked)" if label in LABELS[:2] else "Parameter-ensemble spread",',
         '            elif label == LABELS[2]:\n                values=[r["update_diagnostics"]["ensemble_spread"] for r in run["records"]]\n            else:\n                values=[r["update_diagnostics"]["theta_l1_norm"] for r in run["records"]]\n            axes[row,2].plot(range(1,T+1),values,label=f"seed {run[\'seed\']}")\n        diagnostic_title=("Incenter radius (invalid masked)" if label in LABELS[:2] else\n                          "Parameter-ensemble spread" if label == LABELS[2] else\n                          "SAMD estimate L1 norm")\n        axes[row,2].set(title=diagnostic_title,'),
        ('all 20 steps for ALL THREE algorithms', 'all 20 steps for ALL FOUR algorithms'),
        ('all 60 outputs; true theta', 'all 80 outputs; true theta'),
        ('pair=run_three_algorithm_design(design,OUTPUT,score_config=CFG,use_cache=True)',
         'pair=run_four_algorithm_design(design,OUTPUT,score_config=CFG,samd_config=SAMD_CFG,use_cache=True)'),
        ('{len(ALL_RUNS)}/120 complete runs loaded.',
         '{len(ALL_RUNS)}/160 complete runs loaded.'),
    ])
    cells[3]["source"] = lines(helpers)

    cells[20]["source"] = lines("""## Combined final results and recovery times

These tables include **all 160 runs**. Valid-only averages are conditional, and valid counts/failures are shown alongside them. Pairwise wins are calculated only when both compared estimates are valid; one-sided and two-sided failures are counted separately.

The recovery table lists every seed. “Not reached” is never dropped from success counts. “Sustained” refers only to the observed remainder through T=20 and does not guarantee future stability.
""")

    combined = "".join(cells[21]["source"])
    combined = replace_all(combined, [
        ("assert len(runs)==120", "assert len(runs)==160"),
        ('    else:\n        assert all(x["update_diagnostics"]["sampler_config"]["theta_samples"]==16 for x in r["records"])\n        for record in r["records"]:\n            np.testing.assert_allclose(record["theta_hat_after"],\n                np.mean(record["update_diagnostics"]["theta_samples"],axis=0))',
         '    elif r["algorithm_name"] == LABELS[2]:\n        assert all(x["update_diagnostics"]["sampler_config"]["theta_samples"]==16 for x in r["records"])\n        for record in r["records"]:\n            np.testing.assert_allclose(record["theta_hat_after"],\n                np.mean(record["update_diagnostics"]["theta_samples"],axis=0))\n    else:\n        assert r["metadata"]["comparison_estimator"]=="online signed exponentiated ASL"\n        assert all(x["update_diagnostics"]["epsilon"]==0.0 for x in r["records"])\n        assert all(x["update_diagnostics"]["update_count"]==i+1\n                   for i,x in enumerate(r["records"]))'),
        ('print("ALL OUTPUTS READY: 120/120 runs, 2,400 observed steps, 8 scenarios.")',
         'print("ALL OUTPUTS READY: 160/160 runs, 3,200 observed steps, 8 scenarios.")'),
        ('all five-seed curves, full theta/S/Y tables',
         'all five-seed curves, full theta/S/Y tables'),
    ])
    cells[21]["source"] = lines(combined)

    cells[22]["source"] = lines("""## Independent audit of failed hard-cone incenter estimates

A homogeneous cone always contains zero. A zero incenter radius can mean that the cone has become lower-dimensional, not that every nonzero feasible direction has disappeared.

The independent linear programs below audit Pedro and Genious Pedro at their first invalid step and at T. This audit applies only to their hard-cone estimators; Score Base and SAMD do not use this cone. A zero maximum strict margin with a nonzero feasible coordinate means **no full-dimensional interior, but nonzero feasible directions remain**. We still mark the incenter estimate invalid; no substitute estimator or metric is introduced.
""")

    cells[24]["source"] = lines("""## Interpretation boundaries

- This is the exact former benchmark with Uniform Online SAMD added; historical notebooks 15 and 16 remain unchanged.
- Pedro and Uniform Online SAMD both select S uniformly, so their comparison isolates the estimator/update rule.
- Genious Pedro and Pedro share the same incenter estimator, so their difference isolates query selection while their cones remain valid.
- Score Base differs in both estimator and query selection, so comparisons involving it are comparisons of complete algorithms.
- SAMD receives exactly one online mirror update per observation and uses fixed hyperparameters across every family and seed; it was not tuned per scenario.
- Failed estimates remain failures, not large artificial angles. Conditional averages must always be read with valid counts.
- More reliable decisions do not imply exact parameter recovery; angular error and regret remain separate.
- These eight families and five seeds provide empirical evidence, not a general convergence theorem.
""")

    for cell in cells:
        cell["id"] = cell.get("id", "cell").replace("three-alg", "four-alg")
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            cell["metadata"].pop("execution", None)

    DESTINATION.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
