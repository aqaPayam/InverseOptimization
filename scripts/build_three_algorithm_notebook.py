"""Create notebook 16 by preserving notebook 15's presentation structure."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "15_pedro_vs_score_base_eight_scenarios.ipynb"
DESTINATION = ROOT / "notebooks" / "16_pedro_genious_score_base_eight_scenarios.ipynb"


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def replace_all(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"Notebook template text not found: {old[:80]!r}")
        text = text.replace(old, new)
    return text


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cells[0]["source"] = lines("""# Pedro, Genious Pedro, and Score Base — eight scenarios

This is the exact previous benchmark expanded from two algorithms to three. Every scenario, seed, horizon, candidate pool, parameter-noise realization, hidden test set, metric, and failure rule is unchanged.

| Algorithm | Next query S | Parameter estimate after observing Y |
|---|---|---|
| **Pedro algorithm** | Uniform random candidate | Incenter of the hard-consistency cone |
| **Genious Pedro** | Minimum normalized predicted decision margin | The same hard-consistency cone incenter |
| **Score base model** | Maximum disagreement across parameter samples | Mean of 16 parameter samples |

Genious Pedro uses no posterior or hidden information. At D0 it selects uniformly; after a valid incenter it scores every candidate using the distance to the closest predicted decision boundary. If the cone estimate becomes invalid, it remains explicitly invalid and query selection falls back to uniform.

**Protocol:** eight predeclared 4D/6D families, seeds 0–4, T=20, 120 candidate queries and 120 fresh held-out queries per test distribution. MIN expert, parameter noise, no observation noise (Y=X). All algorithms continue to T, even after zero regret or an invalid estimate. No large-dimensional experiments or training epochs are used.

F(theta,s,x)=theta dot (s*x). The true parameter is fixed within each run and shared by all three algorithms. Seeds vary theta and noise; pools and tests are independent of theta and shared. Standardized Gaussian parameter perturbations are paired across algorithms. The expert's perturbed parameter is normalized, which does not change hard-MIN decisions.

**Failures are part of the result.** Invalid estimates have no reported angular error or regret. We do not replace failure with 90/180 degrees, zero regret, or a softened incenter. Every valid-only aggregate displays its valid count out of five.

**Read without running:** all tables and plots are saved in this notebook. Each scenario includes every-step statistics, all five seed trajectories, query heatmaps, estimator diagnostics, and expandable full theta/S/Y tables.

**Rerun:** source-aware checkpoints live under `outputs/active/16_three_algorithm_comparison`. There are 120 complete runs: 8 scenarios × 5 seeds × 3 algorithms.

Finite sampler budgets are not convergence-certified. Zero regret on finite tests does not imply exact parameter recovery. The cone audit distinguishes loss of a full-dimensional interior from disappearance of every nonzero feasible direction.
""")

    setup = "".join(cells[1]["source"])
    setup = replace_all(setup, [
        ("build_pedro_score_scenarios, run_pedro_score_design,",
         "build_pedro_score_scenarios, run_three_algorithm_design,"),
        ('LABELS = ("Pedro algorithm", "Score base model")',
         'LABELS = ("Pedro algorithm", "Genious Pedro", "Score base model")'),
        ('COLORS = {"Pedro algorithm":"#777777", "Score base model":"#0077bb"}',
         'COLORS = {"Pedro algorithm":"#777777", "Genious Pedro":"#cc3311", "Score base model":"#0077bb"}'),
        ('OUTPUT = ROOT/"outputs"/"active"/"15_pedro_vs_score_base"',
         'OUTPUT = ROOT/"outputs"/"active"/"16_three_algorithm_comparison"'),
        ('print("Exactly two algorithms:", LABELS)',
         'print("Exactly three algorithms:", LABELS)'),
        ('print("Planned runs: 8 x 5 x 2 = 80; all run through T=20.")',
         'print("Planned runs: 8 x 5 x 3 = 120; all run through T=20.")'),
    ])
    cells[1]["source"] = lines(setup)

    helpers = "".join(cells[3]["source"])
    helpers = replace_all(helpers, [
        ('short="Pedro" if label==LABELS[0] else "Score base"',
         'short={LABELS[0]:"Pedro", LABELS[1]:"Genious", LABELS[2]:"Score base"}[label]'),
        ('fig,axes=plt.subplots(2,3,figsize=(15,6),layout="constrained")',
         'fig,axes=plt.subplots(3,3,figsize=(15,9),layout="constrained")'),
        ('if row==0:\n                values=[r["update_diagnostics"]["incenter_radius"] if ok else np.nan',
         'if label in LABELS[:2]:\n                values=[r["update_diagnostics"]["incenter_radius"] if ok else np.nan'),
        ('axes[row,2].set(title="Incenter radius (invalid masked)" if row==0 else "Parameter-ensemble spread",',
         'axes[row,2].set(title="Incenter radius (invalid masked)" if label in LABELS[:2] else "Parameter-ensemble spread",'),
        ('all 20 steps for BOTH algorithms', 'all 20 steps for ALL THREE algorithms'),
        ('all 40 outputs; true theta', 'all 60 outputs; true theta'),
        ('pair=run_pedro_score_design(design,OUTPUT,score_config=CFG,use_cache=True)',
         'pair=run_three_algorithm_design(design,OUTPUT,score_config=CFG,use_cache=True)'),
        ('{len(ALL_RUNS)}/80 complete runs loaded.',
         '{len(ALL_RUNS)}/120 complete runs loaded.'),
    ])
    cells[3]["source"] = lines(helpers)

    cells[20]["source"] = lines("""## Combined final results and recovery times

These tables include **all 120 runs**. Valid-only averages are conditional, and valid counts/failures are shown alongside them. Pairwise wins are calculated only when both compared estimates are valid; one-sided and two-sided failures are counted separately.

The recovery table lists every seed. “Not reached” is never dropped from success counts. “Sustained” refers only to the observed remainder through T=20 and does not guarantee future stability.
""")

    cells[21]["source"] = lines('''runs=list(ALL_RUNS.values())
assert len(runs)==120
assert {r["algorithm_name"] for r in runs}==set(LABELS)
assert all(len(r["records"])==T and r["error"] is None for r in runs)
assert all(not r["metadata"]["external_stopping_enabled"] for r in runs)
for r in runs:
    if r["algorithm_name"] in LABELS[:2]:
        assert r["metadata"]["comparison_estimator"]=="incenter"
        assert all("theta_samples" not in x["update_diagnostics"] for x in r["records"])
    else:
        assert all(x["update_diagnostics"]["sampler_config"]["theta_samples"]==16 for x in r["records"])
        for record in r["records"]:
            np.testing.assert_allclose(record["theta_hat_after"],
                np.mean(record["update_diagnostics"]["theta_samples"],axis=0))
    for ev in r["metadata"]["evaluations_by_distribution"].values():
        for valid,angle,regret in zip(ev["valid_estimate_history"],
                ev["angular_error_history_degrees"],ev["normalized_regret_history"]):
            assert (angle is not None and regret is not None) if valid else (angle is None and regret is None)

summaries=[]
for design in DESIGNS[0]:
    subset=[r for r in runs if r["metadata"]["comparison_family"]==design.family]
    table=conditional_summary(subset,design.test_queries)
    table.insert(0,"scenario",design.title)
    summaries.append(table)
summary=pd.concat(summaries,ignore_index=True)
display(summary.round(4))

recovery=[]
for r in sorted(runs,key=lambda r:(r["metadata"]["comparison_family"],r["seed"],r["algorithm_name"])):
    for distribution,ev in r["metadata"]["evaluations_by_distribution"].items():
        recovery.append({
            "scenario":r["metadata"]["comparison_title"],"seed":r["seed"],
            "algorithm":r["algorithm_name"],"test":distribution,
            "final status":ev["final_status"],
            "first <=5 deg":ev["first_angular_threshold_step"],
            "sustained <=5 deg":ev["stable_angular_threshold_step"],
            "first regret<=.01":ev["first_threshold_step"],
            "sustained regret<=.01":ev["stable_threshold_step"],
            "first joint":ev["first_joint_threshold_step"],
            "sustained joint":ev["stable_joint_threshold_step"],
            "first invalid t":r["metadata"]["first_invalid_step"],
        })
recovery=pd.DataFrame(recovery)
display(HTML(recovery.to_html(index=False,na_rep="not reached / none",float_format=lambda x:f"{x:.0f}")))

from itertools import combinations
pairs=[]
for design in DESIGNS[0]:
    family=design.family
    for distribution in design.test_queries:
        for left,right in combinations(LABELS,2):
            row={"scenario":design.title,"test":distribution,
                 "left":left,"right":right,"both valid":0,
                 "left invalid only":0,"right invalid only":0,"both invalid":0,
                 "left lower angle":0,"right lower angle":0,"angle tied":0,
                 "left lower regret":0,"right lower regret":0,"regret tied":0}
            for seed in SEEDS:
                a=evaluation(ALL_RUNS[(family,seed,left)],distribution)
                b=evaluation(ALL_RUNS[(family,seed,right)],distribution)
                av,bv=a["final_estimate_valid"],b["final_estimate_valid"]
                if av and bv:
                    row["both valid"]+=1
                    for metric,name in [("final_angular_error_degrees","angle"),("final_normalized_regret","regret")]:
                        diff=a[metric]-b[metric]
                        row[("left lower "+name) if diff < -1e-12 else
                            ("right lower "+name) if diff > 1e-12 else (name+" tied")]+=1
                elif not av and bv: row["left invalid only"]+=1
                elif av and not bv: row["right invalid only"]+=1
                else: row["both invalid"]+=1
            pairs.append(row)
display(pd.DataFrame(pairs))
print("ALL OUTPUTS READY: 120/120 runs, 2,400 observed steps, 8 scenarios.")
print("Per-step mean tables, all five-seed curves, full theta/S/Y tables and raw checkpoints are preserved.")
print("Final invalid runs by algorithm:",{label:sum(not evaluation(r)["final_estimate_valid"]
      for r in runs if r["algorithm_name"]==label) for label in LABELS})
''')

    cells[22]["source"] = lines("""## Independent audit of failed hard-cone incenter estimates

A homogeneous cone always contains zero. A zero incenter radius can mean that the cone has become lower-dimensional, not that every nonzero feasible direction has disappeared.

The independent linear programs below audit both Pedro algorithms at their first invalid step and at T. A zero maximum strict margin with a nonzero feasible coordinate means **no full-dimensional interior, but nonzero feasible directions remain**. We still mark the incenter estimate invalid; no substitute estimator or metric is introduced.
""")
    cells[23]["source"] = lines('''from scipy.optimize import linprog

cone_audit=[]
for run in runs:
    if run["algorithm_name"] not in LABELS[:2] or run["metadata"]["first_invalid_step"] is None:
        continue
    d=len(run["true_theta"])
    design=next(item for item in DESIGNS[run["seed"]]
                if item.family==run["metadata"]["comparison_family"])
    problem=PublicDecisionProblem(make_decision_space(
        design.scenario.decision_space,d,np.random.default_rng(0)))
    alternatives=np.asarray(problem.enumerate_decisions(),dtype=float)
    for t in sorted({run["metadata"]["first_invalid_step"],T}):
        normals=np.vstack([np.asarray(record["query"])*
            (np.asarray(record["observed_decision"])-alternatives)
            for record in run["records"][:t]])
        lengths=np.linalg.norm(normals,axis=1)
        normals=normals[lengths>1e-12]/lengths[lengths>1e-12,None]
        margin=linprog(np.r_[np.zeros(d),-1.],A_ub=np.c_[normals,np.ones(len(normals))],
            b_ub=np.zeros(len(normals)),bounds=[(-1,1)]*d+[(0,None)],method="highs")
        extrema=[linprog(sign*np.eye(d)[j],A_ub=normals,b_ub=np.zeros(len(normals)),
            bounds=[(-1,1)]*d,method="highs") for j in range(d) for sign in (-1,1)]
        assert margin.success and all(result.success for result in extrema)
        largest=max(-result.fun for result in extrema)
        cone_audit.append({"scenario":run["metadata"]["comparison_title"],
            "algorithm":run["algorithm_name"],"seed":run["seed"],"t":t,
            "maximum strict margin (box)":-margin.fun,
            "largest feasible coordinate magnitude":largest,
            "nonzero direction exists":largest>1e-8,
            "reported estimate status":run["records"][t-1]["update_diagnostics"]["estimate_status"]})
display(pd.DataFrame(cone_audit).round(8))
print("This audit clarifies failure geometry; no run, estimate, or metric was replaced.")
''')

    cells[24]["source"] = lines("""## Interpretation boundaries

- This is the exact former benchmark with Genious Pedro added; historical notebook 15 remains unchanged.
- Genious Pedro and Pedro share the same estimator, so their difference isolates query selection while their cones remain valid.
- Genious Pedro deliberately targets close predicted boundaries. Under parameter noise this can be informative, but it can also generate contradictory hard constraints and destroy a full-dimensional cone.
- Score Base differs in both estimator and query selection, so comparisons involving it are comparisons of complete algorithms.
- Failed estimates remain failures, not large artificial angles. Conditional averages must always be read with valid counts.
- More reliable decisions do not imply exact parameter recovery; angular error and regret remain separate.
- These eight families and five seeds provide empirical evidence, not a general convergence theorem.
""")

    for cell in cells:
        cell["id"] = cell.get("id", "cell").replace("pedro-score", "three-alg")
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            cell["metadata"].pop("execution", None)

    notebook["metadata"].setdefault("language_info", {})
    DESTINATION.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
