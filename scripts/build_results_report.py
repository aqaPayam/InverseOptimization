"""Publish the saved eight-scenario results; never runs an experiment.

Run --figures-only with the scientific Python environment, then run without
arguments with a Python environment containing reportlab, pypdf and Pillow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import zipfile
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/active/15_pedro_vs_score_base"
OUT = ROOT / "output/pdf"
TMP = ROOT / "tmp/pdfs/pedro_score_report"
FIG = TMP / "figures"
PDF = OUT / "Pedro_vs_Score_Base_Complete_Results.pdf"
P, S = "Pedro algorithm", "Score base model"
FAMILIES = ["bridge-6d", "boundaries-4d", "redundancy-6d", "balanced-choice-4d",
            "balanced-subset-6d", "knapsack-6d", "strong-noise-4d", "query-noise-4d"]
TITLES = ["Connecting groups", "Several boundaries", "Similar versus varied queries",
          "Ordinary balanced choice", "Ordinary subset selection", "Budget-constrained selection",
          "Stronger parameter noise", "Query-dependent noise"]
SHORT = ["Connecting groups", "Several boundaries", "Similar / varied queries", "Balanced choice",
         "Subset selection", "Budget selection", "Stronger noise", "Query-dependent noise"]
SETTINGS = [
    "6D; choose one. Four within-group pairs have 27 candidates each; the connecting pair has 12. Parameter sigma = 0.02.",
    "4D; choose one. Three reference-coordinate comparisons have 40 candidate ratios each. Parameter sigma = 0.02.",
    "6D; choose two. Two fixed query profiles have 48 nearby candidates each; 24 candidates are diverse. Parameter sigma = 0.02.",
    "4D; choose one. Dense unit-sphere query candidates. Parameter sigma = 0.02.",
    "6D; choose exactly three of six. Dense unit-sphere query candidates. Parameter sigma = 0.02.",
    "6D; binary knapsack. Weights [1,2,2,3,3,4], capacity 6; exact small-set MIN. Parameter sigma = 0.02.",
    "4D; same true parameters, candidates and tests as case 4. Parameter sigma increases to 0.08.",
    "4D; same true parameters, candidates and tests as case 4. Parameter sigma(s) = 0.02 + 0.08 * abs(s[0]).",
]
INTERPRETATIONS = [
    "Score base finishes with lower mean error and all five runs meet the joint target. Early target hits are not always stable. Connecting-query counts are 14/100 for Score base and 10/100 for Pedro, so the result is not explained simply by a very large increase in bridge-query frequency.",
    "Pedro is better at t=10 on the valid-only averages (17.89 versus 29.49 degrees), while Score base finishes better (1.72 versus 10.20 degrees). This is evidence against claiming uniform superiority at every intermediate time. One Pedro incenter becomes invalid.",
    "Pedro largely plateaus near 30 degrees, while Score base continues improving. Score base chooses 34/100 diverse queries versus Pedro's 19/100. Only two Score base runs finish within 5 degrees despite all five having low regret: decision quality and parameter recovery are different outcomes.",
    "The advantage persists without a deliberately rare query group. Score base finishes near 2 degrees on average; all five runs pass the joint target. Pedro finishes near 10.7 degrees and none pass the angular target. One Score base run has zero final regret on the finite test set, not exact parameter recovery.",
    "This is the strongest ordinary case for Pedro's decision quality: both methods finish below 1% regret in every run. Score base still estimates the parameter more accurately on average. Pedro has slightly lower final angular error for seed 0, but Score base has lower regret for that seed.",
    "Score base substantially improves final regret, but only two of five runs finish within 5 degrees. All five nevertheless pass the regret target. The small knapsack MIN is solved exactly; no large training job or approximate forward surrogate is involved.",
    "Compared with case 4, stronger parameter noise raises Score base's mean final angular error from 2.01 to 5.18 degrees. It still outperforms Pedro, but only three of five runs meet the joint target at T. Active queries can be informative and also sensitive to parameter perturbations.",
    "All five Score base runs meet the final joint target. Pedro's full query/observation sequences match case 7 for these seeds, despite different perturbed expert parameters; its identical results are therefore expected. Different parameter perturbations need not change a discrete MIN decision.",
]


def load_runs():
    files = sorted(SOURCE.glob("*.json"))
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    assert len(runs) == 80
    runs.sort(key=lambda r: (FAMILIES.index(r["metadata"]["comparison_family"]),
                             r["seed"], (P, S).index(r["algorithm_name"])))
    assert len({(r["metadata"]["comparison_family"], r["seed"], r["algorithm_name"]) for r in runs}) == 80
    for r in runs:
        assert len(r["records"]) == 20 and r["error"] is None and not r["stopped_early"]
        assert not r["metadata"]["external_stopping_enabled"]
        assert all(x["true_decision"] == x["observed_decision"] for x in r["records"])
        for e in r["metadata"]["evaluations_by_distribution"].values():
            assert e["test_query_count"] == 120
            flags = []
            for valid, angle, regret in zip(e["valid_estimate_history"], e["angular_error_history_degrees"], e["normalized_regret_history"]):
                assert ((angle is not None and regret is not None) if valid else (angle is None and regret is None))
                flags.append(bool(valid and angle <= 5 and regret <= .01))
            first = next((i+1 for i, v in enumerate(flags) if v), None)
            stable = next((i+1 for i in range(20) if all(flags[i:])), None)
            assert first == e["first_joint_threshold_step"] and stable == e["stable_joint_threshold_step"]
    return runs, files


def ev(r, distribution="ordinary"):
    return r["metadata"]["evaluations_by_distribution"][distribution]


def group(runs, family, label):
    return [r for r in runs if r["metadata"]["comparison_family"] == family and r["algorithm_name"] == label]


def stats(values):
    a = [x for x in values if x is not None and math.isfinite(x)]
    return (statistics.mean(a), statistics.stdev(a) if len(a) > 1 else 0., len(a)) if a else (None, None, 0)


def fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def pm(values, scale=1, digits=2):
    m, sd, n = stats(values)
    return "-" if not n else f"{m*scale:.{digits}f} +/- {sd*scale:.{digits}f}"


def final_good(r):
    e = ev(r)
    return bool(e["final_estimate_valid"] and e["final_angular_error_degrees"] <= 5 and e["final_normalized_regret"] <= .01)


def series_flags(r):
    e = ev(r)
    return [bool(v and a <= 5 and z <= .01) for v, a, z in zip(e["valid_estimate_history"], e["angular_error_history_degrees"], e["normalized_regret_history"])]


def make_figures(runs):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.grid": True, "grid.alpha": .17, "savefig.facecolor": "white"})
    colors = {P: "#6B7280", S: "#007A9A"}

    def curves(ax, f, key, distribution="ordinary", annotate=True):
        scale = 100 if key == "normalized_regret_history" else 1
        for label in (P, S):
            a = np.array([ev(r, distribution)[key] for r in group(runs, f, label)], dtype=float)*scale
            n = np.isfinite(a).sum(axis=0)
            means = np.divide(np.nansum(a, axis=0), n, out=np.full(20, np.nan), where=n>0)
            for row in a:
                ax.plot(range(1, 21), row, c=colors[label], alpha=.22, lw=.7)
            ax.plot(range(1, 21), means, c=colors[label], lw=1.9, label=label)
        ax.set(xlim=(1,20), xticks=[1,5,10,15,20], xlabel="Observed query count t")
        if scale == 100:
            ax.set_yscale("symlog", linthresh=.01)
            ax.set_ylim(0,110)
            ax.set_yticks([0,.01,.1,1,10,100])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
            ax.set_ylabel("Normalized regret (%)")
            ax.axhline(1, c="#D97706", ls="--", lw=.8)
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Angular error (degrees)")
            ax.axhline(5, c="#D97706", ls="--", lw=.8)
        if annotate and f in FAMILIES[:2]:
            t = 9 if f == FAMILIES[0] else 6
            ax.text(.98,.96,f"Pedro: n=4/5 from t={t}",transform=ax.transAxes,ha="right",va="top",fontsize=7)

    for key, name in [("angular_error_history_degrees","angle"),("normalized_regret_history","regret")]:
        fig, axes = plt.subplots(4,2,figsize=(8,9.4),layout="constrained")
        for i, (f, ax) in enumerate(zip(FAMILIES, axes.flat)):
            curves(ax,f,key)
            ax.set_title(f"{i+1}. {SHORT[i]}",loc="left",fontweight="bold")
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="outside upper center",ncol=2,frameon=False)
        fig.savefig(FIG/f"{name}.png",dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(4,2,figsize=(8,9.4),layout="constrained")
    for i, (f, ax) in enumerate(zip(FAMILIES, axes.flat)):
        for label in (P,S):
            a = np.array([series_flags(r) for r in group(runs,f,label)])
            ax.step(range(1,21),a.sum(axis=0),where="mid",c=colors[label],lw=1.8,label=label)
            ax.step(range(1,21),np.maximum.accumulate(a,axis=1).sum(axis=0),where="mid",c=colors[label],lw=1.2,ls=":")
        ax.set(title=f"{i+1}. {SHORT[i]}",xlim=(1,20),ylim=(-.1,5.15),xticks=[1,5,10,15,20],yticks=range(6),xlabel="Observed query count t",ylabel="Successful runs out of 5")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside upper center",ncol=2,frameon=False)
    fig.savefig(FIG/"success.png",dpi=200); plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(8,5.6),layout="constrained")
    for col, index in enumerate([0,2]):
        f = FAMILIES[index]
        curves(axes[0,col],f,"normalized_regret_history","balanced")
        axes[0,col].set_title(f"{index+1}. Balanced-test regret",loc="left")
        labels = (['pair 1-2','pair 2-3','pair 4-5','pair 5-6','bridge 3-4'] if index==0 else ['profile 1','profile 2','diverse'])
        for k,label in enumerate((P,S)):
            counts = np.bincount([g for r in group(runs,f,label) for g in r["metadata"]["selected_candidate_groups"]],minlength=len(labels))
            axes[1,col].bar(np.arange(len(labels))+(k-.5)*.36,counts,width=.36,color=colors[label],label=label)
        axes[1,col].set(xticks=range(len(labels)),xticklabels=labels,ylabel="Selected queries out of 100",title=f"{index+1}. Query allocation")
        axes[1,col].tick_params(axis="x",rotation=25,labelsize=7)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside upper center",ncol=2,frameon=False)
    fig.savefig(FIG/"balanced.png",dpi=200); plt.close(fig)

    for i,f in enumerate(FAMILIES):
        fig, axes=plt.subplots(1,2,figsize=(8,2.65),layout="constrained")
        curves(axes[0],f,"angular_error_history_degrees")
        curves(axes[1],f,"normalized_regret_history")
        fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncol=2,frameon=False,fontsize=8)
        fig.savefig(FIG/f"case-{i+1}.png",dpi=200); plt.close(fig)
    print("Rendered 12 scientific figures from saved trajectories only.")


def export_data(runs, files):
    OUT.mkdir(parents=True,exist_ok=True)
    rows, summaries = [], []
    for r in runs:
        base = {"scenario_number":FAMILIES.index(r["metadata"]["comparison_family"])+1,
                "scenario":r["metadata"]["comparison_family"],"seed":r["seed"],"algorithm":r["algorithm_name"]}
        summary = dict(base, runtime_seconds=r["runtime_seconds"], true_theta=json.dumps(r["true_theta"]),
                       first_invalid_step=r["metadata"]["first_invalid_step"],
                       noise_changed_decisions=r["metadata"]["parameter_noise_decision_flips"])
        for name in ("ordinary","balanced"):
            e = r["metadata"]["evaluations_by_distribution"].get(name,{})
            for k in ["final_angular_error_degrees","final_normalized_regret","final_zero_regret_rate",
                      "final_estimate_valid","final_status","first_threshold_step","stable_threshold_step",
                      "first_angular_threshold_step","stable_angular_threshold_step",
                      "first_joint_threshold_step","stable_joint_threshold_step"]:
                summary[f"{name}_{k}"] = e.get(k)
        summaries.append(summary)
        for t,rec in enumerate(r["records"]):
            e=ev(r); b=r["metadata"]["evaluations_by_distribution"].get("balanced")
            row=dict(base,step=t+1,estimate_valid=e["valid_estimate_history"][t],
                     estimate_status=e["estimate_status_history"][t],
                     angular_error_degrees=e["angular_error_history_degrees"][t],
                     normalized_regret=e["normalized_regret_history"][t],
                     balanced_normalized_regret=b["normalized_regret_history"][t] if b else None,
                     zero_regret_rate=e["zero_regret_rate_history"][t],
                     maximum_normalized_regret=e["maximum_normalized_regret_history"][t],
                     joint_target_met=series_flags(r)[t],
                     candidate_group=r["metadata"]["selected_candidate_groups"][t],
                     candidate_index=rec["action_diagnostics"]["candidate_index"],
                     incenter_radius=rec["update_diagnostics"].get("incenter_radius"),
                     ensemble_spread=rec["update_diagnostics"].get("ensemble_spread"),
                     legacy_failure_reason=rec["update_diagnostics"].get("failure_reason"))
            for k in ["theta_hat_before","theta_hat_after","query","observed_decision","true_decision","true_theta","expert_parameter"]:
                row[k]=json.dumps(rec[k],separators=(",",":"))
            rows.append(row)
    assert len(rows)==1600 and len(summaries)==80
    for name,records in [("per_step_results.csv",rows),("per_run_results.csv",summaries)]:
        with (OUT/name).open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    manifest = {"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "generated_date":date.today().isoformat(),"runs":80,"steps":1600,
                "source_files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/"source_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def make_pdf(runs, manifest):
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    for name,file in [("Report","arial.ttf"),("ReportBold","arialbd.ttf"),("ReportItalic","ariali.ttf")]:
        pdfmetrics.registerFont(TTFont(name,str(Path("C:/Windows/Fonts")/file)))
    pdfmetrics.registerFontFamily("Report",normal="Report",bold="ReportBold",italic="ReportItalic",boldItalic="ReportBold")
    navy=colors.HexColor("#102E43"); teal=colors.HexColor("#007A9A")
    ink=colors.HexColor("#273746"); muted=colors.HexColor("#586979")
    pale=colors.HexColor("#F0F5F8"); line=colors.HexColor("#D7E2E9")
    c=Canvas(str(PDF),pagesize=A4,pageCompression=1)
    c.setTitle("Pedro vs. Score Base - Complete Experimental Results")
    c.setAuthor("Payam | Inverse Optimization")
    c.setSubject("80 saved runs; eight noisy MIN scenarios; complete per-step metrics and empirical convergence comparison")
    page=0; W,H=A4; M=40; CW=W-2*M; total=36

    def para(text,x,y,width,size=10,leading=None,color=ink,bold=False):
        style=ParagraphStyle("p",fontName="ReportBold" if bold else "Report",fontSize=size,
                             leading=leading or size*1.35,textColor=color)
        p=Paragraph(text,style); _,height=p.wrap(width,1000)
        if y-height<38: raise ValueError(f"Paragraph below footer on page {page}: {text[:80]}")
        p.drawOn(c,x,y-height)
        return y-height

    def footer():
        c.setStrokeColor(line); c.line(M,30,W-M,30)
        c.setFillColor(muted); c.setFont("Report",7)
        c.drawString(M,18,"PEDRO VS SCORE BASE  |  SAVED EXPERIMENT RESULTS")
        c.drawRightString(W-M,18,f"{page} / {total}")

    def new(title,kicker="RESULTS",wide=False):
        nonlocal page,W,H,CW
        if page: footer(); c.showPage()
        page+=1; W,H=landscape(A4) if wide else A4; CW=W-2*M
        c.setPageSize((W,H)); c.bookmarkPage(f"page-{page}")
        c.addOutlineEntry(title,f"page-{page}",level=0 if page<=8 else 1)
        c.setFillColor(teal); c.setFont("ReportBold",8); c.drawString(M,H-29,kicker)
        para(title,M,H-43,CW,18 if wide else 22,leading=25,color=navy,bold=True)
        c.setStrokeColor(line); c.line(M,H-76 if not wide else H-68,W-M,H-76 if not wide else H-68)
        return H-93

    def table(headers,rows,widths,x,y,font=8.4,rowheight=20,headerheight=29,pad=4):
        style=ParagraphStyle("cell",fontName="Report",fontSize=font,leading=font+1.3,textColor=ink)
        hs=ParagraphStyle("head",parent=style,fontName="ReportBold",textColor=colors.white)
        data=[[Paragraph(escape(str(v)).replace("\n","<br/>"),hs) for v in headers]]
        for row in rows:
            data.append([Paragraph(escape(str(v)).replace("\n","<br/>"),style) for v in row])
        heights=[headerheight]+[rowheight]*len(rows)
        for ri,row in enumerate(data):
            for ci,p in enumerate(row):
                _,h=p.wrap(widths[ci]-2*pad,1000)
                if h+2*pad>heights[ri]+.2:
                    raise ValueError(f"Cell does not fit page {page}, row {ri}, col {ci}: {h+2*pad}>{heights[ri]}")
        t=Table(data,colWidths=widths,rowHeights=heights)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),navy),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,pale]),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),pad),
            ("RIGHTPADDING",(0,0),(-1,-1),pad),("TOPPADDING",(0,0),(-1,-1),pad),
            ("BOTTOMPADDING",(0,0),(-1,-1),pad),("LINEBELOW",(0,0),(-1,0),.5,navy)]))
        _,h=t.wrap(0,0)
        if y-h<38: raise ValueError(f"Table below footer on page {page}: {y-h}")
        t.drawOn(c,x,y-h); return y-h

    def picture(name,x,y,width,height=None):
        path=FIG/name
        with Image.open(path) as im: iw,ih=im.size
        height=height or width*ih/iw
        if y-height<38: raise ValueError(f"Image below footer on page {page}")
        c.drawImage(ImageReader(str(path)),x,y-height,width=width,height=height,mask="auto")
        return y-height

    def note(text,y):
        return para(text,M,y,CW,8.5,leading=11.5,color=muted)

    valid_pairs=[(a,b) for a,b in zip(runs[::2],runs[1::2]) if ev(a)["final_estimate_valid"] and ev(b)["final_estimate_valid"]]
    wins_a=sum(ev(s)["final_angular_error_degrees"]<ev(p)["final_angular_error_degrees"] for p,s in valid_pairs)
    wins_r=sum(ev(s)["final_normalized_regret"]<ev(p)["final_normalized_regret"] for p,s in valid_pairs)
    assert len(valid_pairs)==38 and wins_a==wins_r==36

    y=new("Pedro vs. Score Base", "COMPLETE EXPERIMENTAL RESULTS")
    y=para("Eight noisy scenarios. Equal expert-query budgets.<br/>Every run retained, including failures.",M,y,CW,17,23,color=navy)-19
    y=para("<b>Main finding.</b> Score base has lower mean final angular error and ordinary test regret in all eight scenarios. Among the 38 pairs with two valid estimates, it wins 36 pairs on angular error and 36 on regret. This is an empirical comparison of the complete algorithms, not a proof of convergence or a query-policy-only ablation.",M,y,CW,10.5,15)-20
    cardw=(CW-20)/3
    cards=[("31 / 40","Score base final joint successes"),("4 / 40","Pedro final joint successes"),("80 / 80","Runs completed through T=20")]
    for i,(value,label) in enumerate(cards):
        x=M+i*(cardw+10); c.setFillColor(pale); c.roundRect(x,y-90,cardw,90,6,fill=1,stroke=0)
        para(value,x+12,y-12,cardw-24,25,29,color=teal,bold=True)
        para(label,x+12,y-48,cardw-24,9,12,color=muted)
    y-=111
    y=para("What counts as success?",M,y,CW,13,bold=True)-6
    y=para("Angular error at most 5 degrees <b>and</b> mean held-out normalized regret at most 0.01 (1% of the feasible objective range). These thresholds describe approximate recovery, not exact identification of the true parameter.",M,y,CW,10,14)-18
    y=table(["Study size","Observation model","Evaluation"],[["8 scenarios x 5 seeds x 2 methods","MIN expert; parameter noise; clean Y=X","120 held-out queries per test distribution"],["T=20; 1,600 recorded steps","4D or 6D; no training epochs","All 20 post-observation estimates evaluated"]],[CW/3]*3,M,y,font=9,rowheight=43,headerheight=25)-17
    y=para("<b>Common setup:</b> F(theta,s,x) = theta dot (s*x); 120 fixed unit-norm query candidates; repeats allowed. Pedro uses uniform queries and the cone incenter. Score base uses disagreement queries and the mean of 16 parameter samples.",M,y,CW,9.5,13)-15
    y=para("Reading guide",M,y,CW,13,bold=True)-6
    y=para("<b>Pages 2-8:</b> final statistics, full-time curves, success and recovery times, balanced tests, failures and runtime.<br/><b>Pages 9-16:</b> one results sheet per scenario, including every seed.<br/><b>Pages 17-36:</b> all 1,600 per-step metric records. The companion data bundle preserves full-precision vectors and raw diagnostics.",M,y,CW,10,14)-13
    note("Prepared from saved results only; no experiment was rerun. Source: aqaPayam/InverseOptimization, Git commit "+manifest["source_commit"][:7]+". Report date: "+manifest["generated_date"]+". PDF values are rounded; the data bundle retains full precision. A displayed rounded zero is not necessarily an exact zero.",y)

    y=new("Final performance at T=20")
    y=note("Mean +/- sample standard deviation. All numerical averages are conditional on a valid estimate. Counts are shown explicitly; failures are not assigned arbitrary angular errors or zero regret.",y)-14
    widths=[143,50,50,91,91,90]
    rows=[]
    for i,f in enumerate(FAMILIES):
        for label in (P,S):
            g=group(runs,f,label); es=[ev(r) for r in g]
            rows.append([f"{i+1}. {SHORT[i]}","Pedro" if label==P else "Score",f"{sum(e['final_estimate_valid'] for e in es)}/5",
                         pm([e["final_angular_error_degrees"] for e in es]),
                         pm([e["final_normalized_regret"] for e in es],100,3),f"{sum(final_good(r) for r in g)}/5"])
    y=table(["Scenario","Method","Valid","Angle (deg)","Regret (%)","Joint target"],rows,widths,M,y,font=8.1,rowheight=23,headerheight=28)-17
    y=para("Aggregate outcomes",M,y,CW,13,bold=True)-6
    rows=[]
    for label in (P,S):
        g=[r for r in runs if r["algorithm_name"]==label]
        rows.append(["Pedro" if label==P else "Score base",f"{sum(ev(r)['first_joint_threshold_step'] is not None for r in g)}/40",
                     f"{sum(final_good(r) for r in g)}/40",f"{sum(ev(r)['final_estimate_valid'] and ev(r)['final_normalized_regret']<=.01 for r in g)}/40",
                     f"{sum(not ev(r)['final_estimate_valid'] for r in g)}/40"])
    y=table(["Method","Ever joint\ntarget","Final joint\ntarget","Final regret\n<=1%","Invalid at T"],rows,[103]*5,M,y,font=9,rowheight=24,headerheight=33)-13
    note("Regret is mean excess true objective cost divided by the feasible cost range at each test query, then averaged. It is not a decision-error percentage and not cumulative training-query regret. Lower is better. 0.01 in the raw data equals 1% in this report.",y)

    for title,name,caption in [
        ("Angular error over all 20 steps","angle.png","Thin lines are individual seeds; thick lines are valid-only means. Orange dashed line: 5 degrees. Pedro has 4/5 valid runs from t=9 in case 1 and t=6 in case 2; all other plotted counts are 5/5. Curves need not decrease at every step."),
        ("Held-out regret over all 20 steps","regret.png","Ordinary test distributions. Thin lines: seeds; thick lines: valid-only means. Orange dashed line: 1% target. The symlog vertical scale preserves zero while revealing small late-stage differences. Missing invalid estimates are not imputed."),
        ("How often is the accuracy target met?","success.png","Solid lines: number of runs currently meeting both targets. Dotted lines: number that have ever met both targets by that time. Every count has denominator five; invalid estimates cannot pass. A dotted line staying high does not imply stable recovery."),
    ]:
        y=new(title)
        y=picture(name,M,y,CW)-10
        note(caption,y)

    y=new("First versus sustained recovery")
    y=note("Joint target: angle <=5 degrees and ordinary normalized regret <=0.01. Each list is ordered by seed 0,1,2,3,4. '-' means not reached. All runs continued to T=20; these are retrospective measurements, not actual stopping times.",y)-13
    rows=[]
    for i,f in enumerate(FAMILIES):
        for label in (P,S):
            g=group(runs,f,label)
            first=", ".join(str(ev(r)["first_joint_threshold_step"]) if ev(r)["first_joint_threshold_step"] is not None else "-" for r in g)
            stable=", ".join(str(ev(r)["stable_joint_threshold_step"]) if ev(r)["stable_joint_threshold_step"] is not None else "-" for r in g)
            rows.append([f"{i+1}. {SHORT[i]}","Pedro" if label==P else "Score",first,stable])
    y=table(["Scenario","Method","First hit: seeds 0-4","Sustained: seeds 0-4"],rows,[157,52,153,153],M,y,font=8.8,rowheight=22,headerheight=28)-18
    y=para("What this changes in the interpretation",M,y,CW,13,bold=True)-7
    y=para("Score base reaches the target at least once in <b>37/40</b> runs, but only <b>31/40</b> satisfy it at T. Pedro reaches it at least once in <b>6/40</b>, with <b>4/40</b> passing at T. A single favorable estimate is not the same as stable recovery under noise.",M,y,CW,10,14)-12
    y=para("'Sustained from t' means every remaining observed point through T passes. If t=20, only one point has been observed; this does not establish future stability. Comparing medians only among successful runs would also hide the many non-recoveries, so all seed-level times are shown instead.",M,y,CW,10,14)-12
    note("Example: in case 2, Pedro reaches the target at t=10 for seed 3, before Score base at t=15. Score base is more successful overall, but not uniformly faster on every seed.",y)

    y=new("Balanced tests and selected queries")
    y=note("The two imbalanced families have additional fresh tests giving equal weight to their predefined groups. These tests are hidden from the algorithms and are not used to choose queries or stop.",y)-8
    y=picture("balanced.png",M,y,CW)-12
    rows=[]
    for i in [0,2]:
        row=[f"{i+1}. {SHORT[i]}"]
        for label in (P,S):
            g=group(runs,FAMILIES[i],label)
            row.append(pm([ev(r,"balanced")["final_normalized_regret"] for r in g],100,3))
        row.append("4/5; 5/5" if i==0 else "5/5; 5/5"); rows.append(row)
    y=table(["Scenario","Pedro regret (%)","Score regret (%)","Valid P; S"],rows,[157,131,131,96],M,y,font=8.7,rowheight=25,headerheight=28)-12
    y=para("The final advantage remains on balanced tests. In case 3, Score base selects more diverse queries (34 versus 19 out of 100) and far fewer near profile 1 (2 versus 36). In case 1, connecting-query counts differ more modestly (14 versus 10). The specific query values and the estimator also matter; these counts are not a causal decomposition of the gain.",M,y,CW,10,14)

    y=new("Reliability, noise and computing cost")
    rows=[]
    for i,f in enumerate(FAMILIES):
        p=group(runs,f,P); s=group(runs,f,S)
        rows.append([f"{i+1}. {SHORT[i]}",fmt(statistics.mean(r["runtime_seconds"] for r in p)),
                     fmt(statistics.mean(r["runtime_seconds"] for r in s)),
                     f"{sum(r['metadata']['parameter_noise_decision_flips'] for r in p)} / {sum(r['metadata']['parameter_noise_decision_flips'] for r in s)}"])
    y=table(["Scenario","Pedro seconds","Score seconds","Noise-changed decisions\nPedro / Score (each /100)"],rows,[167,92,92,164],M,y,font=8.7,rowheight=23,headerheight=34)-12
    y=para("Mean recorded runtime per full 20-step run is <b>0.12 s for Pedro</b> and <b>15.17 s for Score base</b>. These timings are the benchmark's run-time measurements, not expert-response costs or a controlled hardware microbenchmark. Score base is better at a fixed query budget here, but computationally slower.",M,y,CW,10,14)-14
    y=para("Independent geometric audit of Pedro's failures",M,y,CW,12,bold=True)-6
    y=table(["Case / seed","First invalid t","Strict interior margin","Nonzero feasible direction"],
            [["1 / seed 3","9","0 at first failure and T","Yes, both times"],["2 / seed 4","6","0 at first failure and T","Yes, both times"]],
            [90,85,172,168],M,y,font=8.5,rowheight=24,headerheight=28)-11
    y=para("Independent linear programs in notebook 15 show zero interior margin but remaining nonzero feasible directions. These are <b>degenerate, lower-dimensional cones</b>; the returned zero incenter is invalid. The stored legacy message 'no valid nonzero parameter direction' is too strong. Neither a 90/180-degree penalty nor a substitute estimate is reported.",M,y,CW,9.5,13)-12
    y=para("Interpretation limits",M,y,CW,12,bold=True)-6
    y=para("Both estimation and query selection differ, so this compares complete algorithms. The sampler uses a finite budget and a loss-based target, not the exact Gaussian-noise likelihood. Five seeds, finite test sets, dimensions 4/6 and this objective family do not establish a general convergence guarantee. Zero test regret does not imply exact parameter recovery.",M,y,CW,9.5,13)-10
    note("Noise-changed decisions compare each observed response with the clean MIN response at the selected query. All Y=X; none of these differences is observation noise. Active queries may land near decision boundaries and thus be more sensitive to parameter noise.",y)

    for i,f in enumerate(FAMILIES):
        y=new(f"{i+1}. {TITLES[i]}","SCENARIO RESULTS")
        y=para(SETTINGS[i],M,y,CW,9.5,13)-7
        y=picture(f"case-{i+1}.png",M,y,CW)-12
        rows=[]
        for seed in range(5):
            for label in (P,S):
                r=next(r for r in group(runs,f,label) if r["seed"]==seed); e=ev(r)
                b=r["metadata"]["evaluations_by_distribution"].get("balanced")
                rows.append([seed,"Pedro" if label==P else "Score","valid" if e["final_estimate_valid"] else "FAIL",
                             fmt(e["final_angular_error_degrees"]),fmt(None if e["final_normalized_regret"] is None else 100*e["final_normalized_regret"],3),
                             fmt(None if not b or b["final_normalized_regret"] is None else 100*b["final_normalized_regret"],3),
                             e["first_joint_threshold_step"] or "-",e["stable_joint_threshold_step"] or "-",
                             fmt(r["runtime_seconds"]),r["metadata"]["parameter_noise_decision_flips"]])
        y=table(["Seed","Method","Status","Angle\n(deg)","Regret\n(%)","Balanced\n(%)","First\njoint","Stable\njoint","Time\n(s)","Noise\nflips"],rows,
                [28,47,44,51,61,62,53,55,57,57],M,y,font=8.2,rowheight=20,headerheight=31)-12
        y=para("Reading this result",M,y,CW,12,bold=True)-6
        y=para(INTERPRETATIONS[i],M,y,CW,9.5,13)-10
        note("Each seed has a different hidden parameter, shared by the paired methods. Thin curves are individual seeds; thick curves are valid-only means. Regret percentages use clean held-out decisions. '-' denotes an unavailable metric or an unreached target, not zero. Noise flips are out of 20 responses.",y)

    assert page==16
    # Four complete runs per landscape page: paired algorithms stay side by side.
    for start in range(0,80,4):
        new(f"Complete step records | runs {start+1}-{start+4} of 80","DATA APPENDIX - ALL 1,600 OBSERVED STEPS",wide=True)
        gap=18; tw=(CW-gap)/2
        for j,r in enumerate(runs[start:start+4]):
            col=j%2; tier=j//2; x=M+col*(tw+gap)
            title_y=H-(83 if tier==0 else 327)
            idx=FAMILIES.index(r["metadata"]["comparison_family"])
            label="Pedro" if r["algorithm_name"]==P else "Score base"
            para(f"{idx+1}. {SHORT[idx]} | seed {r['seed']} | {label}",x,title_y,tw,9,11,color=navy,bold=True)
            e=ev(r); b=r["metadata"]["evaluations_by_distribution"].get("balanced")
            rows=[]
            for t in range(20):
                valid=e["valid_estimate_history"][t]
                rows.append([t+1,fmt(e["angular_error_history_degrees"][t],3),
                             fmt(None if e["normalized_regret_history"][t] is None else 100*e["normalized_regret_history"][t],4),
                             fmt(None if not b or b["normalized_regret_history"][t] is None else 100*b["normalized_regret_history"][t],4),
                             "valid" if valid else "FAIL"])
            table(["t","Angle (deg)","Regret (%)","Balanced (%)","Status"],rows,
                  [tw*.08,tw*.23,tw*.25,tw*.26,tw*.18],x,title_y-13,font=7.3,rowheight=10,headerheight=14,pad=.5)
    assert page==total
    footer(); c.save()
    print(f"Created {PDF.name}: {page} pages; all 80 run summaries and all 1,600 per-step metric rows.")


def finish_bundle(files,manifest):
    from pypdf import PdfReader
    reader=PdfReader(PDF)
    assert len(reader.pages)==36
    text="\n".join(p.extract_text() or "" for p in reader.pages)
    assert "31 / 40" in text and "4 / 40" in text
    assert text.count("Complete step records | runs")==20
    # Capture a readable inventory and hashes for the final artifacts.
    readme="""PEDRO VS SCORE BASE - COMPLETE SAVED RESULTS

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

Source commit: """+manifest["source_commit"]+"\nRepository: https://github.com/aqaPayam/InverseOptimization\n"
    (OUT/"README_results.txt").write_text(readme,encoding="utf-8")
    bundle=OUT/"Pedro_vs_Score_Base_Complete_Results_Bundle.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in [PDF.name,"per_run_results.csv","per_step_results.csv","source_manifest.json","README_results.txt"]:
            z.write(OUT/name,name)
        for p in files: z.write(p,"raw_runs/"+p.name)
        z.write(ROOT/"docs/pedro_score_comparison.md","comparison_protocol.md")
    print(f"Bundle ready: {bundle.name}; {PDF.stat().st_size/1e6:.2f} MB PDF; {bundle.stat().st_size/1e6:.2f} MB ZIP.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--figures-only",action="store_true")
    args=parser.parse_args(); runs,files=load_runs()
    if args.figures_only:
        make_figures(runs)
    else:
        manifest=export_data(runs,files); make_pdf(runs,manifest); finish_bundle(files,manifest)
