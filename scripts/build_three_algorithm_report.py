"""Build the complete three-algorithm PDF from saved benchmark trajectories."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "active" / "16_three_algorithm_comparison"
OUTPUT = ROOT / "output" / "pdf" / "Pedro_Genious_Pedro_Score_Base_Complete_Results.pdf"
TMP = ROOT / "tmp" / "pdfs" / "three_algorithm_report"
ALGORITHMS = ("Pedro algorithm", "Genious Pedro", "Score base model")
SHORT = {"Pedro algorithm": "Pedro", "Genious Pedro": "Genious", "Score base model": "Score Base"}
COLORS = {"Pedro algorithm": "#6b7280", "Genious Pedro": "#cc3311", "Score base model": "#0077bb"}
FAMILY_ORDER = (
    "bridge-6d", "boundaries-4d", "redundancy-6d", "balanced-choice-4d",
    "balanced-subset-6d", "knapsack-6d", "strong-noise-4d", "query-noise-4d",
)


def evaluation(run, distribution="ordinary"):
    return run["metadata"]["evaluations_by_distribution"][distribution]


def valid_mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def mean_history(runs, key):
    array = np.asarray([
        [np.nan if value is None else value for value in evaluation(run)[key]]
        for run in runs
    ], dtype=float)
    count = np.isfinite(array).sum(axis=0)
    return np.divide(np.nansum(array, axis=0), count,
                     out=np.full(array.shape[1], np.nan), where=count > 0)


def vector(value):
    return "[" + ",".join(f"{float(item):.3f}" for item in value) + "]"


def fmt(value, digits=3):
    return "-" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def load_runs():
    files = sorted(SOURCE.glob("*.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    assert len(runs) == 120, f"expected 120 runs, found {len(runs)}"
    assert {run["algorithm_name"] for run in runs} == set(ALGORITHMS)
    assert all(len(run["records"]) == 20 for run in runs)
    assert len({(run["metadata"]["comparison_family"], run["seed"], run["algorithm_name"])
                for run in runs}) == 120
    return sorted(runs, key=lambda run: (
        FAMILY_ORDER.index(run["metadata"]["comparison_family"]),
        run["seed"], ALGORITHMS.index(run["algorithm_name"]),
    ))


def make_charts(runs):
    TMP.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["metadata"]["comparison_family"], run["algorithm_name"])].append(run)
    titles = {run["metadata"]["comparison_family"]: run["metadata"]["comparison_title"]
              for run in runs}

    for key, filename, ylabel, threshold in (
        ("angular_error_history_degrees", "angle_histories.png", "Angular error (deg)", 5),
        ("normalized_regret_history", "regret_histories.png", "Normalized regret", .01),
    ):
        fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True, layout="constrained")
        for family, axis in zip(FAMILY_ORDER, axes.flat):
            for algorithm in ALGORITHMS:
                subset = grouped[(family, algorithm)]
                for run in subset:
                    values = np.asarray([
                        np.nan if value is None else value for value in evaluation(run)[key]
                    ])
                    axis.plot(range(1, 21), values, color=COLORS[algorithm], alpha=.16, lw=.8)
                axis.plot(range(1, 21), mean_history(subset, key), color=COLORS[algorithm],
                          lw=2, label=SHORT[algorithm])
            axis.axhline(threshold, color="#ee7733", ls="--", lw=1)
            axis.set_title(titles[family], fontsize=9)
            axis.set_ylabel(ylabel, fontsize=8)
            axis.set_xticks([1, 5, 10, 15, 20])
            axis.grid(alpha=.2)
        axes[-1, 0].set_xlabel("Observed query count t")
        axes[-1, 1].set_xlabel("Observed query count t")
        axes[0, 0].legend(ncol=3, fontsize=7)
        fig.savefig(TMP / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True, sharey=True,
                             layout="constrained")
    for family, axis in zip(FAMILY_ORDER, axes.flat):
        for algorithm in ALGORITHMS:
            subset = grouped[(family, algorithm)]
            counts = np.sum([evaluation(run)["valid_estimate_history"] for run in subset], axis=0)
            axis.step(range(1, 21), counts, where="mid", color=COLORS[algorithm],
                      lw=2, label=SHORT[algorithm])
        axis.set_title(titles[family], fontsize=9)
        axis.set_ylim(-.2, 5.2)
        axis.set_yticks(range(6))
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.grid(alpha=.2)
    axes[-1, 0].set_xlabel("Observed query count t")
    axes[-1, 1].set_xlabel("Observed query count t")
    axes[0, 0].legend(ncol=3, fontsize=7)
    fig.savefig(TMP / "validity_histories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def table(data, widths=None, font=7, header=True, align="CENTER"):
    result = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("LEADING", (0, 0), (-1, -1), font + 1.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), align),
        ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1),
         [colors.white, colors.HexColor("#f8fafc")]),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    result.setStyle(TableStyle(commands))
    return result


def build_pdf(runs):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title2", parent=styles["Title"], fontName="Helvetica-Bold",
                           fontSize=25, leading=29, textColor=colors.HexColor("#17324d"),
                           alignment=TA_CENTER, spaceAfter=12)
    h1 = ParagraphStyle("H1x", parent=styles["Heading1"], fontName="Helvetica-Bold",
                        fontSize=16, leading=19, textColor=colors.HexColor("#17324d"), spaceAfter=8)
    h2 = ParagraphStyle("H2x", parent=styles["Heading2"], fontName="Helvetica-Bold",
                        fontSize=11, leading=14, textColor=colors.HexColor("#0077bb"), spaceAfter=5)
    body = ParagraphStyle("Bodyx", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=9, leading=13, alignment=TA_LEFT, spaceAfter=7)
    small = ParagraphStyle("Smallx", parent=body, fontSize=7, leading=9, spaceAfter=3)

    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=14 * mm, bottomMargin=13 * mm,
        title="Pedro, Genious Pedro, and Score Base - Complete Experimental Results",
        author="invoptlab benchmark",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d1d5db"))
        canvas.line(14 * mm, 10 * mm, landscape(A4)[0] - 14 * mm, 10 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(14 * mm, 6 * mm, "Exact three-algorithm benchmark | 8 scenarios | 5 seeds | T=20")
        canvas.drawRightString(landscape(A4)[0] - 14 * mm, 6 * mm, f"Page {doc.page}")
        canvas.restoreState()

    overall = {}
    for algorithm in ALGORITHMS:
        group = [run for run in runs if run["algorithm_name"] == algorithm]
        evaluations = [evaluation(run) for run in group]
        valid = [item for item in evaluations if item["final_estimate_valid"]]
        overall[algorithm] = {
            "valid": len(valid),
            "invalid": 40 - len(valid),
            "ever_invalid": sum(run["metadata"]["first_invalid_step"] is not None for run in group),
            "joint": sum(item["final_estimate_valid"] and
                         item["final_angular_error_degrees"] <= 5 and
                         item["final_normalized_regret"] <= .01 for item in evaluations),
            "angle": valid_mean([item["final_angular_error_degrees"] for item in valid]),
            "regret": valid_mean([item["final_normalized_regret"] for item in valid]),
            "runtime": float(np.mean([run["runtime_seconds"] for run in group])),
        }

    story = [Spacer(1, 22 * mm), Paragraph("Pedro, Genious Pedro, and Score Base", title),
             Paragraph("Complete experimental results - exact extension of the previous benchmark", h2),
             Spacer(1, 8 * mm)]
    cards = [["Algorithm", "Valid final", "Final joint success", "Mean angle (valid)",
              "Mean regret (valid)", "Mean runtime"]]
    for algorithm in ALGORITHMS:
        value = overall[algorithm]
        cards.append([SHORT[algorithm], f"{value['valid']}/40", f"{value['joint']}/40",
                      f"{value['angle']:.2f} deg", f"{value['regret']:.5f}",
                      f"{value['runtime']:.2f} s"])
    story.extend([table(cards, [110, 90, 105, 110, 110, 90], font=9), Spacer(1, 8 * mm),
        Paragraph("Main finding", h1),
        Paragraph(
            "Score Base is the most reliable method in this noisy benchmark: all 40 final estimates "
            "are valid and 31 satisfy both the 5 degree and 0.01-regret targets. Genious Pedro shows "
            "strong accuracy in several valid runs, especially subset and knapsack cases, but only "
            "15 of 40 final hard-cone estimates remain valid. Its boundary-seeking query rule therefore "
            "trades information for substantial sensitivity to contradictory noisy demonstrations. "
            "Valid-only means must not be interpreted without these failure counts.", body),
        Paragraph(
            "All 120 runs reached T=20. Invalid estimates retain missing angle/regret values; no 90/180 "
            "degree penalty, zero-regret replacement, softened cone, favorable seed filtering, or early "
            "stopping is used.", body), PageBreak()])

    story.extend([Paragraph("1. Protocol and algorithm definitions", h1),
        Paragraph("The scenario protocol is byte-for-byte shared with the previous comparison: eight "
                  "4D/6D MIN families, five paired seeds, 120 fixed candidate queries, T=20, clean Y=X, "
                  "and parameter noise. Each estimate is evaluated after every observation on 120 fresh "
                  "hidden clean queries per disclosed test distribution.", body),
        table([
            ["Algorithm", "Next S", "theta estimate", "Additional information"],
            ["Pedro", "Uniform candidate", "Hard-cone incenter", "None"],
            ["Genious Pedro", "Minimum normalized predicted decision margin", "Same hard-cone incenter", "None"],
            ["Score Base", "Maximum optimizer disagreement", "Mean of 16 parameter samples", "Sample ensemble"],
        ], [100, 210, 180, 135], font=8), Spacer(1, 5 * mm),
        Paragraph("Objective: F(theta,s,x) = theta dot (s*x). The expert minimizes this objective. "
                  "Standardized Gaussian parameter perturbations are paired by seed and round. Because "
                  "the three algorithms may choose different S, their realized expert decisions may differ.", body),
        Paragraph("Metrics", h2),
        Paragraph("Angular error measures parameter direction. Normalized regret measures clean held-out "
                  "decision quality. The joint target requires angle <= 5 degrees and regret <= 0.01. "
                  "A valid-only mean is conditional and is always accompanied by n/5.", body),
        Paragraph("Genious Pedro selects uniformly at D0. Thereafter it chooses the candidate whose predicted "
                  "optimal decision has the smallest normalized objective gap to an alternative. If its "
                  "incenter becomes invalid, it records failure and uses the agreed uniform fallback while "
                  "continuing to T.", body), PageBreak()])

    story.append(Paragraph("2. Final results by scenario", h1))
    rows = [["Scenario", "Algorithm", "Valid", "Ever invalid", "Angle mean", "Regret mean", "Joint"]]
    for family in FAMILY_ORDER:
        title_text = next(run["metadata"]["comparison_title"] for run in runs
                          if run["metadata"]["comparison_family"] == family)
        for algorithm in ALGORITHMS:
            group = [run for run in runs if run["metadata"]["comparison_family"] == family
                     and run["algorithm_name"] == algorithm]
            evaluations = [evaluation(run) for run in group]
            valid = [item for item in evaluations if item["final_estimate_valid"]]
            rows.append([title_text, SHORT[algorithm], f"{len(valid)}/5",
                         sum(run["metadata"]["first_invalid_step"] is not None for run in group),
                         fmt(valid_mean([item["final_angular_error_degrees"] for item in valid])),
                         fmt(valid_mean([item["final_normalized_regret"] for item in valid]), 5),
                         sum(item["final_estimate_valid"] and item["final_angular_error_degrees"] <= 5
                             and item["final_normalized_regret"] <= .01 for item in evaluations)])
    story.extend([table(rows, [170, 80, 48, 62, 72, 75, 42], font=7), Spacer(1, 4 * mm),
                  Paragraph("A missing mean means no seed had a valid final estimate. Genious Pedro has no "
                            "valid final estimate in the boundary-localization or strong-noise families.", small),
                  PageBreak()])

    story.append(Paragraph("2.1 Balanced hidden-test distributions", h1))
    story.append(Paragraph(
        "The connecting-groups and redundant-query families also use a separate hidden test set "
        "that weights their predefined candidate groups equally. These tests are never supplied to "
        "an algorithm and are not mixed with ordinary regret.", body))
    balanced_rows = [["Scenario", "Algorithm", "Valid", "Angle mean", "Regret mean", "Joint"]]
    for family in ("bridge-6d", "redundancy-6d"):
        title_text = next(run["metadata"]["comparison_title"] for run in runs
                          if run["metadata"]["comparison_family"] == family)
        for algorithm in ALGORITHMS:
            group = [run for run in runs if run["metadata"]["comparison_family"] == family
                     and run["algorithm_name"] == algorithm]
            items = [evaluation(run, "balanced") for run in group]
            valid = [item for item in items if item["final_estimate_valid"]]
            balanced_rows.append([
                title_text, SHORT[algorithm], f"{len(valid)}/5",
                fmt(valid_mean([item["final_angular_error_degrees"] for item in valid])),
                fmt(valid_mean([item["final_normalized_regret"] for item in valid]), 5),
                sum(item["final_estimate_valid"] and item["final_angular_error_degrees"] <= 5
                    and item["final_normalized_regret"] <= .01 for item in items),
            ])
    story.extend([table(balanced_rows, [190, 90, 55, 85, 85, 55], font=8), Spacer(1, 6 * mm),
                  Paragraph("Selected candidate-group totals", h2)])
    coverage_rows = [["Scenario", "Algorithm", "Group counts across 100 selected queries"]]
    for family in ("bridge-6d", "redundancy-6d"):
        title_text = next(run["metadata"]["comparison_title"] for run in runs
                          if run["metadata"]["comparison_family"] == family)
        for algorithm in ALGORITHMS:
            group = [run for run in runs if run["metadata"]["comparison_family"] == family
                     and run["algorithm_name"] == algorithm]
            labels = group[0]["metadata"]["candidate_group_labels"]
            counts = np.zeros(len(labels), dtype=int)
            for run in group:
                counts += np.bincount(run["metadata"]["selected_candidate_groups"],
                                      minlength=len(labels))
            coverage_rows.append([title_text, SHORT[algorithm],
                                  "; ".join(f"{label}: {count}" for label, count in zip(labels, counts))])
    story.extend([table(coverage_rows, [170, 80, 390], font=7, align="LEFT"), PageBreak()])

    story.append(Paragraph("2.2 Recovery and paired comparisons", h1))
    recovery_rows = [["Algorithm", "Ever joint", "Stable joint", "Final joint",
                      "Final valid", "Ever invalid"]]
    for algorithm in ALGORITHMS:
        group = [run for run in runs if run["algorithm_name"] == algorithm]
        items = [evaluation(run) for run in group]
        recovery_rows.append([
            SHORT[algorithm],
            f"{sum(item['first_joint_threshold_step'] is not None for item in items)}/40",
            f"{sum(item['stable_joint_threshold_step'] is not None for item in items)}/40",
            f"{overall[algorithm]['joint']}/40", f"{overall[algorithm]['valid']}/40",
            f"{overall[algorithm]['ever_invalid']}/40",
        ])
    story.extend([table(recovery_rows, [105, 90, 90, 90, 90, 90], font=8), Spacer(1, 7 * mm),
                  Paragraph("Paired ordinary-test outcomes at T=20", h2)])
    pair_rows = [["Left", "Right", "Both valid", "Left invalid only", "Right invalid only",
                  "Both invalid", "Left angle wins", "Right angle wins",
                  "Left regret wins", "Right regret wins"]]
    for left, right in combinations(ALGORITHMS, 2):
        counts = defaultdict(int)
        for family in FAMILY_ORDER:
            for seed in range(5):
                a = evaluation(next(run for run in runs if run["metadata"]["comparison_family"] == family
                                    and run["seed"] == seed and run["algorithm_name"] == left))
                b = evaluation(next(run for run in runs if run["metadata"]["comparison_family"] == family
                                    and run["seed"] == seed and run["algorithm_name"] == right))
                av, bv = a["final_estimate_valid"], b["final_estimate_valid"]
                if av and bv:
                    counts["both"] += 1
                    if a["final_angular_error_degrees"] < b["final_angular_error_degrees"] - 1e-12:
                        counts["left_angle"] += 1
                    elif b["final_angular_error_degrees"] < a["final_angular_error_degrees"] - 1e-12:
                        counts["right_angle"] += 1
                    if a["final_normalized_regret"] < b["final_normalized_regret"] - 1e-12:
                        counts["left_regret"] += 1
                    elif b["final_normalized_regret"] < a["final_normalized_regret"] - 1e-12:
                        counts["right_regret"] += 1
                elif not av and bv:
                    counts["left_invalid"] += 1
                elif av and not bv:
                    counts["right_invalid"] += 1
                else:
                    counts["both_invalid"] += 1
        pair_rows.append([
            SHORT[left], SHORT[right], counts["both"], counts["left_invalid"],
            counts["right_invalid"], counts["both_invalid"], counts["left_angle"],
            counts["right_angle"], counts["left_regret"], counts["right_regret"],
        ])
    story.extend([table(pair_rows, [63, 63, 48, 62, 62, 48, 58, 58, 58, 58], font=6.5),
                  Spacer(1, 5 * mm),
                  Paragraph("Wins are counted only when both estimates are valid. Ties are omitted from the "
                            "win columns, so win counts need not sum to the both-valid count.", small),
                  PageBreak()])

    for heading, filename, note in (
        ("3. Angular-error trajectories", "angle_histories.png",
         "Thin curves are individual seeds; thick curves are valid-only means. The dashed line is 5 degrees."),
        ("4. Held-out-regret trajectories", "regret_histories.png",
         "Thin curves are individual seeds; thick curves are valid-only means. The dashed line is 0.01."),
        ("5. Valid-estimate trajectories", "validity_histories.png",
         "Each curve reports how many of the five estimates remain valid at each time."),
    ):
        story.extend([Paragraph(heading, h1), Paragraph(note, body),
                      Image(str(TMP / filename), width=220 * mm, height=154 * mm), PageBreak()])

    story.append(Paragraph("6. Every-step conditional statistics", h1))
    story.append(Paragraph("The following eight pages reproduce the previous notebook's time-resolved "
                           "summary format for all three algorithms.", body))
    story.append(PageBreak())
    for family in FAMILY_ORDER:
        family_runs = [run for run in runs if run["metadata"]["comparison_family"] == family]
        title_text = family_runs[0]["metadata"]["comparison_title"]
        story.append(Paragraph(title_text, h1))
        rows = [["t"]]
        for algorithm in ALGORITHMS:
            rows[0].extend([SHORT[algorithm] + " n", "angle", "regret"])
        for t in range(20):
            row = [t + 1]
            for algorithm in ALGORITHMS:
                group = [run for run in family_runs if run["algorithm_name"] == algorithm]
                angles = [evaluation(run)["angular_error_history_degrees"][t] for run in group]
                regrets = [evaluation(run)["normalized_regret_history"][t] for run in group]
                count = sum(value is not None for value in angles)
                row.extend([f"{count}/5", fmt(valid_mean(angles)), fmt(valid_mean(regrets), 5)])
            rows.append(row)
        story.extend([table(rows, [25] + [48, 55, 60] * 3, font=7),
                      Paragraph("Means are conditional on a valid estimate at that time.", small), PageBreak()])

    story.append(Paragraph("7. Complete run summaries", h1))
    summary_rows = [["Scenario", "Seed", "Algorithm", "Status", "Angle", "Regret",
                     "First joint", "Stable joint", "First invalid", "Seconds"]]
    for run in runs:
        item = evaluation(run)
        summary_rows.append([
            run["metadata"]["comparison_family"], run["seed"], SHORT[run["algorithm_name"]],
            item["final_status"], fmt(item["final_angular_error_degrees"]),
            fmt(item["final_normalized_regret"], 5), item["first_joint_threshold_step"] or "-",
            item["stable_joint_threshold_step"] or "-", run["metadata"]["first_invalid_step"] or "-",
            f"{run['runtime_seconds']:.2f}",
        ])
    for start in range(0, len(summary_rows) - 1, 26):
        story.extend([table([summary_rows[0]] + summary_rows[1 + start:1 + start + 26],
                            [105, 28, 62, 72, 48, 55, 48, 48, 48, 45], font=6.5), PageBreak()])

    story.append(Paragraph("8. Complete 2,400-step appendix", h1))
    story.append(Paragraph("Every observed step is included below. Vector values are rounded to three decimals "
                           "for readability; the JSON checkpoints preserve full precision.", body))
    story.append(PageBreak())
    for run in runs:
        item = evaluation(run)
        story.append(Paragraph(
            f"{run['metadata']['comparison_title']} | seed {run['seed']} | {SHORT[run['algorithm_name']]}",
            h2,
        ))
        rows = [["t", "status", "angle", "regret", "theta_hat", "S", "observed Y"]]
        for index, record in enumerate(run["records"]):
            rows.append([
                index + 1, item["estimate_status_history"][index],
                fmt(item["angular_error_history_degrees"][index], 2),
                fmt(item["normalized_regret_history"][index], 4),
                vector(record["theta_hat_after"]), vector(record["query"]),
                vector(record["observed_decision"]),
            ])
        story.append(table(rows, [18, 65, 38, 42, 145, 145, 145], font=5.1))
        story.append(PageBreak())

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


def main():
    runs = load_runs()
    make_charts(runs)
    build_pdf(runs)


if __name__ == "__main__":
    main()
