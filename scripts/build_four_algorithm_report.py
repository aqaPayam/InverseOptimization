"""Build the complete four-algorithm PDF from saved benchmark trajectories."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

import build_three_algorithm_report as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "active" / "17_four_algorithm_comparison"
OUTPUT = ROOT / "output" / "pdf" / "Four_Algorithm_Active_Inverse_Optimization_Complete_Results.pdf"
TMP = ROOT / "tmp" / "pdfs" / "four_algorithm_report"
ALGORITHMS = (
    "Pedro algorithm",
    "Genious Pedro",
    "Score base model",
    "Uniform Online SAMD",
)
SHORT = {
    "Pedro algorithm": "Pedro",
    "Genious Pedro": "Genious",
    "Score base model": "Score Base",
    "Uniform Online SAMD": "Online SAMD",
}
COLORS = {
    "Pedro algorithm": "#6b7280",
    "Genious Pedro": "#cc3311",
    "Score base model": "#0077bb",
    "Uniform Online SAMD": "#009988",
}
FAMILY_ORDER = (
    "bridge-6d",
    "boundaries-4d",
    "redundancy-6d",
    "balanced-choice-4d",
    "balanced-subset-6d",
    "knapsack-6d",
    "strong-noise-4d",
    "query-noise-4d",
)


def evaluation(run, distribution="ordinary"):
    return run["metadata"]["evaluations_by_distribution"][distribution]


def valid_mean(values):
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def fmt(value, digits=3):
    return "-" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def vector(value):
    return "[" + ",".join(f"{float(item):.3f}" for item in value) + "]"


def load_runs():
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(SOURCE.glob("*.json"))]
    assert len(runs) == 160, f"expected 160 runs, found {len(runs)}"
    assert {run["algorithm_name"] for run in runs} == set(ALGORITHMS)
    assert all(run["error"] is None and len(run["records"]) == 20 for run in runs)
    assert len({
        (run["metadata"]["comparison_family"], run["seed"], run["algorithm_name"])
        for run in runs
    }) == 160
    return sorted(runs, key=lambda run: (
        FAMILY_ORDER.index(run["metadata"]["comparison_family"]),
        run["seed"],
        ALGORITHMS.index(run["algorithm_name"]),
    ))


def configure_base_helpers():
    base.SOURCE = SOURCE
    base.OUTPUT = OUTPUT
    base.TMP = TMP
    base.ALGORITHMS = ALGORITHMS
    base.SHORT = SHORT
    base.COLORS = COLORS
    base.FAMILY_ORDER = FAMILY_ORDER


def build_pdf(runs):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleFour",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#17324d"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "H1Four",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        textColor=colors.HexColor("#17324d"),
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2Four",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0077bb"),
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "BodyFour",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        alignment=TA_LEFT,
        spaceAfter=7,
    )
    small = ParagraphStyle("SmallFour", parent=body, fontSize=7, leading=9, spaceAfter=3)

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=13 * mm,
        title="Four-Algorithm Active Inverse Optimization - Complete Experimental Results",
        author="invoptlab benchmark",
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d1d5db"))
        canvas.line(14 * mm, 10 * mm, landscape(A4)[0] - 14 * mm, 10 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(14 * mm, 6 * mm, "Four-algorithm benchmark | 8 scenarios | 5 seeds | T=20")
        canvas.drawRightString(landscape(A4)[0] - 14 * mm, 6 * mm, f"Page {doc.page}")
        canvas.restoreState()

    overall = {}
    for algorithm in ALGORITHMS:
        group = [run for run in runs if run["algorithm_name"] == algorithm]
        items = [evaluation(run) for run in group]
        valid = [item for item in items if item["final_estimate_valid"]]
        stable = [
            item["stable_joint_threshold_step"]
            for item in items
            if item["stable_joint_threshold_step"] is not None
        ]
        overall[algorithm] = {
            "valid": len(valid),
            "ever_invalid": sum(run["metadata"]["first_invalid_step"] is not None for run in group),
            "angle": valid_mean([item["final_angular_error_degrees"] for item in valid]),
            "regret": valid_mean([item["final_normalized_regret"] for item in valid]),
            "joint": len(stable),
            "joint_step": valid_mean(stable),
            "runtime": float(np.mean([run["runtime_seconds"] for run in group])),
        }

    story = [
        Spacer(1, 18 * mm),
        Paragraph("Four-Algorithm Active Inverse Optimization", title),
        Paragraph("Complete experimental results with convergence figures and all trajectories", h2),
        Spacer(1, 7 * mm),
    ]
    cards = [[
        "Algorithm", "Valid at T", "Stable joint success", "Mean angle (valid)",
        "Mean regret (valid)", "Mean runtime",
    ]]
    for algorithm in ALGORITHMS:
        item = overall[algorithm]
        cards.append([
            SHORT[algorithm],
            f"{item['valid']}/40",
            f"{item['joint']}/40",
            f"{item['angle']:.2f} deg",
            f"{item['regret']:.5f}",
            f"{item['runtime']:.3f} s",
        ])
    story.extend([
        base.table(cards, [105, 80, 105, 110, 110, 90], font=8.5),
        Spacer(1, 7 * mm),
        Paragraph("Main finding", h1),
        Paragraph(
            "Score Base is the clear winner at the fixed query horizon. It keeps all 40 estimates "
            "valid, reaches mean final angular error 3.83 degrees and mean regret 0.0010, and "
            "achieves the sustained joint target in 31 of 40 runs. Pedro is reliable but slower. "
            "Genious Pedro can be accurate when its cone survives, but parameter noise collapses "
            "25 of its 40 hard cones. Online SAMD stays structurally valid in all runs but is not "
            "query-efficient at T=20 under the frozen one-update-per-observation configuration.",
            body,
        ),
        Paragraph(
            "All 160 runs continue to T=20. Invalid estimates remain missing rather than being "
            "replaced by 90/180 degree errors or zero regret. Every conditional mean is accompanied "
            "by its valid count.",
            body,
        ),
        PageBreak(),
    ])

    story.extend([
        Paragraph("1. Protocol and algorithm definitions", h1),
        Paragraph(
            "The comparison uses the unchanged eight-scenario protocol: 4D and 6D MIN problems, "
            "five paired seeds, 120 fixed candidate queries, 120 fresh hidden test queries, T=20, "
            "parameter noise, and no observation noise (Y=X). The objective is "
            "F(theta,s,x) = theta dot (s*x). Algorithms never see theta-star or hidden evaluation data.",
            body,
        ),
        base.table([
            ["Algorithm", "Next S", "Estimate after observing Y", "Key property"],
            ["Pedro", "Uniform candidate", "Hard-cone incenter", "Exact constraints"],
            ["Genious Pedro", "Minimum predicted boundary margin", "Hard-cone incenter", "Active boundary seeking"],
            ["Score Base", "Maximum sample disagreement", "Mean of 16 samples", "Noise-tolerant distribution"],
            ["Online SAMD", "Uniform candidate", "One signed mirror update", "Exact finite ASL oracle"],
        ], [90, 190, 180, 155], font=8),
        Spacer(1, 5 * mm),
        Paragraph("Fairness rules", h2),
        Paragraph(
            "All algorithms receive one expert response per round and share the same scenario seeds, "
            "candidate pools, test queries, horizon, noise settings, and metrics. The SAMD learning "
            "rate and L1 radius are fixed before the benchmark and are not tuned per scenario. Wall "
            "time is reported separately from query efficiency.",
            body,
        ),
        Paragraph("Interpretation boundary", h2),
        Paragraph(
            "The fourth method is an online finite-oracle specialization using one SAMD update per "
            "new observation. It is not a batch rerun with many internal optimization iterations. "
            "Therefore poor T=20 query efficiency does not contradict the paper's convergence theory.",
            body,
        ),
        PageBreak(),
    ])

    story.append(Paragraph("2. Final results by scenario", h1))
    rows = [["Scenario", "Algorithm", "Valid", "Ever invalid", "Angle", "Regret", "Stable joint"]]
    for family in FAMILY_ORDER:
        title_text = next(
            run["metadata"]["comparison_title"]
            for run in runs
            if run["metadata"]["comparison_family"] == family
        )
        for algorithm in ALGORITHMS:
            group = [
                run for run in runs
                if run["metadata"]["comparison_family"] == family
                and run["algorithm_name"] == algorithm
            ]
            items = [evaluation(run) for run in group]
            valid = [item for item in items if item["final_estimate_valid"]]
            rows.append([
                title_text,
                SHORT[algorithm],
                f"{len(valid)}/5",
                sum(run["metadata"]["first_invalid_step"] is not None for run in group),
                fmt(valid_mean([item["final_angular_error_degrees"] for item in valid])),
                fmt(valid_mean([item["final_normalized_regret"] for item in valid]), 5),
                sum(item["stable_joint_threshold_step"] is not None for item in items),
            ])
    story.extend([
        base.table(rows, [165, 80, 45, 60, 65, 70, 62], font=6.5),
        PageBreak(),
    ])

    story.extend([
        Paragraph("2.1 Recovery and pairwise outcomes", h1),
        base.table([
            ["Algorithm", "Final valid", "Ever invalid", "Stable joint", "Mean stable step", "Mean runtime"],
            *[
                [
                    SHORT[algorithm],
                    f"{overall[algorithm]['valid']}/40",
                    f"{overall[algorithm]['ever_invalid']}/40",
                    f"{overall[algorithm]['joint']}/40",
                    fmt(overall[algorithm]["joint_step"], 1),
                    f"{overall[algorithm]['runtime']:.3f} s",
                ]
                for algorithm in ALGORITHMS
            ],
        ], [100, 85, 85, 85, 95, 90], font=8),
        Spacer(1, 7 * mm),
        Paragraph("Paired ordinary-test outcomes at T=20", h2),
    ])
    pair_rows = [[
        "Left", "Right", "Both valid", "Left invalid", "Right invalid", "Both invalid",
        "Left angle wins", "Right angle wins", "Left regret wins", "Right regret wins",
    ]]
    for left, right in combinations(ALGORITHMS, 2):
        counts = defaultdict(int)
        for family in FAMILY_ORDER:
            for seed in range(5):
                a = evaluation(next(
                    run for run in runs
                    if run["metadata"]["comparison_family"] == family
                    and run["seed"] == seed and run["algorithm_name"] == left
                ))
                b = evaluation(next(
                    run for run in runs
                    if run["metadata"]["comparison_family"] == family
                    and run["seed"] == seed and run["algorithm_name"] == right
                ))
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
    story.extend([
        base.table(pair_rows, [62, 62, 48, 52, 52, 48, 58, 58, 58, 58], font=6.2),
        Spacer(1, 5 * mm),
        Paragraph(
            "Wins are counted only when both estimates are valid. Ties are omitted from win columns.",
            small,
        ),
        PageBreak(),
    ])

    trajectory_rows = [["Algorithm", "t=1", "t=5", "t=10", "t=15", "t=20"]]
    for algorithm in ALGORITHMS:
        group = [run for run in runs if run["algorithm_name"] == algorithm]
        row = [SHORT[algorithm]]
        for step in (1, 5, 10, 15, 20):
            index = step - 1
            angles = [evaluation(run)["angular_error_history_degrees"][index] for run in group]
            regrets = [evaluation(run)["normalized_regret_history"][index] for run in group]
            row.append(f"{fmt(valid_mean(angles), 2)} deg / {fmt(valid_mean(regrets), 4)}")
        trajectory_rows.append(row)
    story.extend([
        Paragraph("2.2 Convergence checkpoints", h1),
        Paragraph(
            "Each cell reports the valid-only mean angular error and normalized regret across all "
            "eight scenarios and five seeds.",
            body,
        ),
        base.table(trajectory_rows, [100, 100, 100, 100, 100, 100], font=7.5),
        Spacer(1, 7 * mm),
        Paragraph(
            "Score Base improves steadily from 53.61 degrees at t=1 to 3.83 degrees at t=20. "
            "Online SAMD improves early but plateaus near 41-42 degrees. Genious Pedro's late "
            "valid-only mean is affected by attrition: only 15 of 40 estimates remain valid at t=20.",
            body,
        ),
        PageBreak(),
    ])

    for heading, filename, note in (
        (
            "3. Angular-error trajectories",
            "angle_histories.png",
            "Thin curves are individual seeds; thick curves are valid-only means. The dashed line is 5 degrees.",
        ),
        (
            "4. Held-out-regret trajectories",
            "regret_histories.png",
            "Thin curves are individual seeds; thick curves are valid-only means. The dashed line is 0.01.",
        ),
        (
            "5. Valid-estimate trajectories",
            "validity_histories.png",
            "Each curve reports how many of the five estimates remain valid at every time step.",
        ),
    ):
        story.extend([
            Paragraph(heading, h1),
            Paragraph(note, body),
            Image(str(TMP / filename), width=220 * mm, height=154 * mm),
            PageBreak(),
        ])

    story.extend([
        Paragraph("6. Every-step conditional statistics", h1),
        Paragraph(
            "The next eight pages report all 20 time steps for every algorithm and scenario. "
            "Each n value is the number of valid estimates among the five paired seeds.",
            body,
        ),
        PageBreak(),
    ])
    for family in FAMILY_ORDER:
        family_runs = [run for run in runs if run["metadata"]["comparison_family"] == family]
        story.append(Paragraph(family_runs[0]["metadata"]["comparison_title"], h1))
        rows = [["t"]]
        for algorithm in ALGORITHMS:
            rows[0].extend([SHORT[algorithm] + " n", "angle", "regret"])
        for index in range(20):
            row = [index + 1]
            for algorithm in ALGORITHMS:
                group = [run for run in family_runs if run["algorithm_name"] == algorithm]
                angles = [evaluation(run)["angular_error_history_degrees"][index] for run in group]
                regrets = [evaluation(run)["normalized_regret_history"][index] for run in group]
                row.extend([
                    f"{sum(value is not None for value in angles)}/5",
                    fmt(valid_mean(angles)),
                    fmt(valid_mean(regrets), 5),
                ])
            rows.append(row)
        story.extend([
            base.table(rows, [24] + [39, 44, 48] * 4, font=5.8),
            Paragraph("All means are conditional on a valid estimate at that time.", small),
            PageBreak(),
        ])

    story.append(Paragraph("7. Complete run summaries", h1))
    summary_rows = [[
        "Scenario", "Seed", "Algorithm", "Status", "Angle", "Regret",
        "First joint", "Stable joint", "First invalid", "Seconds",
    ]]
    for run in runs:
        item = evaluation(run)
        summary_rows.append([
            run["metadata"]["comparison_family"],
            run["seed"],
            SHORT[run["algorithm_name"]],
            item["final_status"],
            fmt(item["final_angular_error_degrees"]),
            fmt(item["final_normalized_regret"], 5),
            item["first_joint_threshold_step"] or "-",
            item["stable_joint_threshold_step"] or "-",
            run["metadata"]["first_invalid_step"] or "-",
            f"{run['runtime_seconds']:.3f}",
        ])
    for start in range(0, len(summary_rows) - 1, 26):
        story.extend([
            base.table(
                [summary_rows[0]] + summary_rows[1 + start:1 + start + 26],
                [100, 28, 65, 72, 48, 55, 48, 48, 48, 48],
                font=6.2,
            ),
            PageBreak(),
        ])

    story.extend([
        Paragraph("8. Complete 3,200-step appendix", h1),
        Paragraph(
            "Every observed step is included below. Vectors are rounded to three decimals for "
            "readability; the versioned JSON checkpoints preserve full precision.",
            body,
        ),
        PageBreak(),
    ])
    for run in runs:
        item = evaluation(run)
        story.append(Paragraph(
            f"{run['metadata']['comparison_title']} | seed {run['seed']} | {SHORT[run['algorithm_name']]}",
            h2,
        ))
        rows = [["t", "status", "angle", "regret", "theta_hat", "S", "observed Y"]]
        for index, record in enumerate(run["records"]):
            rows.append([
                index + 1,
                item["estimate_status_history"][index],
                fmt(item["angular_error_history_degrees"][index], 2),
                fmt(item["normalized_regret_history"][index], 4),
                vector(record["theta_hat_after"]),
                vector(record["query"]),
                vector(record["observed_decision"]),
            ])
        story.extend([
            base.table(rows, [18, 65, 38, 42, 145, 145, 145], font=5.1),
            PageBreak(),
        ])

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


def main():
    runs = load_runs()
    configure_base_helpers()
    base.make_charts(runs)
    build_pdf(runs)


if __name__ == "__main__":
    main()
