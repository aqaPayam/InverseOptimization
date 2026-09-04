"""Build reproducible summary tables from the saved four-algorithm trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "active" / "17_four_algorithm_comparison"
ORDER = (
    "Pedro algorithm",
    "Genious Pedro",
    "Score base model",
    "Uniform Online SAMD",
)


def mean(values: list[float | None]) -> float:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def main() -> None:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in SOURCE.glob("*.json")]
    if len(runs) != 160:
        raise RuntimeError(f"expected 160 complete runs, found {len(runs)}")

    family_order = []
    for run in runs:
        family = run["metadata"]["comparison_family"]
        if family not in family_order:
            family_order.append(family)

    final_rows = []
    for family in family_order:
        for algorithm in ORDER:
            group = [
                run for run in runs
                if run["metadata"]["comparison_family"] == family
                and run["algorithm_name"] == algorithm
            ]
            evaluations = [run["evaluation"] for run in group]
            stable_joint = [
                item["stable_joint_threshold_step"]
                for item in evaluations
                if item["stable_joint_threshold_step"] is not None
            ]
            final_rows.append({
                "scenario": group[0]["metadata"]["comparison_title"],
                "family": family,
                "algorithm": algorithm,
                "valid_at_T": sum(item["final_estimate_valid"] for item in evaluations),
                "ever_invalid": sum(
                    run["metadata"]["first_invalid_step"] is not None for run in group
                ),
                "final_angle_mean_degrees_valid_only": mean([
                    item["final_angular_error_degrees"] for item in evaluations
                ]),
                "final_regret_mean_valid_only": mean([
                    item["final_normalized_regret"] for item in evaluations
                ]),
                "final_angle_le_5": sum(
                    item["final_angular_error_degrees"] is not None
                    and item["final_angular_error_degrees"] <= 5
                    for item in evaluations
                ),
                "final_regret_le_0_01": sum(
                    item["final_normalized_regret"] is not None
                    and item["final_normalized_regret"] <= 0.01
                    for item in evaluations
                ),
                "sustained_joint_successes": len(stable_joint),
                "mean_sustained_joint_step_successes_only": (
                    float(np.mean(stable_joint)) if stable_joint else np.nan
                ),
                "mean_runtime_seconds": float(np.mean([
                    run["runtime_seconds"] for run in group
                ])),
            })
    final = pd.DataFrame(final_rows)

    trajectory_rows = []
    for algorithm in ORDER:
        group = [run for run in runs if run["algorithm_name"] == algorithm]
        for step in (1, 5, 10, 15, 20):
            index = step - 1
            trajectory_rows.append({
                "algorithm": algorithm,
                "t": step,
                "valid": sum(run["evaluation"]["valid_estimate_history"][index] for run in group),
                "angle_mean_degrees_valid_only": mean([
                    run["evaluation"]["angular_error_history_degrees"][index]
                    for run in group
                ]),
                "regret_mean_valid_only": mean([
                    run["evaluation"]["normalized_regret_history"][index]
                    for run in group
                ]),
            })
    trajectory = pd.DataFrame(trajectory_rows)

    overall_rows = []
    for algorithm in ORDER:
        group = final[final["algorithm"] == algorithm]
        raw = [run for run in runs if run["algorithm_name"] == algorithm]
        evaluations = [run["evaluation"] for run in raw]
        stable_joint = [
            item["stable_joint_threshold_step"]
            for item in evaluations
            if item["stable_joint_threshold_step"] is not None
        ]
        overall_rows.append({
            "algorithm": algorithm,
            "valid_at_T": int(group["valid_at_T"].sum()),
            "ever_invalid": int(group["ever_invalid"].sum()),
            "final_angle_mean_degrees_valid_only": mean([
                item["final_angular_error_degrees"] for item in evaluations
            ]),
            "final_regret_mean_valid_only": mean([
                item["final_normalized_regret"] for item in evaluations
            ]),
            "final_angle_le_5": sum(
                item["final_angular_error_degrees"] is not None
                and item["final_angular_error_degrees"] <= 5
                for item in evaluations
            ),
            "final_regret_le_0_01": sum(
                item["final_normalized_regret"] is not None
                and item["final_normalized_regret"] <= 0.01
                for item in evaluations
            ),
            "sustained_joint_successes": len(stable_joint),
            "mean_sustained_joint_step_successes_only": (
                float(np.mean(stable_joint)) if stable_joint else np.nan
            ),
            "mean_runtime_seconds": float(np.mean([
                run["runtime_seconds"] for run in raw
            ])),
        })
    overall = pd.DataFrame(overall_rows)

    final.to_csv(SOURCE / "final_summary.csv", index=False)
    trajectory.to_csv(SOURCE / "trajectory_summary.csv", index=False)
    overall.to_csv(SOURCE / "overall_summary.csv", index=False)

    print("OVERALL")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFINAL BY SCENARIO")
    print(final.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTRAJECTORY CHECKPOINTS")
    print(trajectory.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
