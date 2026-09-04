"""Run the exact eight-family protocol for the three named algorithms."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoptlab.active import build_pedro_score_scenarios, run_three_algorithm_design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "active" / "16_three_algorithm_comparison",
    )
    args = parser.parse_args()
    count = 0
    total = 8 * len(args.seeds) * 3
    for family in range(8):
        for seed in args.seeds:
            design = build_pedro_score_scenarios(seed=seed, horizon=args.horizon)[family]
            run_three_algorithm_design(
                design, args.output, progress=lambda message: print(message, flush=True)
            )
            count += 3
            print(f"PROGRESS {count}/{total} complete", flush=True)
    print(f"FINISHED {count} runs", flush=True)


if __name__ == "__main__":
    main()
