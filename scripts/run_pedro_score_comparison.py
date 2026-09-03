"""Run the eight agreed families with per-run, source-validated checkpoints."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoptlab.active import build_pedro_score_scenarios, run_pedro_score_design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/active/15_pedro_vs_score_base")
    args = parser.parse_args()
    count = 0
    for family in range(8):
        for seed in args.seeds:
            design = build_pedro_score_scenarios(seed=seed, horizon=args.horizon)[family]
            run_pedro_score_design(design, args.output, progress=lambda s: print(s, flush=True))
            count += 2
            print(f"PROGRESS {count}/{8*len(args.seeds)*2} complete", flush=True)
    print(f"FINISHED {count} runs", flush=True)


if __name__ == "__main__":
    main()
