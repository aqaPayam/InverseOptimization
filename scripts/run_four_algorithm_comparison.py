"""Run the locked eight-scenario protocol for all four named algorithms."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoptlab.active import (
    NestedLangevinConfig,
    OnlineSAMDConfig,
    build_pedro_score_scenarios,
    run_four_algorithm_design,
)


SCORE_CONFIG = NestedLangevinConfig(
    sampler="gaussian_gibbs",
    theta_samples=16,
    point_estimate="mean",
    query_policy="disagreement",
    query_tie_breaking="random",
    beta=20.0,
    parameter_domain="box",
    bound=1.0,
    tau_schedule=(0.5, 0.1, 0.02),
    gibbs_sweeps=6,
    conditional_slice_steps=4,
    target_slice_steps=32,
    radial_refresh=True,
    record_chain_trace=False,
    workers=1,
)

# Frozen before the benchmark; identical in every family and seed.
SAMD_CONFIG = OnlineSAMDConfig()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "active" / "17_four_algorithm_comparison",
    )
    args = parser.parse_args()

    completed = 0
    total = 8 * len(args.seeds) * 4
    for family in range(8):
        for seed in args.seeds:
            design = build_pedro_score_scenarios(seed=seed, horizon=args.horizon)[family]
            run_four_algorithm_design(
                design,
                args.output,
                score_config=SCORE_CONFIG,
                samd_config=SAMD_CONFIG,
                progress=lambda message: print(message, flush=True),
            )
            completed += 4
            print(f"PROGRESS {completed}/{total} complete", flush=True)
    print(f"FINISHED {completed} runs", flush=True)


if __name__ == "__main__":
    main()
