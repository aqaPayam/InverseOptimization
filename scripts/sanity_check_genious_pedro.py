"""Tiny plumbing/formula check for Genious Pedro; not a benchmark experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoptlab.active import (  # noqa: E402
    ActiveBenchmarkRunner,
    ActiveScenarioConfig,
    DecisionSpaceConfig,
    GeniousPedroAlgorithm,
    QuerySpaceConfig,
    RegretStoppingConfig,
)


def main() -> None:
    root_two = np.sqrt(2.0)
    scenario = ActiveScenarioConfig(
        name="genious-pedro-tiny-sanity-only",
        dimension=2,
        horizon=5,
        seed=17,
        true_theta=[-0.8, 0.6],
        decision_space=DecisionSpaceConfig(kind="fixed_cardinality", cardinality=1),
        query_space=QuerySpaceConfig(
            kind="explicit",
            candidates=[
                [1.0, 0.0],
                [0.0, 1.0],
                [1/root_two, 1/root_two],
                [1/root_two, -1/root_two],
                [-1/root_two, 1/root_two],
            ],
        ),
    )
    run = ActiveBenchmarkRunner(
        stopping_config=RegretStoppingConfig(enabled=False)
    ).run(scenario, GeniousPedroAlgorithm(), algorithm_seed=17)
    assert run.error is None and len(run.records) == 5 and not run.stopped_early
    assert run.algorithm_name == "Genious Pedro"
    assert run.records[0].action_diagnostics["query_rule"] == "uniform-random-fallback"
    assert "D_0 is empty" in run.records[0].action_diagnostics["fallback_reason"]
    assert all(record.true_decision.tolist() == record.observed_decision.tolist()
               for record in run.records)
    for record in run.records[1:]:
        action = record.action_diagnostics
        assert action["query_rule"] == "minimum-normalized-decision-margin"
        finite = [value for value in action["candidate_margins"] if value is not None]
        assert finite and np.isclose(action["selected_margin"], min(finite))
        assert action["predicted_decision"] is not None
        assert action["nearest_alternative"] is not None
    print("GENIOUS PEDRO SANITY CHECK PASSED")
    print("This was one tiny 2D, five-step plumbing/formula check - not the benchmark.")
    for record in run.records:
        action = record.action_diagnostics
        print(
            f"t={record.step}: rule={action['query_rule']}, "
            f"candidate={action['candidate_index']}, "
            f"margin={action['selected_margin']}, "
            f"incenter_status={record.update_diagnostics['estimate_status']}"
        )


if __name__ == "__main__":
    main()
