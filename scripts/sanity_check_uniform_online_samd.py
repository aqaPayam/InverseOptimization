"""Small correctness check for Uniform Online SAMD; not a benchmark run."""

from __future__ import annotations

import numpy as np

from invoptlab.active import (
    ActiveBenchmarkRunner,
    ActiveScenarioConfig,
    DecisionSpaceConfig,
    QuerySpaceConfig,
    UniformOnlineSAMDAlgorithm,
)


scenario = ActiveScenarioConfig(
    name="uniform-online-samd-sanity",
    dimension=3,
    horizon=4,
    seed=7,
    true_theta=[-1.0, 0.4, 0.8],
    decision_space=DecisionSpaceConfig(kind="fixed_cardinality", cardinality=1),
    query_space=QuerySpaceConfig(
        kind="explicit",
        candidates=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
    ),
)
result = ActiveBenchmarkRunner().run(scenario, UniformOnlineSAMDAlgorithm())
assert result.error is None and len(result.records) == scenario.horizon
assert all(np.all(np.isfinite(record.theta_hat_after)) for record in result.records)
assert [record.update_diagnostics["update_count"] for record in result.records] == [1, 2, 3, 4]
assert all(record.update_diagnostics["epsilon"] == 0.0 for record in result.records)

print("Uniform Online SAMD sanity check passed")
print("updates:", len(result.records))
print("final status:", result.records[-1].update_diagnostics["estimate_status"])
print("final theta_hat:", np.round(result.parameter_history[-1], 6).tolist())
print("final L1 norm:", round(float(np.linalg.norm(result.parameter_history[-1], 1)), 6))
