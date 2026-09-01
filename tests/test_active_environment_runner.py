import numpy as np
import pytest

from invoptlab.active import (
    ActiveBenchmarkGrid,
    ActiveBenchmarkRunner,
    ActiveBenchmarkSuite,
    ActiveScenarioConfig,
    DecisionSpaceConfig,
    DecisionSpaceKind,
    ExpertConfig,
    ExpertKind,
    ObservationNoiseConfig,
    ParameterNoiseConfig,
    QuerySpaceConfig,
    RandomActiveAlgorithm,
    RegretStoppingConfig,
)


NO_EXTERNAL_STOPPING = RegretStoppingConfig(enabled=False)


@pytest.mark.parametrize("decision_kind", list(DecisionSpaceKind))
@pytest.mark.parametrize("expert_kind", list(ExpertKind))
def test_environment_runs_every_decision_and_expert_family(decision_kind, expert_kind):
    scenario = ActiveScenarioConfig(
        name=f"{decision_kind.value}-{expert_kind.value}",
        dimension=5,
        horizon=2,
        seed=3,
        expert=ExpertConfig(kind=expert_kind, temperature="low", gibbs_burn_in=4, gibbs_steps=2),
        decision_space=DecisionSpaceConfig(kind=decision_kind),
        query_space=QuerySpaceConfig(candidate_count=10),
        observation_noise=ObservationNoiseConfig(kind="clean"),
        parameter_noise=ParameterNoiseConfig(kind="none"),
    )
    result = ActiveBenchmarkRunner(stopping_config=NO_EXTERNAL_STOPPING).run(
        scenario, RandomActiveAlgorithm()
    )
    assert len(result.records) == 2
    assert result.queries.shape == (2, 5)
    assert result.parameter_history.shape == (2, 5)
    assert result.metadata["evaluation_applied"] is False
    assert result.metadata["scoring_applied"] is False


def test_active_run_is_reproducible_and_public_observation_hides_latent_state():
    scenario = ActiveScenarioConfig(
        dimension=5,
        horizon=3,
        seed=44,
        query_space=QuerySpaceConfig(candidate_count=12),
        observation_noise=ObservationNoiseConfig(kind="outlier", outlier_probability=0.3),
        parameter_noise=ParameterNoiseConfig(kind="isotropic", sigma=0.1),
    )
    runner = ActiveBenchmarkRunner(stopping_config=NO_EXTERNAL_STOPPING)
    first = runner.run(scenario, RandomActiveAlgorithm())
    second = runner.run(scenario, RandomActiveAlgorithm())
    np.testing.assert_allclose(first.queries, second.queries)
    np.testing.assert_allclose(first.parameter_history, second.parameter_history)
    for left, right in zip(first.records, second.records):
        np.testing.assert_allclose(left.expert_parameter, right.expert_parameter)
        np.testing.assert_allclose(left.observed_decision, right.observed_decision)
    latent_record = first.records[0]
    assert latent_record.true_theta.shape == (5,)  # retained privately by the benchmark
    public_record = latent_record.to_dict(include_latent=False)
    assert "true_theta" not in public_record
    assert "expert_parameter" not in public_record
    assert "true_decision" not in public_record


def test_lazy_grid_and_multi_algorithm_suite():
    grid = ActiveBenchmarkGrid(
        dimensions=(5,),
        experts=(ExpertConfig(kind="min"),),
        decision_spaces=(DecisionSpaceConfig(kind="independent_binary"),),
        query_spaces=(QuerySpaceConfig(kind="balanced", candidate_count=8),),
        observation_noises=(ObservationNoiseConfig(kind="clean"),),
        parameter_noises=(ParameterNoiseConfig(kind="none"),),
        seeds=(1, 2),
        horizon=2,
    )
    assert grid.size == 2
    suite = ActiveBenchmarkSuite.from_grid(grid)
    result = suite.run(
        {
            "random-a": lambda: RandomActiveAlgorithm(),
            "random-b": lambda: RandomActiveAlgorithm(without_replacement=True),
        },
        stopping_config=NO_EXTERNAL_STOPPING,
    )
    assert len(result.runs) == 4
    assert not result.failed_runs


def test_result_can_hide_latent_values_when_exported(tmp_path):
    scenario = ActiveScenarioConfig(
        dimension=5,
        horizon=1,
        query_space=QuerySpaceConfig(candidate_count=6),
    )
    result = ActiveBenchmarkRunner(stopping_config=NO_EXTERNAL_STOPPING).run(
        scenario, RandomActiveAlgorithm()
    )
    public = result.to_dict(include_latent=False)
    assert "true_theta" not in public
    assert "true_theta" not in public["records"][0]
    assert "expert_parameter" not in public["records"][0]
    path = result.save_json(tmp_path / "run.json", include_latent=False)
    assert path.exists()
