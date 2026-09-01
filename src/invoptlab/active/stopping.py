from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..exceptions import ValidationError
from .config import ActiveScenarioConfig, _jsonable
from .decision_spaces import DecisionSpace
from .evaluation import normalized_test_regret, sample_uniform_test_queries


Array = np.ndarray


@dataclass(slots=True)
class RegretStoppingConfig:
    enabled: bool = True
    test_query_count: int = 128
    seed: int = 0
    zero_regret_tolerance: float = 1e-8
    minimum_steps: int = 1

    def __post_init__(self) -> None:
        if self.test_query_count < 1 or self.minimum_steps < 1:
            raise ValidationError("stopping test-query count and minimum steps must be positive")
        if self.zero_regret_tolerance < 0:
            raise ValidationError("stopping regret tolerance must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class RegretStoppingCheck:
    step: int
    should_stop: bool
    mean_normalized_regret: float
    maximum_normalized_regret: float
    zero_regret_rate: float
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


class RegretStoppingRule:
    """External benchmark rule based on hidden clean test-query regret."""

    def __init__(self, config: RegretStoppingConfig | None = None):
        self.config = config or RegretStoppingConfig()
        self.history: list[RegretStoppingCheck] = []
        self.test_queries: Array | None = None
        self.theta_true: Array | None = None
        self.decision_space: DecisionSpace | None = None

    def reset(
        self,
        scenario: ActiveScenarioConfig,
        theta_true: Array,
        decision_space: DecisionSpace,
    ) -> None:
        self.history = []
        self.theta_true = np.asarray(theta_true, dtype=float).copy()
        self.decision_space = decision_space
        self.test_queries = sample_uniform_test_queries(
            scenario.dimension,
            self.config.test_query_count,
            scenario_seed=scenario.seed,
            evaluation_seed=self.config.seed,
        )

    def check(self, theta_hat: Array, step: int) -> RegretStoppingCheck:
        if self.test_queries is None or self.theta_true is None or self.decision_space is None:
            raise RuntimeError("the regret stopping rule must be reset before use")
        mean_regret, zero_rate, regrets = normalized_test_regret(
            theta_hat,
            self.theta_true,
            self.test_queries,
            self.decision_space,
            zero_regret_tolerance=self.config.zero_regret_tolerance,
        )
        maximum_regret = float(np.max(regrets))
        all_zero = bool(np.all(regrets <= self.config.zero_regret_tolerance))
        should_stop = bool(
            self.config.enabled and step >= self.config.minimum_steps and all_zero
        )
        check = RegretStoppingCheck(
            step=int(step),
            should_stop=should_stop,
            mean_normalized_regret=mean_regret,
            maximum_normalized_regret=maximum_regret,
            zero_regret_rate=zero_rate,
            reason="zero hidden-test regret" if should_stop else None,
        )
        self.history.append(check)
        return check
