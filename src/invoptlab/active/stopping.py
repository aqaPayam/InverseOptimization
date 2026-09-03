from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..exceptions import ValidationError
from .config import ActiveScenarioConfig, _jsonable
from .decision_spaces import DecisionSpace
from .evaluation import estimate_status, normalized_test_regret, sample_scenario_hidden_queries


Array = np.ndarray


@dataclass(slots=True)
class RegretStoppingConfig:
    enabled: bool = True
    test_query_count: int = 128
    seed: int = 0
    zero_regret_tolerance: float = 1e-8
    minimum_steps: int = 1
    consecutive_successes: int = 1
    query_distribution: str = "uniform_unit_sphere"

    def __post_init__(self) -> None:
        if (
            self.test_query_count < 1
            or self.minimum_steps < 1
            or self.consecutive_successes < 1
        ):
            raise ValidationError("stopping test-query count and minimum steps must be positive")
        if self.zero_regret_tolerance < 0:
            raise ValidationError("stopping regret tolerance must be nonnegative")
        if self.query_distribution not in {"uniform_unit_sphere", "scenario"}:
            raise ValidationError("query_distribution must be uniform_unit_sphere or scenario")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class RegretStoppingCheck:
    step: int
    should_stop: bool
    mean_normalized_regret: float | None
    maximum_normalized_regret: float | None
    zero_regret_rate: float | None
    consecutive_successes: int = 0
    reason: str | None = None
    estimate_status: str = "valid"

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
        self.success_streak = 0

    def reset(
        self,
        scenario: ActiveScenarioConfig,
        theta_true: Array,
        decision_space: DecisionSpace,
    ) -> None:
        self.history = []
        self.theta_true = np.asarray(theta_true, dtype=float).copy()
        self.decision_space = decision_space
        self.success_streak = 0
        self.test_queries = sample_scenario_hidden_queries(
            scenario,
            self.theta_true,
            self.config.test_query_count,
            evaluation_seed=self.config.seed,
            distribution=self.config.query_distribution,
            decision_space=decision_space,
        )

    def check(self, theta_hat: Array, step: int,
              diagnostics: dict | None = None) -> RegretStoppingCheck:
        if self.test_queries is None or self.theta_true is None or self.decision_space is None:
            raise RuntimeError("the regret stopping rule must be reset before use")
        status, failure_reason = estimate_status(theta_hat, diagnostics)
        if status != "valid":
            self.success_streak = 0
            check = RegretStoppingCheck(int(step), False, None, None, None,
                reason=failure_reason or "invalid estimate cannot meet stopping criterion",
                estimate_status=status)
            self.history.append(check)
            return check
        mean_regret, zero_rate, regrets = normalized_test_regret(
            theta_hat,
            self.theta_true,
            self.test_queries,
            self.decision_space,
            zero_regret_tolerance=self.config.zero_regret_tolerance,
        )
        maximum_regret = float(np.max(regrets))
        all_zero = bool(np.all(regrets <= self.config.zero_regret_tolerance))
        self.success_streak = self.success_streak + 1 if all_zero else 0
        should_stop = bool(
            self.config.enabled
            and step >= self.config.minimum_steps
            and self.success_streak >= self.config.consecutive_successes
        )
        check = RegretStoppingCheck(
            step=int(step),
            should_stop=should_stop,
            mean_normalized_regret=mean_regret,
            maximum_normalized_regret=maximum_regret,
            zero_regret_rate=zero_rate,
            consecutive_successes=self.success_streak,
            reason=(
                None
                if not should_stop
                else "zero hidden-test regret"
                if self.config.consecutive_successes == 1
                else f"zero validation regret for {self.config.consecutive_successes} consecutive steps"
            ),
        )
        self.history.append(check)
        return check
