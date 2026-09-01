from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .config import ExpertConfig, ExpertKind
from .decision_spaces import DecisionSpace


Array = np.ndarray


@dataclass(slots=True)
class ExpertResponse:
    decision: Array
    objective_value: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Expert(ABC):
    kind: ExpertKind

    def __init__(self, decision_space: DecisionSpace):
        self.decision_space = decision_space

    @abstractmethod
    def respond(
        self,
        query: Array,
        theta: Array,
        rng: np.random.Generator,
    ) -> ExpertResponse:
        ...


class MinExpert(Expert):
    kind = ExpertKind.MIN

    def __init__(self, decision_space: DecisionSpace, tie_breaking: str = "lexicographic"):
        super().__init__(decision_space)
        self.tie_breaking = tie_breaking

    def respond(self, query, theta, rng) -> ExpertResponse:
        cost = np.asarray(query, dtype=float) * np.asarray(theta, dtype=float)
        decision = self.decision_space.min_decision(
            cost,
            rng,
            tie_breaking=self.tie_breaking,
        )
        return ExpertResponse(
            decision=decision,
            objective_value=float(np.dot(cost, decision)),
            metadata={"kind": self.kind.value, "tie_breaking": self.tie_breaking},
        )


class GibbsExpert(Expert):
    kind = ExpertKind.GIBBS

    def __init__(
        self,
        decision_space: DecisionSpace,
        normalized_temperature: float,
        reference_gap: float,
        *,
        burn_in: int = 40,
        steps: int = 20,
    ):
        super().__init__(decision_space)
        self.normalized_temperature = float(normalized_temperature)
        self.reference_gap = float(reference_gap)
        self.temperature = self.normalized_temperature * self.reference_gap
        self.burn_in = int(burn_in)
        self.steps = int(steps)

    def respond(self, query, theta, rng) -> ExpertResponse:
        cost = np.asarray(query, dtype=float) * np.asarray(theta, dtype=float)
        decision = self.decision_space.sample_gibbs(
            cost,
            self.temperature,
            rng,
            burn_in=self.burn_in,
            steps=self.steps,
        )
        return ExpertResponse(
            decision=decision,
            objective_value=float(np.dot(cost, decision)),
            metadata={
                "kind": self.kind.value,
                "normalized_temperature": self.normalized_temperature,
                "reference_gap": self.reference_gap,
                "temperature": self.temperature,
            },
        )


def compute_reference_gap(
    decision_space: DecisionSpace,
    theta_true: Array,
    query_candidates: Array,
    rng: np.random.Generator,
    *,
    maximum_queries: int = 64,
) -> float:
    candidates = np.asarray(query_candidates, dtype=float)
    if candidates.shape[0] > maximum_queries:
        indices = np.linspace(0, candidates.shape[0] - 1, maximum_queries, dtype=int)
        candidates = candidates[indices]
    gaps = []
    for query in candidates:
        gap = decision_space.reference_energy_gap(query * theta_true, rng)
        if np.isfinite(gap) and gap > 1e-10:
            gaps.append(float(gap))
    if gaps:
        return float(np.median(gaps))
    effective_scales = np.linalg.norm(candidates * theta_true[None, :], axis=1)
    fallback = float(np.median(effective_scales) / max(1.0, np.sqrt(theta_true.size)))
    return max(fallback, 1e-6)


def make_expert(
    config: ExpertConfig,
    decision_space: DecisionSpace,
    theta_true: Array,
    query_candidates: Array,
    reference_rng: np.random.Generator,
) -> Expert:
    if config.kind == ExpertKind.MIN:
        return MinExpert(decision_space, tie_breaking=config.tie_breaking)
    reference_gap = config.reference_gap or compute_reference_gap(
        decision_space,
        theta_true,
        query_candidates,
        reference_rng,
    )
    return GibbsExpert(
        decision_space,
        config.normalized_temperature,
        reference_gap,
        burn_in=config.gibbs_burn_in,
        steps=config.gibbs_steps,
    )

