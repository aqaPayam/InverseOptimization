from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np

from .types import ActiveAction, AlgorithmContext, AlgorithmObservation


Array = np.ndarray


class ActiveAlgorithm(ABC):
    """Minimal interface for algorithms evaluated by the benchmark.

    ``propose`` returns both the current parameter estimate and next query.
    After the environment returns the public observation, ``observe`` updates
    the algorithm and ``current_estimate`` exposes theta_hat after that update.
    """

    name: str = "active-algorithm"

    @abstractmethod
    def reset(self, context: AlgorithmContext, rng: np.random.Generator) -> None:
        ...

    @abstractmethod
    def propose(self, history: Sequence[AlgorithmObservation]) -> ActiveAction:
        ...

    @abstractmethod
    def observe(self, observation: AlgorithmObservation) -> None:
        ...

    @abstractmethod
    def current_estimate(self) -> Array:
        ...


class CallbackActiveAlgorithm(ActiveAlgorithm):
    """Adapter for concise user algorithms implemented as Python callbacks."""

    def __init__(
        self,
        propose: Callable[[AlgorithmContext, Sequence[AlgorithmObservation]], ActiveAction],
        update: Callable[[AlgorithmContext, AlgorithmObservation], Array | None] | None = None,
        *,
        name: str = "callback-algorithm",
    ):
        self._propose_callback = propose
        self._update_callback = update
        self.name = name

    def reset(self, context, rng) -> None:
        self.context = context
        self.rng = rng
        self._estimate = np.zeros(context.dimension)

    def propose(self, history) -> ActiveAction:
        action = self._propose_callback(self.context, history)
        self._estimate = np.asarray(action.theta_hat, dtype=float).copy()
        return action

    def observe(self, observation) -> None:
        if self._update_callback is not None:
            estimate = self._update_callback(self.context, observation)
            if estimate is not None:
                self._estimate = np.asarray(estimate, dtype=float).copy()

    def current_estimate(self) -> Array:
        return self._estimate.copy()


class RandomActiveAlgorithm(ActiveAlgorithm):
    """Meaningless baseline used only to verify benchmark plumbing."""

    name = "random-smoke-test"

    def __init__(self, *, without_replacement: bool = False):
        self.without_replacement = without_replacement

    @staticmethod
    def _random_theta(dimension: int, rng: np.random.Generator) -> Array:
        value = rng.normal(size=dimension)
        norm = np.linalg.norm(value)
        return value / norm if norm > 1e-15 else np.eye(1, dimension, 0).reshape(-1)

    def reset(self, context, rng) -> None:
        self.context = context
        self.rng = rng
        self._estimate = self._random_theta(context.dimension, rng)
        self._available = list(range(context.query_candidates.shape[0]))

    def propose(self, history) -> ActiveAction:
        del history
        if self.without_replacement and self._available:
            position = int(self.rng.integers(len(self._available)))
            index = self._available.pop(position)
        else:
            index = int(self.rng.integers(self.context.query_candidates.shape[0]))
        return ActiveAction(
            query=self.context.query_candidates[index].copy(),
            theta_hat=self._estimate.copy(),
            diagnostics={"candidate_index": index, "purpose": "plumbing-only"},
        )

    def observe(self, observation) -> None:
        del observation
        self._estimate = self._random_theta(self.context.dimension, self.rng)

    def current_estimate(self) -> Array:
        return self._estimate.copy()

