"""Online ASL mirror descent with uniform active queries.

The parameter update is the online specialization of the paper's SAMD
algorithm.  The signed parameter is represented as ``theta_plus -
theta_minus`` in a nonnegative ``2d``-dimensional space and updated with the
entropy mirror map.  Query selection is deliberately external to SAMD and is
uniform, so Pedro and this algorithm can be compared under the same query
policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..exceptions import CapabilityError, ValidationError
from .algorithms import ActiveAlgorithm
from .public import PublicDecisionProblem
from .types import ActiveAction, AlgorithmContext, AlgorithmObservation


Array = np.ndarray


@dataclass(frozen=True, slots=True)
class OnlineSAMDConfig:
    """Configuration for one online SAMD update per expert observation.

    ``l1_radius=None`` resolves to ``sqrt(d)``.  This contains every unit-L2
    ground-truth direction used by the active benchmark, without consulting a
    hidden parameter or an offline optimum.
    """

    learning_rate: float = 1.0
    l1_radius: float | None = None
    margin_scale: float = 1.0
    normalize_subgradient: bool = True
    tolerance: float = 1e-12
    exponent_clip: float = 50.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or not np.isfinite(self.learning_rate):
            raise ValidationError("learning_rate must be finite and positive")
        if self.l1_radius is not None and (
            self.l1_radius <= 0 or not np.isfinite(self.l1_radius)
        ):
            raise ValidationError("l1_radius must be finite and positive when supplied")
        if self.margin_scale < 0 or not np.isfinite(self.margin_scale):
            raise ValidationError("margin_scale must be finite and nonnegative")
        if self.tolerance <= 0 or not np.isfinite(self.tolerance):
            raise ValidationError("tolerance must be finite and positive")
        if self.exponent_clip <= 0 or not np.isfinite(self.exponent_clip):
            raise ValidationError("exponent_clip must be finite and positive")


class UniformOnlineSAMDAlgorithm(ActiveAlgorithm):
    """Uniform next S with one signed exponentiated ASL update per new pair.

    The loss-augmented competitor is solved exactly by enumerating the public
    finite decision set.  This is the epsilon=0 special case of SAMD.  Partial
    or infeasible observations are skipped explicitly rather than filled using
    latent information.
    """

    name = "Uniform Online SAMD"

    def __init__(self, config: OnlineSAMDConfig | None = None):
        self.config = config or OnlineSAMDConfig()

    def reset(self, context: AlgorithmContext, rng: np.random.Generator) -> None:
        if not isinstance(context.decision_problem, PublicDecisionProblem):
            raise ValidationError("Uniform Online SAMD requires a public decision problem")
        try:
            decisions = np.asarray(
                context.decision_problem.enumerate_decisions(), dtype=float
            )
        except CapabilityError as exc:
            raise ValidationError(
                "Uniform Online SAMD currently requires a finite enumerable decision space"
            ) from exc
        if (
            decisions.ndim != 2
            or decisions.shape[0] < 1
            or decisions.shape[1] != context.dimension
            or not np.all(np.isfinite(decisions))
        ):
            raise ValidationError(
                "Uniform Online SAMD requires finite decisions matching the parameter dimension"
            )

        self.context = context
        self.decision_problem = context.decision_problem
        self.rng = rng
        self._decisions = decisions
        self.l1_radius_ = float(
            np.sqrt(context.dimension)
            if self.config.l1_radius is None
            else self.config.l1_radius
        )
        # Strict positivity is required by the entropy mirror map.  Equal
        # positive and negative masses encode the neutral initial theta=0.
        self._split_parameter = np.full(
            2 * context.dimension,
            self.l1_radius_ / (2 * context.dimension),
            dtype=float,
        )
        self._estimate = np.zeros(context.dimension, dtype=float)
        self.update_count_ = 0
        self.estimate_status_ = "insufficient_information"
        self.failure_reason_: str | None = "no ASL update has been observed"
        self.update_history_: list[dict[str, Any]] = []
        query_config = context.public_environment.get("query_space", {})
        self.allow_repeated_queries = bool(
            query_config.get("allow_repeated_queries", True)
        )
        self._available_queries = list(range(context.query_candidates.shape[0]))

    def _select_query_index(self) -> int:
        if self.allow_repeated_queries:
            return int(self.rng.integers(self.context.query_candidates.shape[0]))
        if not self._available_queries:
            raise ValidationError("the non-repeating query set has been exhausted")
        position = int(self.rng.integers(len(self._available_queries)))
        return self._available_queries.pop(position)

    def propose(self, history) -> ActiveAction:
        del history
        index = self._select_query_index()
        return ActiveAction(
            query=self.context.query_candidates[index].copy(),
            theta_hat=self._estimate.copy(),
            diagnostics={
                "query_rule": "uniform-random",
                "candidate_index": index,
                "estimator": "online-samd-signed-exponentiated-asl",
                "updates_before_query": self.update_count_,
                "l1_radius": self.l1_radius_,
                "estimate_status_before_query": self.estimate_status_,
            },
        )

    def _skip(self, observation: AlgorithmObservation, reason: str) -> None:
        if np.linalg.norm(self._estimate) <= self.config.tolerance:
            self.estimate_status_ = "insufficient_information"
            self.failure_reason_ = reason
        self.update_history_.append(
            {
                "step": observation.step,
                "theta_hat": self._estimate.copy(),
                "estimate_status": self.estimate_status_,
                "failure_reason": self.failure_reason_,
                "update_applied": False,
                "skipped_reason": reason,
                "update_count": self.update_count_,
                "l1_radius": self.l1_radius_,
            }
        )

    def observe(self, observation: AlgorithmObservation) -> None:
        query = np.asarray(observation.query, dtype=float).reshape(-1)
        observed = np.asarray(observation.observed_decision, dtype=float).reshape(-1)
        if query.size != self.context.dimension or observed.size != self.context.dimension:
            raise ValidationError("query and observed decision must match the SAMD dimension")
        if observation.observation_mask is not None:
            mask = np.asarray(observation.observation_mask).reshape(-1)
            if mask.size != self.context.dimension:
                raise ValidationError("observation mask has the wrong dimension")
            if np.any(mask == 0):
                self._skip(observation, "the complete observed decision is unavailable")
                return
        if not self.decision_problem.contains(observed):
            self._skip(observation, "the observed decision is infeasible")
            return

        theta_before = self._estimate.copy()

        # argmax_x d(y,x) - theta^T phi(s,x), written as a minimization.
        costs = query * self._estimate
        distances = self.config.margin_scale * np.sum(
            np.abs(self._decisions - observed), axis=1
        )
        augmented_objectives = self._decisions @ costs - distances
        competitor_index = int(np.argmin(augmented_objectives))
        competitor = self._decisions[competitor_index].copy()
        margin = float(distances[competitor_index])

        gradient = query * (observed - competitor)
        sample_asl_value_before = max(
            0.0,
            float(np.dot(theta_before, gradient)) + margin,
        )
        split_gradient = np.concatenate([gradient, -gradient])
        dual_norm = float(np.linalg.norm(split_gradient, ord=np.inf))
        update_gradient = split_gradient.copy()
        if self.config.normalize_subgradient and dual_norm > self.config.tolerance:
            update_gradient /= dual_norm

        self.update_count_ += 1
        step_size = self.config.learning_rate / np.sqrt(self.update_count_)
        exponent = np.clip(
            -step_size * update_gradient,
            -self.config.exponent_clip,
            self.config.exponent_clip,
        )
        self._split_parameter *= np.exp(exponent)
        split_l1 = float(np.sum(self._split_parameter))
        if split_l1 > self.l1_radius_:
            self._split_parameter *= self.l1_radius_ / split_l1
            split_l1 = self.l1_radius_
        dimension = self.context.dimension
        self._estimate = (
            self._split_parameter[:dimension]
            - self._split_parameter[dimension:]
        )
        estimate_norm = float(np.linalg.norm(self._estimate))
        if estimate_norm <= self.config.tolerance:
            self.estimate_status_ = "insufficient_information"
            self.failure_reason_ = "the ASL update was uninformative"
        else:
            self.estimate_status_ = "valid"
            self.failure_reason_ = None
        self.update_history_.append(
            {
                "step": observation.step,
                "theta_hat": self._estimate.copy(),
                "estimate_status": self.estimate_status_,
                "failure_reason": self.failure_reason_,
                "update_applied": True,
                "skipped_reason": None,
                "update_count": self.update_count_,
                "step_size": step_size,
                "subgradient": gradient.copy(),
                "subgradient_dual_norm": dual_norm,
                "subgradient_normalized": self.config.normalize_subgradient,
                "competitor": competitor,
                "margin": margin,
                "sample_asl_value_before_update": sample_asl_value_before,
                "split_l1_norm": split_l1,
                "theta_l1_norm": float(np.linalg.norm(self._estimate, ord=1)),
                "l1_radius": self.l1_radius_,
                "loss_augmented_oracle": "exact-enumeration",
                "epsilon": 0.0,
            }
        )

    def current_estimate(self) -> Array:
        return self._estimate.copy()

    def diagnostics(self) -> Mapping[str, Any]:
        if not self.update_history_:
            return {
                "theta_hat": self._estimate.copy(),
                "estimate_status": self.estimate_status_,
                "failure_reason": self.failure_reason_,
                "update_count": self.update_count_,
                "l1_radius": self.l1_radius_,
            }
        return dict(self.update_history_[-1])


def create_uniform_online_samd_algorithm() -> UniformOnlineSAMDAlgorithm:
    """Stable import-path factory for configuration-driven benchmarks."""

    return UniformOnlineSAMDAlgorithm()
