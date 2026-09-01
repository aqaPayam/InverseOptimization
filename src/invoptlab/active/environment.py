from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..exceptions import ValidationError
from .config import ActiveScenarioConfig
from .decision_spaces import DecisionSpace, make_decision_space
from .experts import Expert, make_expert
from .noise import ObservationNoise, ParameterNoise, make_observation_noise, make_parameter_noise
from .query_spaces import QuerySpace, make_query_space
from .public import PublicDecisionProblem
from .types import AlgorithmContext, EnvironmentFeedback


Array = np.ndarray


def _unit_vector(value: Array, dimension: int, name: str) -> Array:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != dimension or not np.all(np.isfinite(vector)):
        raise ValidationError(f"{name} must be finite and have dimension {dimension}")
    norm = np.linalg.norm(vector)
    if norm <= 1e-15:
        raise ValidationError(f"{name} cannot be zero")
    return vector / norm


class ActiveInverseEnvironment:
    """Algorithm-independent active inverse-optimization environment.

    The algorithm receives only ``AlgorithmObservation`` objects. Latent values
    such as theta*, the expert's perturbed parameter, and the true decision are
    returned to the benchmark runner for storage and future evaluation.
    """

    def __init__(self, config: ActiveScenarioConfig):
        self.config = config
        self.dimension = config.dimension
        self.horizon = config.horizon
        self.reset()

    def reset(self, *, seed: int | None = None) -> "ActiveInverseEnvironment":
        if seed is not None:
            self.config = replace(self.config, seed=int(seed))
        sequence = np.random.SeedSequence(self.config.seed)
        streams = sequence.spawn(8)
        (
            theta_seed,
            decision_seed,
            query_seed,
            reference_seed,
            expert_seed,
            parameter_seed,
            observation_seed,
            auxiliary_seed,
        ) = streams
        theta_rng = np.random.default_rng(theta_seed)
        decision_rng = np.random.default_rng(decision_seed)
        query_rng = np.random.default_rng(query_seed)
        reference_rng = np.random.default_rng(reference_seed)
        self.expert_rng = np.random.default_rng(expert_seed)
        self.parameter_rng = np.random.default_rng(parameter_seed)
        self.observation_rng = np.random.default_rng(observation_seed)
        self.auxiliary_rng = np.random.default_rng(auxiliary_seed)

        if self.config.true_theta is None:
            self.theta_true = _unit_vector(theta_rng.normal(size=self.dimension), self.dimension, "theta_true")
        else:
            supplied = np.asarray(self.config.true_theta, dtype=float)
            self.theta_true = (
                _unit_vector(supplied, self.dimension, "theta_true")
                if self.config.normalize_true_theta
                else supplied.reshape(-1).copy()
            )

        self.decision_space: DecisionSpace = make_decision_space(
            self.config.decision_space,
            self.dimension,
            decision_rng,
        )
        self.query_space: QuerySpace = make_query_space(
            self.config.query_space,
            self.dimension,
            self.theta_true,
            self.decision_space,
            query_rng,
        )
        self.expert: Expert = make_expert(
            self.config.expert,
            self.decision_space,
            self.theta_true,
            self.query_space.candidates,
            reference_rng,
        )
        self.parameter_noise: ParameterNoise = make_parameter_noise(
            self.config.parameter_noise,
            self.dimension,
        )
        self.observation_noise: ObservationNoise = make_observation_noise(
            self.config.observation_noise,
            self.dimension,
        )
        self.parameter_noise.reset(self.theta_true, self.parameter_rng)
        self.current_step = 0
        self.used_query_indices: set[int] = set()
        self.feedback_history: list[EnvironmentFeedback] = []
        return self

    @staticmethod
    def objective(theta: Array, query: Array, decision: Array) -> float:
        return float(np.dot(np.asarray(query) * np.asarray(theta), np.asarray(decision)))

    def algorithm_context(self, *, algorithm_seed: int | None = None) -> AlgorithmContext:
        public = self.config.to_dict()
        public.pop("true_theta", None)
        public["objective"] = "(s * theta)^T x"
        decision_problem = PublicDecisionProblem(self.decision_space)
        public["decision_problem"] = decision_problem.description()
        public["query_space_metadata"] = {
            "kind": self.query_space.kind.value,
            "candidate_count": self.query_space.size,
        }
        return AlgorithmContext(
            dimension=self.dimension,
            horizon=self.horizon,
            query_candidates=self.query_space.candidates.copy(),
            decision_problem=decision_problem,
            seed=self.config.seed if algorithm_seed is None else int(algorithm_seed),
            scenario_name=self.config.name,
            public_environment=public,
        )

    def validate_query(self, query: Array) -> tuple[Array, int]:
        value = _unit_vector(query, self.dimension, "query")
        if not np.allclose(value, np.asarray(query, dtype=float).reshape(-1), atol=1e-7):
            raise ValidationError("the algorithm must return a unit-norm query")
        index = self.query_space.index_of(value)
        if index is None:
            raise ValidationError("the selected query is not in the configured query set")
        if not self.query_space.allow_repeated_queries and index in self.used_query_indices:
            raise ValidationError("the query space does not allow repeated queries")
        return value, index

    def step(self, query: Array) -> EnvironmentFeedback:
        if self.current_step >= self.horizon:
            raise RuntimeError("the environment horizon has been exhausted")
        value, query_index = self.validate_query(query)
        self.current_step += 1
        self.used_query_indices.add(query_index)
        expert_parameter, parameter_metadata = self.parameter_noise.apply(
            self.theta_true,
            value,
            self.current_step,
            self.parameter_rng,
        )
        expert_response = self.expert.respond(value, expert_parameter, self.expert_rng)
        if not self.decision_space.contains(expert_response.decision):
            raise RuntimeError("expert produced an infeasible decision")
        observed = self.observation_noise.apply(
            expert_response.decision,
            value,
            self.decision_space,
            self.current_step,
            self.observation_rng,
        )
        if observed.mask is None and not self.decision_space.contains(observed.decision):
            raise RuntimeError("observation channel produced an infeasible decision")
        feedback = EnvironmentFeedback(
            step=self.current_step,
            query=value.copy(),
            expert_parameter=np.asarray(expert_parameter, dtype=float).copy(),
            true_decision=np.asarray(expert_response.decision).copy(),
            observed_decision=np.asarray(observed.decision).copy(),
            observation_mask=None if observed.mask is None else np.asarray(observed.mask).copy(),
            objective_value=expert_response.objective_value,
            expert_metadata=dict(expert_response.metadata),
            parameter_noise_metadata=dict(parameter_metadata),
            observation_noise_metadata=dict(observed.metadata),
        )
        self.feedback_history.append(feedback)
        return feedback

    def latent_snapshot(self) -> dict[str, Any]:
        return {
            "theta_true": self.theta_true.copy(),
            "current_step": self.current_step,
            "used_query_indices": sorted(self.used_query_indices),
            "decision_space": type(self.decision_space).__name__,
            "query_space": self.query_space.kind.value,
            "expert": type(self.expert).__name__,
            "parameter_noise": type(self.parameter_noise).__name__,
            "observation_noise": type(self.observation_noise).__name__,
        }
