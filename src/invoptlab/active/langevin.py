"""Loss-based active learning, with corrected and legacy sampling backends.

The default Gaussian-augmented slice sampler preserves the specified bounded
target. ``projected_langevin`` retains the original v2 Euler approximation for
reproduction; its finite-step stationary bias is not a convergence guarantee.
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

from ..exceptions import CapabilityError, SolverError, ValidationError
from .algorithms import ActiveAlgorithm
from .public import PublicDecisionProblem
from .types import ActiveAction, AlgorithmObservation


Array = np.ndarray
ForwardOptimizer = Callable[[Array, Array], Array]


def _vector(value: Array, dimension: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (dimension,) or not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} must be a finite vector of dimension {dimension}")
    return result


@dataclass(frozen=True)
class NestedLangevinConfig:
    """Small numerical defaults; not a claim of convergence on every scenario.

    ``tau_schedule`` contains VARIANCES. Legacy Euler step-size sequences must
    have one entry per level; they are unused by the default corrected backend.
    Legacy retention uses steps burn_in+1, +1+thinning, ... .
    ``workers`` parallelizes independent trajectories, with separate RNGs and
    public solver copies. One worker avoids threading overhead for tiny runs.
    """

    beta: float = 20.0
    parameter_domain: str = "box"
    bound: float = 1.0
    tau_schedule: tuple[float, ...] = (0.5, 0.1, 0.02)
    inner_step_sizes: tuple[float, ...] = (0.02, 0.005, 0.001)
    outer_step_sizes: tuple[float, ...] = (0.1, 0.02, 0.004)
    inner_steps: int = 64
    inner_burn_in: int = 32
    inner_thinning: int = 4
    outer_steps: int = 8
    theta_samples: int = 16
    initialization_std: float = 1.0
    warm_start: bool = True
    warm_start_renoise_std: float = 0.0
    warm_start_inner: bool = False
    workers: int = 1
    record_chain_trace: bool = True
    max_state_norm: float = 1e6
    sampler: str = "gaussian_gibbs"
    point_estimate: str = "mean"
    query_policy: str = "disagreement"
    query_tie_breaking: str = "random"
    gibbs_sweeps: int = 6
    conditional_slice_steps: int = 4
    target_slice_steps: int = 32
    max_slice_shrinks: int = 1000
    radial_refresh: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.radial_refresh, bool):
            raise ValidationError("radial_refresh must be boolean")
        for name, choices in (
            ("sampler", {"gaussian_gibbs", "projected_langevin"}),
            ("point_estimate", {"first", "mean"}),
            ("query_policy", {"uniform", "disagreement"}),
            ("query_tie_breaking", {"first", "random"}),
        ):
            if getattr(self, name) not in choices:
                raise ValidationError(f"{name} must be one of {sorted(choices)}")
        if self.parameter_domain not in {"box", "ball"}:
            raise ValidationError("parameter_domain must be box or ball (not the sphere)")
        for name in ("beta", "bound", "initialization_std", "max_state_norm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValidationError(f"{name} must be finite and positive")
        if not np.isfinite(self.warm_start_renoise_std) or self.warm_start_renoise_std < 0:
            raise ValidationError("warm_start_renoise_std must be finite and nonnegative")
        for name in ("inner_steps", "inner_thinning", "outer_steps", "theta_samples", "workers",
                     "gibbs_sweeps", "conditional_slice_steps", "target_slice_steps",
                     "max_slice_shrinks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValidationError(f"{name} must be a positive integer")
        if self.theta_samples < 2:
            raise ValidationError("disagreement requires at least two theta samples")
        if (isinstance(self.inner_burn_in, bool)
                or not isinstance(self.inner_burn_in, (int, np.integer))
                or not 0 <= self.inner_burn_in < self.inner_steps):
            raise ValidationError("inner_burn_in must be an integer in [0, inner_steps)")
        for name in ("tau_schedule", "inner_step_sizes", "outer_step_sizes"):
            values = np.asarray(getattr(self, name), dtype=float)
            if (values.ndim != 1 or values.size == 0
                    or not np.all(np.isfinite(values)) or np.any(values <= 0)):
                raise ValidationError(f"{name} must be a nonempty positive finite sequence")
            object.__setattr__(self, name, tuple(float(item) for item in values))
        if np.any(np.diff(self.tau_schedule) >= 0):
            raise ValidationError("tau_schedule must be strictly decreasing")
        if (self.sampler == "projected_langevin" and (
                len(self.inner_step_sizes) != len(self.tau_schedule)
                or len(self.outer_step_sizes) != len(self.tau_schedule))):
            raise ValidationError("step-size sequences must match tau_schedule length")

    def project(self, value: Array) -> Array:
        value = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(value)):
            raise SolverError("non-finite inner state; reduce step sizes")
        if self.parameter_domain == "box":
            return np.clip(value, -self.bound, self.bound)
        norm = float(np.linalg.norm(value))
        return value.copy() if norm <= self.bound else value * (self.bound / norm)


class InverseLossTarget:
    """Loss and interior log-density subgradient using ONLY observed (s, y).

    A callback receives (theta, s), so a future s-dependent feasible set can
    be supplied without changing the sampler. Every evaluation uses the full
    cumulative SUM loss, not a mean or a minibatch approximation.
    """

    def __init__(
        self, dimension: int, beta: float, forward_optimize: ForwardOptimizer,
        observations: Sequence[AlgorithmObservation] = (),
        *, batch_forward_optimize: ForwardOptimizer | None = None,
    ):
        if dimension < 1 or not np.isfinite(beta) or beta <= 0:
            raise ValidationError("target dimension and beta must be positive")
        self.dimension = dimension
        self.beta = float(beta)
        self._forward_optimize = forward_optimize
        self._batch_forward_optimize = batch_forward_optimize
        self._queries = np.empty((0, dimension))
        self._observed = np.empty((0, dimension))
        self.data: list[tuple[Array, Array]] = []
        for observation in observations:
            self.append(observation)
        self.forward_calls = 0
        self.forward_seconds = 0.0
        self.score_calls = 0

    def append(self, observation: AlgorithmObservation) -> None:
        if observation.observation_mask is not None:
            mask = _vector(observation.observation_mask, self.dimension, "observation mask")
            if np.any(mask != 1):
                raise CapabilityError("nested Langevin requires complete observed Y, not partial Y")
        query = _vector(observation.query, self.dimension, "query")
        observed = _vector(observation.observed_decision, self.dimension, "observed Y")
        self.data.append((query.copy(), observed.copy()))
        self._queries = np.vstack([self._queries, query])
        self._observed = np.vstack([self._observed, observed])

    def forward(self, theta: Array, query: Array) -> Array:
        theta = _vector(theta, self.dimension, "theta")
        query = _vector(query, self.dimension, "query")
        started = time.perf_counter()
        try:
            decision = self._forward_optimize(theta, query)
        finally:
            self.forward_calls += 1
            self.forward_seconds += time.perf_counter() - started
        return _vector(decision, self.dimension, "forward decision").copy()

    def loss_and_subgradient(self, theta: Array) -> tuple[float, Array]:
        theta = _vector(theta, self.dimension, "theta")
        if self._batch_forward_optimize is not None and self.data:
            decisions = self.batch_forward(theta, self._queries)
            gradient = np.sum(self._queries * (self._observed - decisions), axis=0)
            return float(theta @ gradient), gradient
        gradient = np.zeros(self.dimension)
        for query, observed in self.data:
            gradient += query * (observed - self.forward(theta, query))
        return float(theta @ gradient), gradient

    def batch_forward(self, theta: Array, queries: Array) -> Array:
        theta = _vector(theta, self.dimension, "theta")
        queries = np.asarray(queries, dtype=float)
        if queries.ndim != 2 or queries.shape[1] != self.dimension or not np.all(np.isfinite(queries)):
            raise ValidationError("query batch must have shape (N, dimension) and be finite")
        if not len(queries):
            return np.empty_like(queries)
        if self._batch_forward_optimize is None:
            return np.vstack([self.forward(theta, s) for s in queries])
        started = time.perf_counter()
        try:
            values = np.asarray(self._batch_forward_optimize(theta, queries), dtype=float)
        finally:
            self.forward_calls += len(queries)  # Equivalent individual MIN problems.
            self.forward_seconds += time.perf_counter() - started
        if values.shape != queries.shape or not np.all(np.isfinite(values)):
            raise ValidationError("batch forward decisions have invalid shape or values")
        return values.copy()

    def loss(self, theta: Array) -> float:
        return self.loss_and_subgradient(theta)[0]

    def target_score(self, theta: Array) -> Array:
        self.score_calls += 1
        return -self.beta * self.loss_and_subgradient(theta)[1]


@dataclass
class InnerChainResult:
    mean: Array
    last_state: Array
    retained_states: Array


@dataclass
class TrajectoryResult:
    theta: Array
    outer_state: Array
    trace: list[dict]
    summary: dict


class GaussianSmoothedSampler:
    """One independent nested trajectory, with an isolated target and RNG."""

    def __init__(
        self, target: InverseLossTarget, config: NestedLangevinConfig,
        rng: np.random.Generator, *, round_index: int = 0, trajectory_id: int = 0,
    ):
        self.target, self.config, self.rng = target, config, rng
        self.round_index, self.trajectory_id = round_index, trajectory_id
        self.trace: list[dict] = []
        self.max_target_score_norm = 0.0
        self.max_smoothed_score_norm = 0.0
        self.max_outer_norm = 0.0
        self.projected_inner_steps = 0
        self.inner_update_count = 0

    def _guard(self, value: Array, name: str) -> None:
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm > self.config.max_state_norm:
            raise SolverError(f"unstable {name}; reduce Langevin step sizes or beta")

    def _log(self, **values) -> None:
        if self.config.record_chain_trace:
            self.trace.append({
                "round": self.round_index, "trajectory": self.trajectory_id,
                "forward_calls": self.target.forward_calls,
                "forward_seconds": self.target.forward_seconds, **values,
            })

    def run_inner_chain(
        self, u: Array, tau: float, step_size: float, *, initial_z: Array | None = None,
        level: int = 0, outer_step: int = 0, phase: str = "sampling",
    ) -> InnerChainResult:
        # u is fixed for the WHOLE conditional chain, and is never projected.
        u = _vector(u, self.target.dimension, "outer state").copy()
        z = self.config.project(u if initial_z is None else initial_z)
        retained: list[Array] = []
        for step in range(1, self.config.inner_steps + 1):
            score = self.target.target_score(z)
            self._guard(score, "target score")
            score_norm = float(np.linalg.norm(score))
            self.max_target_score_norm = max(self.max_target_score_norm, score_norm)
            proposal = (z + step_size * (score + (u - z) / tau)
                        + np.sqrt(2 * step_size) * self.rng.normal(size=z.size))
            self._guard(proposal, "inner proposal")
            z = self.config.project(proposal)
            self.projected_inner_steps += int(not np.array_equal(z, proposal))
            self.inner_update_count += 1
            keep = (step > self.config.inner_burn_in
                    and (step - self.config.inner_burn_in - 1) % self.config.inner_thinning == 0)
            if keep:
                retained.append(z.copy())
            self._log(
                phase=phase, chain="inner", level=level, tau=tau, outer_step=outer_step,
                inner_step=step, target_score_norm=score_norm,
                z_norm=float(np.linalg.norm(z)), u_norm=float(np.linalg.norm(u)), retained=keep,
            )
        states = np.vstack(retained)
        return InnerChainResult(states.mean(axis=0), z.copy(), states)

    @staticmethod
    def estimate_smoothed_score(u: Array, conditional_mean: Array, tau: float) -> Array:
        return (conditional_mean - u) / tau

    def outer_update(self, u: Array, score: Array, step_size: float) -> Array:
        value = u + step_size * score + np.sqrt(2 * step_size) * self.rng.normal(size=u.size)
        self._guard(value, "outer state")
        return value  # Deliberately no projection: q_tau has full R^d support.

    def sample(self, warm_start: Array | None = None) -> TrajectoryResult:
        started = time.perf_counter()
        config = self.config
        if warm_start is not None and config.warm_start:
            u = _vector(warm_start, self.target.dimension, "warm start").copy()
            if config.warm_start_renoise_std:
                u += self.rng.normal(scale=config.warm_start_renoise_std, size=u.size)
        else:
            u = self.rng.normal(scale=config.initialization_std, size=self.target.dimension)
        self._guard(u, "initial outer state")
        self.max_outer_norm = float(np.linalg.norm(u))
        last_z = None
        for level, (tau, delta, eta) in enumerate(zip(
            config.tau_schedule, config.inner_step_sizes, config.outer_step_sizes
        )):
            for step in range(1, config.outer_steps + 1):
                inner = self.run_inner_chain(
                    u, tau, delta, initial_z=last_z if config.warm_start_inner else None,
                    level=level, outer_step=step,
                )
                last_z = inner.last_state
                score = self.estimate_smoothed_score(u, inner.mean, tau)
                self._guard(score, "smoothed score")
                norm = float(np.linalg.norm(score))
                self.max_smoothed_score_norm = max(self.max_smoothed_score_norm, norm)
                u = self.outer_update(u, score, eta)
                self.max_outer_norm = max(self.max_outer_norm, float(np.linalg.norm(u)))
                self._log(
                    phase="sampling", chain="outer", level=level, tau=tau, outer_step=step,
                    inner_step=None, smoothed_score_norm=norm,
                    u_norm=float(np.linalg.norm(u)), z_norm=float(np.linalg.norm(last_z)),
                )
        # A fresh conditional run at the FINAL updated u, not the preceding u.
        final = self.run_inner_chain(
            u, config.tau_schedule[-1], config.inner_step_sizes[-1],
            initial_z=last_z if config.warm_start_inner else None,
            level=len(config.tau_schedule) - 1, phase="extraction",
        )
        return TrajectoryResult(
            theta=final.retained_states[-1].copy(), outer_state=u.copy(), trace=self.trace,
            summary={
                "trajectory": self.trajectory_id,
                "forward_calls": self.target.forward_calls,
                "forward_seconds": self.target.forward_seconds,
                "target_score_calls": self.target.score_calls,
                "runtime_seconds": time.perf_counter() - started,
                "max_target_score_norm": self.max_target_score_norm,
                "max_smoothed_score_norm": self.max_smoothed_score_norm,
                "max_outer_norm": self.max_outer_norm,
                "inner_projection_rate": self.projected_inner_steps / self.inner_update_count,
            },
        )


def disagreement_score(predictions: Array) -> float:
    """Average pairwise squared Euclidean distance in O(M*d), M >= 2."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValidationError("predictions must be a finite (M, d) array with M >= 2")
    centered = values - values.mean(axis=0)
    return float(2.0 / (values.shape[0] - 1) * np.sum(centered * centered))


class NestedLangevinActiveAlgorithm(ActiveAlgorithm):
    """Configurable point estimate; ALL samples for disagreement query selection."""

    name = "nested-langevin-disagreement"

    def __init__(self, config: NestedLangevinConfig | None = None):
        self.config = config or NestedLangevinConfig()
        self.name = f"{self.config.sampler}-{self.config.query_policy}"

    def reset(self, context, rng) -> None:
        if not isinstance(context.decision_problem, PublicDecisionProblem):
            raise ValidationError("nested Langevin requires a public forward decision problem")
        candidates = np.asarray(context.query_candidates, dtype=float)
        if (candidates.ndim != 2 or candidates.shape[1] != context.dimension
                or not candidates.shape[0] or not np.all(np.isfinite(candidates))):
            raise ValidationError("query candidates must be a nonempty finite (N, d) array")
        self.context, self.rng = context, rng
        self.candidates_ = candidates.copy()
        self.observations_: list[AlgorithmObservation] = []
        self.theta_samples_: Array | None = None
        self._available = list(range(len(candidates)))
        self._allow_repeats = context.public_environment.get("query_space", {}).get(
            "allow_repeated_queries", True
        )
        # Independent streams make worker scheduling irrelevant to the samples.
        seeds = rng.integers(0, 2**63, size=self.config.theta_samples, dtype=np.int64)
        self._trajectory_rngs = [np.random.default_rng(int(seed)) for seed in seeds]
        # Selection consumes no sampling RNG. Equal histories give equal samples
        # under uniform and disagreement policies, including parallel workers.
        self._query_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        self._problems = [copy.deepcopy(context.decision_problem) for _ in seeds]
        self.forward_calls_ = 0
        self.forward_seconds_ = 0.0
        self._resample()

    @staticmethod
    def _target(problem, dimension, beta, observations, *, use_batch=False) -> InverseLossTarget:
        # Public MIN uses deterministic tie-breaking; this RNG is not a sampling stream.
        solver_rng = np.random.default_rng(0)
        return InverseLossTarget(
            dimension, beta, lambda theta, s: problem.minimize(theta * s, solver_rng), observations,
            batch_forward_optimize=(lambda theta, s: problem.minimize_batch(theta * s, solver_rng))
            if use_batch else None,
        )

    def _resample(self) -> None:
        started = time.perf_counter()
        previous = self.theta_samples_

        def trajectory(index):
            target = self._target(
                self._problems[index], self.context.dimension, self.config.beta, self.observations_,
                use_batch=self.config.sampler == "gaussian_gibbs",
            )
            from .sampling import GaussianGibbsSampler

            sampler_class = (GaussianGibbsSampler if self.config.sampler == "gaussian_gibbs"
                             else GaussianSmoothedSampler)
            sampler = sampler_class(
                target, self.config, self._trajectory_rngs[index],
                round_index=len(self.observations_), trajectory_id=index,
            )
            return sampler.sample(None if previous is None else previous[index])

        indices = range(self.config.theta_samples)
        if self.config.workers == 1:
            results = [trajectory(index) for index in indices]
        else:
            with ThreadPoolExecutor(max_workers=min(self.config.workers, self.config.theta_samples)) as pool:
                results = list(pool.map(trajectory, indices))
        self.theta_samples_ = np.vstack([result.theta for result in results])
        self.outer_states_ = np.vstack([result.outer_state for result in results])
        self._estimate = (self.theta_samples_[0].copy() if self.config.point_estimate == "first"
                          else self.theta_samples_.mean(axis=0))
        self.forward_calls_ += sum(result.summary["forward_calls"] for result in results)
        self.forward_seconds_ += sum(result.summary["forward_seconds"] for result in results)
        self._sampling_diagnostics = {
            "round": len(self.observations_),
            "estimate_status": "valid" if np.linalg.norm(self._estimate) > 1e-12 else "invalid_estimate",
            "point_estimate_rule": ("first-retained-latent-sample"
                                    if self.config.point_estimate == "first" else "ensemble-mean"),
            "sample_extraction": ("final-target-slice-state"
                                  if self.config.sampler == "gaussian_gibbs"
                                  else "late-inner-z-at-final-u"),
            "sampling_quality": "finite-chain-not-certified",
            "sample_norms": np.linalg.norm(self.theta_samples_, axis=1),
            "ensemble_spread": float(np.mean(np.sum(
                (self.theta_samples_ - self.theta_samples_.mean(axis=0)) ** 2, axis=1))),
            "theta_samples": self.theta_samples_.copy(),
            "theta_hat": self._estimate.copy(),
            "outer_states": self.outer_states_.copy(),
            "sampler_config": asdict(self.config),
            "sampling_runtime_seconds": time.perf_counter() - started,
            "sampling_forward_calls": sum(result.summary["forward_calls"] for result in results),
            "trajectories": [result.summary for result in results],
            "chain_trace": [entry for result in results for entry in result.trace],
        }

    def score_candidates(self) -> tuple[Array, Array]:
        if not self._available:
            raise ValidationError("the non-repeating query set has been exhausted")
        target = self._target(self.context.decision_problem, self.context.dimension, self.config.beta, (),
                              use_batch=self.config.sampler == "gaussian_gibbs")
        indices = np.asarray(self._available, dtype=int)
        if self.config.sampler == "gaussian_gibbs":
            predictions = np.stack([target.batch_forward(theta, self.candidates_[indices])
                                    for theta in self.theta_samples_])
            centered = predictions - predictions.mean(axis=0)
            scores = 2 / (len(predictions) - 1) * np.sum(centered ** 2, axis=(0, 2))
        else:
            scores = np.asarray([
                disagreement_score(np.vstack([
                    target.forward(theta, self.candidates_[index]) for theta in self.theta_samples_
                ])) for index in indices
            ])
        self.forward_calls_ += target.forward_calls
        self.forward_seconds_ += target.forward_seconds
        return indices, scores

    def propose(self, history) -> ActiveAction:
        del history  # observe() owns a copied history of public observations.
        if not self._available:
            raise ValidationError("the non-repeating query set has been exhausted")
        if self.config.query_policy == "uniform":
            indices = np.asarray(self._available, dtype=int)
            scores = None
            best = int(self._query_rng.integers(len(indices)))
        else:
            indices, scores = self.score_candidates()
            ties = np.flatnonzero(scores == np.max(scores))
            best = int(ties[0] if self.config.query_tie_breaking == "first"
                       else self._query_rng.choice(ties))
        index = int(indices[best])
        diagnostics = {
            **self.diagnostics(),
            "query_rule": ("uniform-candidate" if scores is None else "maximum-optimizer-disagreement"),
            "query_tie_breaking": self.config.query_tie_breaking,
            "candidate_indices": indices.copy(),
            "candidate_scores": None if scores is None else scores.copy(),
            "candidate_index": index, "selected_score": None if scores is None else float(scores[best]),
            "candidate_score_ties": None if scores is None else int(np.sum(scores == scores[best])),
            "candidate_forward_calls": 0 if scores is None else len(indices) * self.config.theta_samples,
        }
        # Large chain traces are recorded once, in update diagnostics (and at t=0).
        if self.observations_:
            diagnostics.pop("chain_trace", None)
        if not self._allow_repeats:
            # Treat duplicate candidate rows as the same query, as the environment does.
            chosen = self.candidates_[index]
            self._available = [i for i in self._available
                               if np.linalg.norm(self.candidates_[i] - chosen) > 1e-7]
        return ActiveAction(self.candidates_[index].copy(), self.current_estimate(), diagnostics=diagnostics)

    def observe(self, observation: AlgorithmObservation) -> None:
        # Reject unsupported feedback explicitly; never substitute hidden X or impute Y.
        validator = self._target(self.context.decision_problem, self.context.dimension, self.config.beta, ())
        validator.append(observation)
        if not self.context.decision_problem.contains(observation.observed_decision):
            raise ValidationError("nested Langevin requires a feasible observed decision Y")
        query, observed = validator.data[0]
        self.observations_.append(AlgorithmObservation(observation.step, query, observed))
        self._resample()

    def current_estimate(self) -> Array:
        return self._estimate.copy()

    def diagnostics(self) -> dict:
        return copy.deepcopy({
            **self._sampling_diagnostics,
            "forward_calls_total": self.forward_calls_,
            "forward_seconds_total": self.forward_seconds_,
        })


def create_nested_langevin_algorithm() -> NestedLangevinActiveAlgorithm:
    return NestedLangevinActiveAlgorithm()
