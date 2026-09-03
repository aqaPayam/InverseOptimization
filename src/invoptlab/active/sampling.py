"""Target-invariant Gaussian augmentation and random-direction slice updates.

This is an explicitly different integrator from the unadjusted v2 algorithm.
No estimated score is substituted into a purportedly exact outer transition.
Each finite conditional update preserves r(z|u), and u|z is sampled exactly.
Consequently each fixed-tau marginal kernel preserves pi(z). This statement
concerns stationarity, NOT independence or convergence of a short run.
"""

from __future__ import annotations

import time

import numpy as np

from ..exceptions import SolverError, ValidationError
from .langevin import InverseLossTarget, NestedLangevinConfig, TrajectoryResult, _vector


class GaussianGibbsSampler:
    def __init__(self, target: InverseLossTarget, config: NestedLangevinConfig,
                 rng: np.random.Generator, *, round_index: int = 0, trajectory_id: int = 0):
        self.target, self.config, self.rng = target, config, rng
        self.round_index, self.trajectory_id = round_index, trajectory_id
        self.trace: list[dict] = []
        self.slice_updates = self.density_evaluations = self.shrink_count = 0
        self.radial_updates = 0
        self.squared_jump_sum = 0.0

    def _inside(self, z: np.ndarray) -> bool:
        if not np.all(np.isfinite(z)):
            return False
        if self.config.parameter_domain == "box":
            return bool(np.all(np.abs(z) <= self.config.bound))
        return bool(np.linalg.norm(z) <= self.config.bound)

    def _uniform_support(self) -> np.ndarray:
        d, b = self.target.dimension, self.config.bound
        if self.config.parameter_domain == "box":
            return self.rng.uniform(-b, b, size=d)
        direction = self.rng.normal(size=d)
        while np.linalg.norm(direction) == 0:
            direction = self.rng.normal(size=d)
        return direction / np.linalg.norm(direction) * b * self.rng.random() ** (1.0 / d)

    def log_density(self, z: np.ndarray, u: np.ndarray | None = None,
                    tau: float | None = None) -> float:
        if not self._inside(z):
            return -np.inf
        self.density_evaluations += 1
        result = -self.target.beta * self.target.loss(z)
        if u is not None:
            if tau is None or not np.isfinite(tau) or tau <= 0:
                raise ValidationError("conditional variance must be finite and positive")
            result -= float(np.sum((z - u) ** 2)) / (2 * tau)
        if not np.isfinite(result):
            raise SolverError("non-finite slice log density")
        return float(result)

    def chord(self, z: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
        """Full support interval along a unit direction, with current point at 0."""
        b = self.config.bound
        if self.config.parameter_domain == "box":
            active = direction != 0
            a = (-b - z[active]) / direction[active]
            c = (b - z[active]) / direction[active]
            return float(np.max(np.minimum(a, c))), float(np.min(np.maximum(a, c)))
        projection = float(z @ direction)
        radius = np.sqrt(max(0.0, projection ** 2 + b ** 2 - float(z @ z)))
        return -projection - radius, -projection + radius

    def slice_step(self, z: np.ndarray, *, u: np.ndarray | None = None,
                   tau: float | None = None) -> np.ndarray:
        """One reversible random-line slice update; never clip a proposed state.

        The initial interval is the WHOLE bounded chord, not a state-dependent
        narrow window. Shrinking around the current point preserves the slice.
        A work limit raises an error rather than silently returning a biased draw.
        """
        z = _vector(z, self.target.dimension, "slice state")
        if not self._inside(z):
            raise ValidationError("slice state must lie in the parameter support")
        if u is not None:
            u = _vector(u, self.target.dimension, "Gaussian auxiliary state")
        log_height = self.log_density(z, u, tau) - self.rng.exponential()
        direction = self.rng.normal(size=z.size)
        while np.linalg.norm(direction) == 0:
            direction = self.rng.normal(size=z.size)
        direction /= np.linalg.norm(direction)
        lower, upper = self.chord(z, direction)
        if not lower < upper or not lower <= 0 <= upper:
            raise SolverError("degenerate slice chord; initialize inside the support")
        for attempt in range(self.config.max_slice_shrinks):
            distance = float(self.rng.uniform(lower, upper))
            proposal = z + distance * direction
            if self.log_density(proposal, u, tau) >= log_height:
                self.slice_updates += 1
                self.shrink_count += attempt
                self.squared_jump_sum += float(np.sum((proposal - z) ** 2))
                return proposal
            if distance < 0:
                lower = distance
            else:
                upper = distance
        raise SolverError("slice shrink limit exceeded; sampling result is unavailable")

    def radial_step(self, z: np.ndarray) -> np.ndarray:
        """Update radius conditional on direction under Lebesgue measure.

        The polar Jacobian r**(d-1) is ESSENTIAL. Omitting it would change pi,
        over-weighting the origin. The full interval (0, support radius) is
        used with slice shrinkage, so no numerical step-size tuning is needed.
        """
        z = _vector(z, self.target.dimension, "radial state")
        if not self._inside(z):
            raise ValidationError("radial state must lie in the parameter support")
        radius = float(np.linalg.norm(z))
        if radius == 0:
            # Polar coordinates are undefined at this measure-zero point.
            return self.slice_step(z)
        direction = z / radius
        upper = (self.config.bound if self.config.parameter_domain == "ball"
                 else self.config.bound / np.max(np.abs(direction)))
        lower = 0.0
        power = self.target.dimension - 1
        height = self.log_density(z) + power * np.log(radius) - self.rng.exponential()
        for attempt in range(self.config.max_slice_shrinks):
            proposed_radius = float(self.rng.uniform(lower, upper))
            if proposed_radius <= 0:
                continue
            proposal = proposed_radius * direction
            log_value = self.log_density(proposal) + power * np.log(proposed_radius)
            if log_value >= height:
                self.radial_updates += 1
                self.shrink_count += attempt
                self.squared_jump_sum += float(np.sum((proposal-z)**2))
                return proposal
            if proposed_radius < radius:
                lower = proposed_radius
            else:
                upper = proposed_radius
        raise SolverError("radial slice shrink limit exceeded; sampling result is unavailable")

    def sample(self, warm_start: np.ndarray | None = None) -> TrajectoryResult:
        started = time.perf_counter()
        config = self.config
        if not self.target.data:
            z = self._uniform_support()
        elif warm_start is not None and config.warm_start:
            z = _vector(warm_start, self.target.dimension, "warm start").copy()
            if not self._inside(z):
                raise ValidationError("warm start lies outside the parameter support")
        else:
            z = self._uniform_support()
        max_outer_norm = 0.0
        # With no data, pi is uniform: draw it exactly, independently of previous z.
        if not self.target.data:
            u = z + np.sqrt(config.tau_schedule[-1]) * self.rng.normal(size=z.size)
            max_outer_norm = float(np.linalg.norm(u))
        else:
            for level, tau in enumerate(config.tau_schedule):
                for sweep in range(config.gibbs_sweeps):
                    # Exact Gaussian conditional, unbounded u; tau is a variance.
                    u = z + np.sqrt(tau) * self.rng.normal(size=z.size)
                    max_outer_norm = max(max_outer_norm, float(np.linalg.norm(u)))
                    # Continue from the CURRENT z. Resetting to project(u) would
                    # destroy the conditional-kernel invariance argument.
                    for _ in range(config.conditional_slice_steps):
                        z = self.slice_step(z, u=u, tau=tau)
                    if config.record_chain_trace:
                        self.trace.append({"round": self.round_index,
                            "trajectory": self.trajectory_id, "chain": "gaussian-gibbs",
                            "level": level, "tau": tau, "sweep": sweep + 1,
                            "z": z.tolist(), "u_norm": float(np.linalg.norm(u))})
            # Broad moves for the ORIGINAL pi, with no Gaussian localization.
            # These are target-invariant refresh moves, not an exact finite-time cure.
            for step in range(config.target_slice_steps):
                z = self.slice_step(z)
                if config.radial_refresh:
                    z = self.radial_step(z)
                if config.record_chain_trace:
                    self.trace.append({"round": self.round_index,
                        "trajectory": self.trajectory_id, "chain": "target-slice",
                        "step": step + 1, "z": z.tolist()})
        return TrajectoryResult(z.copy(), u.copy(), self.trace, {
            "trajectory": self.trajectory_id, "sampler": "gaussian_gibbs",
            "forward_calls": self.target.forward_calls,
            "forward_seconds": self.target.forward_seconds,
            "target_score_calls": self.target.score_calls,
            "density_evaluations": self.density_evaluations,
            "slice_updates": self.slice_updates,
            "radial_updates": self.radial_updates,
            "slice_shrinks": self.shrink_count,
            "mean_squared_jump": self.squared_jump_sum / max(1, self.slice_updates + self.radial_updates),
            "exact_uniform_initialization": not self.target.data,
            "finite_chain_convergence_certified": False,
            "max_outer_norm": max_outer_norm,
            "runtime_seconds": time.perf_counter() - started,
        })
