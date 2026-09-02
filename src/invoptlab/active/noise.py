from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from ..exceptions import CapabilityError, ValidationError
from .config import (
    ObservationNoiseConfig,
    ObservationNoiseKind,
    ParameterNoiseConfig,
    ParameterNoiseKind,
)
from .decision_spaces import DecisionSpace, IndependentBinaryDecisionSpace


Array = np.ndarray


@dataclass(slots=True)
class BehavioralNoiseCalibration:
    target_change_rate: float
    achieved_change_rate: float
    effective_strength: float
    trials: int
    channel: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_parameter(value: Array, fallback: Array) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    norm = np.linalg.norm(result)
    if norm <= 1e-15:
        return np.asarray(fallback, dtype=float).copy()
    return result / norm


def query_dependent_scale(
    query: Array,
    profile: str,
    minimum: float,
    maximum: float,
) -> float:
    value = np.asarray(query, dtype=float).reshape(-1)
    if profile == "first_coordinate":
        score = 0.5 * (float(value[0]) + 1.0)
    elif profile == "absolute_first":
        score = abs(float(value[0]))
    elif profile == "sparsity":
        score = float(np.count_nonzero(np.abs(value) > 1e-12) / value.size)
    elif profile == "norm":
        score = min(1.0, float(np.linalg.norm(value)))
    else:  # pragma: no cover - validated by config
        raise ValidationError(f"unknown query profile: {profile}")
    return float(minimum + np.clip(score, 0.0, 1.0) * (maximum - minimum))


class ParameterNoise(ABC):
    kind: ParameterNoiseKind
    calibration: BehavioralNoiseCalibration | None = None

    def reset(self, theta_true: Array, rng: np.random.Generator) -> None:
        del theta_true, rng

    @abstractmethod
    def apply(
        self,
        theta_true: Array,
        query: Array,
        step: int,
        rng: np.random.Generator,
    ) -> tuple[Array, Mapping[str, Any]]:
        ...


class NoParameterNoise(ParameterNoise):
    kind = ParameterNoiseKind.NONE

    def apply(self, theta_true, query, step, rng):
        del query, step, rng
        return np.asarray(theta_true, dtype=float).copy(), {"kind": self.kind.value}


class IsotropicParameterNoise(ParameterNoise):
    kind = ParameterNoiseKind.ISOTROPIC

    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def apply(self, theta_true, query, step, rng):
        del query, step
        perturbation = rng.normal(scale=self.sigma, size=np.asarray(theta_true).size)
        value = normalize_parameter(np.asarray(theta_true) + perturbation, np.asarray(theta_true))
        metadata = {
            "kind": self.kind.value,
            "sigma": self.sigma,
            "perturbation": perturbation,
        }
        if self.calibration is not None:
            metadata["behavioral_calibration"] = self.calibration.to_dict()
        return value, metadata


class AnisotropicParameterNoise(ParameterNoise):
    kind = ParameterNoiseKind.ANISOTROPIC

    def __init__(self, covariance: Array):
        covariance = np.asarray(covariance, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValidationError("parameter covariance must be square")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValidationError("parameter covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if np.min(eigenvalues) < -1e-10:
            raise ValidationError("parameter covariance must be positive semidefinite")
        self.covariance = covariance

    def apply(self, theta_true, query, step, rng):
        del query, step
        perturbation = rng.multivariate_normal(np.zeros(self.covariance.shape[0]), self.covariance)
        value = normalize_parameter(np.asarray(theta_true) + perturbation, np.asarray(theta_true))
        return value, {
            "kind": self.kind.value,
            "perturbation": perturbation,
            "covariance": self.covariance,
        }


class QueryDependentParameterNoise(ParameterNoise):
    kind = ParameterNoiseKind.QUERY_DEPENDENT

    def __init__(self, profile: str, minimum: float, maximum: float, covariance: Array | None = None):
        self.profile = profile
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.covariance = None if covariance is None else np.asarray(covariance, dtype=float)

    def apply(self, theta_true, query, step, rng):
        del step
        scale = query_dependent_scale(query, self.profile, self.minimum, self.maximum)
        dimension = np.asarray(theta_true).size
        if self.covariance is None:
            perturbation = rng.normal(scale=scale, size=dimension)
        else:
            if self.covariance.shape != (dimension, dimension):
                raise ValidationError("query-dependent covariance dimension mismatch")
            perturbation = rng.multivariate_normal(np.zeros(dimension), scale**2 * self.covariance)
        value = normalize_parameter(np.asarray(theta_true) + perturbation, np.asarray(theta_true))
        return value, {
            "kind": self.kind.value,
            "profile": self.profile,
            "scale": scale,
            "perturbation": perturbation,
        }


class PersistentParameterNoise(ParameterNoise):
    kind = ParameterNoiseKind.PERSISTENT

    def __init__(self, sigma: float):
        self.sigma = float(sigma)
        self.perturbation: Array | None = None
        self.shifted_parameter: Array | None = None

    def reset(self, theta_true, rng) -> None:
        self.perturbation = rng.normal(scale=self.sigma, size=np.asarray(theta_true).size)
        self.shifted_parameter = normalize_parameter(
            np.asarray(theta_true) + self.perturbation,
            np.asarray(theta_true),
        )

    def apply(self, theta_true, query, step, rng):
        del query, step, rng
        if self.shifted_parameter is None or self.perturbation is None:
            raise RuntimeError("persistent parameter noise must be reset before use")
        return self.shifted_parameter.copy(), {
            "kind": self.kind.value,
            "sigma": self.sigma,
            "persistent_perturbation": self.perturbation,
        }


def make_parameter_noise(config: ParameterNoiseConfig, dimension: int) -> ParameterNoise:
    if config.kind == ParameterNoiseKind.NONE:
        return NoParameterNoise()
    if config.kind == ParameterNoiseKind.ISOTROPIC:
        return IsotropicParameterNoise(config.sigma)
    if config.kind == ParameterNoiseKind.ANISOTROPIC:
        covariance = (
            np.asarray(config.covariance, dtype=float)
            if config.covariance is not None
            else np.diag(np.linspace(0.5, 1.5, dimension) * config.sigma**2)
        )
        if covariance.shape != (dimension, dimension):
            raise ValidationError("anisotropic covariance must match dimension")
        return AnisotropicParameterNoise(covariance)
    if config.kind == ParameterNoiseKind.QUERY_DEPENDENT:
        covariance = None if config.covariance is None else np.asarray(config.covariance, dtype=float)
        return QueryDependentParameterNoise(
            config.query_profile,
            config.minimum_scale,
            config.maximum_scale,
            covariance,
        )
    return PersistentParameterNoise(config.sigma)


@dataclass(slots=True)
class ObservationNoiseResult:
    decision: Array
    mask: Array | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ObservationNoise(ABC):
    kind: ObservationNoiseKind
    calibration: BehavioralNoiseCalibration | None = None

    @abstractmethod
    def apply(
        self,
        true_decision: Array,
        query: Array,
        decision_space: DecisionSpace,
        step: int,
        rng: np.random.Generator,
    ) -> ObservationNoiseResult:
        ...


class CleanObservationNoise(ObservationNoise):
    kind = ObservationNoiseKind.CLEAN

    def apply(self, true_decision, query, decision_space, step, rng):
        del query, decision_space, step, rng
        return ObservationNoiseResult(
            np.asarray(true_decision).copy(),
            metadata={"kind": self.kind.value},
        )


class LocalObservationNoise(ObservationNoise):
    kind = ObservationNoiseKind.LOCAL

    def __init__(self, sigma: float, distance: str = "euclidean"):
        self.sigma = float(sigma)
        self.distance = distance

    def _apply_with_sigma(self, true_decision, decision_space, rng, sigma) -> ObservationNoiseResult:
        noisy = decision_space.sample_local(
            np.asarray(true_decision),
            sigma,
            rng,
            distance=self.distance,
        )
        metadata = {
            "kind": self.kind.value,
            "sigma": sigma,
            "distance": self.distance,
            "changed": not np.allclose(noisy, true_decision),
        }
        if self.calibration is not None:
            metadata["behavioral_calibration"] = self.calibration.to_dict()
        return ObservationNoiseResult(
            noisy,
            metadata=metadata,
        )

    def apply(self, true_decision, query, decision_space, step, rng):
        del query, step
        return self._apply_with_sigma(true_decision, decision_space, rng, self.sigma)


class OutlierObservationNoise(ObservationNoise):
    kind = ObservationNoiseKind.OUTLIER

    def __init__(self, probability: float):
        self.probability = float(probability)

    def apply(self, true_decision, query, decision_space, step, rng):
        del query, step
        contaminated = bool(rng.random() < self.probability)
        decision = np.asarray(true_decision).copy()
        if contaminated:
            for _ in range(20):
                candidate = decision_space.sample_feasible(rng)
                decision = candidate
                if not np.allclose(candidate, true_decision):
                    break
        metadata = {
            "kind": self.kind.value,
            "probability": self.probability,
            "contaminated": contaminated,
            "changed": not np.allclose(decision, true_decision),
        }
        if self.calibration is not None:
            metadata["behavioral_calibration"] = self.calibration.to_dict()
        return ObservationNoiseResult(
            decision,
            metadata=metadata,
        )


class BiasedObservationNoise(ObservationNoise):
    kind = ObservationNoiseKind.BIASED

    def __init__(
        self,
        sigma: float,
        bias: Array,
        confusion_matrix: Array | None = None,
    ):
        self.sigma = float(sigma)
        self.bias = np.asarray(bias, dtype=float).reshape(-1)
        self.confusion_matrix = None if confusion_matrix is None else np.asarray(confusion_matrix, dtype=float)

    def _from_confusion(self, true_decision, decision_space, rng) -> Array:
        decisions = decision_space.enumerate_decisions()
        matrix = self.confusion_matrix
        if matrix is None or matrix.shape != (len(decisions), len(decisions)):
            raise ValidationError("confusion matrix must match the number of feasible discrete decisions")
        if np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
            raise ValidationError("each confusion-matrix row must be a probability vector")
        matches = [index for index, item in enumerate(decisions) if np.array_equal(item, true_decision)]
        if not matches:
            raise ValidationError("true decision was not found in the confusion-matrix state order")
        index = int(rng.choice(len(decisions), p=matrix[matches[0]]))
        return decisions[index].copy()

    def apply(self, true_decision, query, decision_space, step, rng):
        del query, step
        if decision_space.is_discrete and self.confusion_matrix is not None:
            noisy = self._from_confusion(true_decision, decision_space, rng)
            channel = "confusion_matrix"
        elif isinstance(decision_space, IndependentBinaryDecisionSpace):
            reference = np.asarray(true_decision, dtype=int)
            p01 = min(1.0, self.sigma)
            p10 = min(1.0, 2.0 * self.sigma)
            noisy = reference.copy()
            zero_flips = (reference == 0) & (rng.random(reference.size) < p01)
            one_flips = (reference == 1) & (rng.random(reference.size) < p10)
            noisy[zero_flips], noisy[one_flips] = 1, 0
            channel = "factorized_asymmetric"
        else:
            perturbation = self.bias + rng.normal(scale=self.sigma, size=self.bias.size)
            noisy = decision_space.project(np.asarray(true_decision, dtype=float) + perturbation)
            channel = "biased_projected"
        return ObservationNoiseResult(
            noisy,
            metadata={
                "kind": self.kind.value,
                "sigma": self.sigma,
                "bias": self.bias,
                "channel": channel,
                "changed": not np.allclose(noisy, true_decision),
            },
        )


class QueryDependentObservationNoise(LocalObservationNoise):
    kind = ObservationNoiseKind.QUERY_DEPENDENT

    def __init__(self, profile: str, minimum: float, maximum: float, distance: str):
        super().__init__(0.0, distance)
        self.profile = profile
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def apply(self, true_decision, query, decision_space, step, rng):
        del step
        sigma = query_dependent_scale(query, self.profile, self.minimum, self.maximum)
        result = self._apply_with_sigma(true_decision, decision_space, rng, sigma)
        result.metadata = {
            **result.metadata,
            "kind": self.kind.value,
            "profile": self.profile,
        }
        return result


class PartialObservationNoise(ObservationNoise):
    kind = ObservationNoiseKind.PARTIAL

    def __init__(self, mask_probability: float):
        self.mask_probability = float(mask_probability)

    def apply(self, true_decision, query, decision_space, step, rng):
        del query, decision_space, step
        value = np.asarray(true_decision).copy()
        mask = (rng.random(value.size) >= self.mask_probability).astype(int)
        observed = mask * value
        return ObservationNoiseResult(
            observed,
            mask=mask,
            metadata={
                "kind": self.kind.value,
                "mask_probability": self.mask_probability,
                "observed_fraction": float(mask.mean()),
            },
        )


def make_observation_noise(config: ObservationNoiseConfig, dimension: int) -> ObservationNoise:
    if config.kind == ObservationNoiseKind.CLEAN:
        return CleanObservationNoise()
    if config.kind == ObservationNoiseKind.LOCAL:
        return LocalObservationNoise(config.sigma, config.distance)
    if config.kind == ObservationNoiseKind.OUTLIER:
        return OutlierObservationNoise(config.outlier_probability)
    if config.kind == ObservationNoiseKind.BIASED:
        if config.bias is None:
            bias = np.zeros(dimension)
            bias[0] = config.sigma
        else:
            bias = np.asarray(config.bias, dtype=float).reshape(-1)
        if bias.size != dimension:
            raise ValidationError("observation bias must match dimension")
        matrix = None if config.confusion_matrix is None else np.asarray(config.confusion_matrix, dtype=float)
        return BiasedObservationNoise(config.sigma, bias, matrix)
    if config.kind == ObservationNoiseKind.QUERY_DEPENDENT:
        return QueryDependentObservationNoise(
            config.query_profile,
            config.minimum_scale,
            config.maximum_scale,
            config.distance,
        )
    return PartialObservationNoise(config.mask_probability)


def _closest_behavioral_strength(
    strengths: Array,
    estimate_rate,
    target: float,
) -> tuple[float, float]:
    rates = np.asarray([estimate_rate(float(value)) for value in strengths], dtype=float)
    index = int(np.argmin(np.abs(rates - target)))
    return float(strengths[index]), float(rates[index])


def calibrate_noise_behavior(
    parameter_noise: ParameterNoise,
    observation_noise: ObservationNoise,
    parameter_config: ParameterNoiseConfig,
    observation_config: ObservationNoiseConfig,
    theta_true: Array,
    queries: Array,
    decision_space: DecisionSpace,
    *,
    seed: int,
) -> dict[str, Any]:
    """Calibrate supported channels by the probability that behavior changes.

    Calibration is private benchmark setup: it uses the clean parameter only to choose
    an effective channel strength, and exposes neither theta nor calibration samples to
    the evaluated algorithm.
    """

    theta = np.asarray(theta_true, dtype=float)
    candidates = np.asarray(queries, dtype=float)
    decision_rng = np.random.default_rng(np.random.SeedSequence([seed, 77_219]))
    clean = [
        decision_space.min_decision(query * theta, decision_rng, tie_breaking="lexicographic")
        for query in candidates
    ]
    results: dict[str, Any] = {}

    if parameter_config.target_decision_change_rate is not None:
        if not isinstance(parameter_noise, IsotropicParameterNoise):  # validated config guard
            raise ValidationError("parameter behavioral calibration requires isotropic noise")
        trials = parameter_config.calibration_trials

        def parameter_rate(sigma: float) -> float:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 91_337]))
            changed = 0
            for trial in range(trials):
                index = trial % len(candidates)
                perturbed = normalize_parameter(
                    theta + rng.normal(scale=sigma, size=theta.size), theta
                )
                response = decision_space.min_decision(
                    candidates[index] * perturbed,
                    rng,
                    tie_breaking="lexicographic",
                )
                changed += int(not np.allclose(response, clean[index]))
            return changed / trials

        strength, achieved = _closest_behavioral_strength(
            np.geomspace(1e-4, 2.0, 31),
            parameter_rate,
            parameter_config.target_decision_change_rate,
        )
        parameter_noise.sigma = strength
        parameter_noise.calibration = BehavioralNoiseCalibration(
            parameter_config.target_decision_change_rate,
            achieved,
            strength,
            trials,
            "isotropic_parameter",
        )
        results["parameter_noise"] = parameter_noise.calibration.to_dict()

    if observation_config.target_decision_change_rate is not None:
        trials = observation_config.calibration_trials
        if isinstance(observation_noise, LocalObservationNoise):

            def observation_rate(strength: float) -> float:
                rng = np.random.default_rng(np.random.SeedSequence([seed, 54_011]))
                changed = 0
                for trial in range(trials):
                    index = trial % len(candidates)
                    response = decision_space.sample_local(
                        clean[index],
                        strength,
                        rng,
                        distance=observation_noise.distance,
                    )
                    changed += int(not np.allclose(response, clean[index]))
                return changed / trials

            strength, achieved = _closest_behavioral_strength(
                np.geomspace(1e-3, 4.0, 31),
                observation_rate,
                observation_config.target_decision_change_rate,
            )
            observation_noise.sigma = strength
            channel = "local_observation"
        elif isinstance(observation_noise, OutlierObservationNoise):

            def observation_rate(strength: float) -> float:
                rng = np.random.default_rng(np.random.SeedSequence([seed, 54_011]))
                changed = 0
                for trial in range(trials):
                    index = trial % len(candidates)
                    if rng.random() >= strength:
                        continue
                    response = decision_space.sample_feasible(rng)
                    changed += int(not np.allclose(response, clean[index]))
                return changed / trials

            strength, achieved = _closest_behavioral_strength(
                np.linspace(0.0, 1.0, 31),
                observation_rate,
                observation_config.target_decision_change_rate,
            )
            observation_noise.probability = strength
            channel = "outlier_observation"
        else:  # pragma: no cover - validated config guard
            raise ValidationError("unsupported observation channel for behavioral calibration")
        observation_noise.calibration = BehavioralNoiseCalibration(
            observation_config.target_decision_change_rate,
            achieved,
            strength,
            trials,
            channel,
        )
        results["observation_noise"] = observation_noise.calibration.to_dict()

    return results
