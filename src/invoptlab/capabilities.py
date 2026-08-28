from __future__ import annotations

from enum import Enum
from typing import Iterable

from .exceptions import CapabilityError


class Capability(str, Enum):
    LINEAR_IN_THETA = "linear_in_theta"
    DIFFERENTIABLE_IN_THETA = "differentiable_in_theta"
    DIFFERENTIABLE_IN_X = "differentiable_in_x"
    FINITE_FEASIBLE_SET = "finite_feasible_set"
    CONVEX_FORWARD_PROBLEM = "convex_forward_problem"
    MIXED_INTEGER_FORWARD_PROBLEM = "mixed_integer_forward_problem"
    SUPPORTS_ENUMERATION = "supports_enumeration"
    SUPPORTS_SEPARATION = "supports_separation"
    SUPPORTS_KKT = "supports_kkt"
    SUPPORTS_DUALITY = "supports_duality"
    SUPPORTS_WARM_START = "supports_warm_start"


def require_capabilities(
    available: Iterable[Capability], required: Iterable[Capability], owner: str
) -> None:
    available_set = set(available)
    missing = set(required) - available_set
    if missing:
        names = ", ".join(sorted(item.value for item in missing))
        raise CapabilityError(f"{owner} requires unavailable capabilities: {names}")

