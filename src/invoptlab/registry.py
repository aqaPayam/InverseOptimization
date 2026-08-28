from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, Any] = {}

    def register(self, name: str, value: Any | None = None):
        def decorator(item: Any):
            if name in self._entries:
                raise KeyError(f"{self.kind} {name!r} is already registered")
            self._entries[name] = item
            return item

        return decorator(value) if value is not None else decorator

    def get(self, name: str) -> Any:
        if name not in self._entries:
            raise KeyError(f"Unknown {self.kind} {name!r}; available: {sorted(self._entries)}")
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)


ESTIMATORS = Registry("estimator")
LOSSES = Registry("loss")
NOISE_MODELS = Registry("noise model")
PROBLEMS = Registry("problem")
METRICS = Registry("metric")

