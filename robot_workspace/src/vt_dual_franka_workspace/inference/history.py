from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any


class ObservationHistory:
    def __init__(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("Observation horizon must be positive")
        self.horizon = int(horizon)
        self._items: deque[dict[str, Any]] = deque(maxlen=self.horizon)

    def initialize_with_padding(self, observation: dict[str, Any]) -> None:
        self._items.clear()
        for _ in range(self.horizon):
            self._items.append(deepcopy(observation))

    def append(self, observation: dict[str, Any]) -> None:
        self._items.append(deepcopy(observation))

    def window(self) -> list[dict[str, Any]]:
        if len(self._items) != self.horizon:
            raise RuntimeError(f"Observation history is not initialized: {len(self._items)}/{self.horizon}")
        return list(self._items)
