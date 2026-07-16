from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    @abstractmethod
    def predict(self, observation_window: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a chunk of executable action dictionaries."""

    def ensure_loaded(self) -> None:
        return None

    def reset(self) -> None:
        return None

    def start_episode(self, observation_window: list[dict[str, Any]]) -> None:
        del observation_window
        return None

    def observe_executed_actions(self, actions: list[dict[str, Any]]) -> None:
        del actions
        return None

    def close(self) -> None:
        return None
