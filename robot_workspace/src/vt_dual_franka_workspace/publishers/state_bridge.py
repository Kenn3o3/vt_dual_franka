from __future__ import annotations

import logging
import time
from threading import Event, Thread
from vt_dual_franka_shared.timing import precise_sleep
from vt_dual_franka_shared.models import DualArmControllerState

from ..recording.raw_recorder import JsonlStreamRecorder
from ..config import QuestFeedbackSettings
from ..runtime.dual_arm import DualArmCoordinator
from .quest_udp import QuestUdpPublisher

LOGGER = logging.getLogger(__name__)


class DualStateBridge:
    def __init__(
        self,
        coordinator: DualArmCoordinator,
        quest_publisher: QuestUdpPublisher,
        settings: QuestFeedbackSettings,
        recorder: JsonlStreamRecorder | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.quest_publisher = quest_publisher
        self.settings = settings
        self.recorder = recorder
        self._running = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = Thread(target=self._loop, name="state-bridge-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        period = 1.0 / self.settings.state_publish_hz
        while self._running.is_set():
            try:
                state = self._get_state()
                self.quest_publisher.publish_robot_state(state)
                if self.recorder is not None:
                    self.recorder.record_event(
                        {
                            "schema_version": "vt_dual_franka_controller_state_v1",
                            "source_wall_time": max(state.left.wall_time, state.right.wall_time),
                            "source_monotonic_time": max(state.left.monotonic_time, state.right.monotonic_time),
                            "received_wall_time": time.time(),
                            "state_by_arm": {
                                "left": state.left.model_dump(mode="json"),
                                "right": state.right.model_dump(mode="json"),
                            },
                        },
                        event_time=max(state.left.wall_time, state.right.wall_time),
                    )
            except Exception as exc:
                LOGGER.warning("State bridge iteration failed: %s", exc)
            precise_sleep(period)

    def _get_state(self) -> DualArmControllerState:
        return self.coordinator.get_state()
