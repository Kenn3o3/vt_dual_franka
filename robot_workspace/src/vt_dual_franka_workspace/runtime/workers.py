from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

LOGGER = logging.getLogger(__name__)


@dataclass
class ThreadWorker:
    name: str
    thread: threading.Thread
    stop_event: threading.Event
    required: bool
    error: Exception | None = None

    def is_alive(self) -> bool:
        return self.thread.is_alive()


def start_thread_worker(
    workers: dict[str, ThreadWorker],
    name: str,
    target: Callable[[threading.Event], None],
    *,
    required: bool,
    startup_delay_sec: float = 0.2,
) -> ThreadWorker:
    stop_event = threading.Event()
    worker = ThreadWorker(
        name=name,
        thread=threading.Thread(
            target=lambda: _run_thread_worker(workers, name, target, stop_event),
            name=name,
            daemon=True,
        ),
        stop_event=stop_event,
        required=required,
    )
    workers[name] = worker
    worker.thread.start()
    if startup_delay_sec > 0.0:
        time.sleep(startup_delay_sec)
    return worker


def stop_thread_workers(workers: dict[str, ThreadWorker], *, join_timeout_sec: float = 2.0) -> None:
    for worker in workers.values():
        worker.stop_event.set()
    for worker in workers.values():
        worker.thread.join(timeout=join_timeout_sec)


def _run_thread_worker(
    workers: dict[str, ThreadWorker],
    name: str,
    target: Callable[[threading.Event], None],
    stop_event: threading.Event,
) -> None:
    try:
        target(stop_event)
    except Exception as exc:  # pragma: no cover - thread failure path
        LOGGER.exception("%s worker failed", name)
        workers[name].error = exc
