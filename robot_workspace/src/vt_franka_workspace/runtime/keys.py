from __future__ import annotations

import select
import sys
import termios
import time
import tty


class KeyReader:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_attrs = None

    def __enter__(self) -> "KeyReader":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def read_key(self, timeout_sec: float) -> str | None:
        if self._fd is None:
            time.sleep(timeout_sec)
            return None
        ready, _, _ = select.select([sys.stdin], [], [], timeout_sec)
        if not ready:
            return None
        return sys.stdin.read(1)
