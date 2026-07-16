from __future__ import annotations

import threading
import time
from typing import Any

import requests

from vt_dual_franka_shared.models import ArmId, ControllerState, GripperGraspCommand, GripperWidthCommand, ResetCommand, TcpTargetCommand


class ControllerClientError(RuntimeError):
    pass


class ControllerClient:
    def __init__(self, host: str, port: int, request_timeout_sec: float = 1.0, arm_id: ArmId | None = None) -> None:
        self.base_url = f"http://{host}:{port}"
        self.request_timeout_sec = request_timeout_sec
        self.arm_id = arm_id
        self._local = threading.local()

    def health(self) -> dict[str, Any]:
        return self._get_json("/api/v1/health")

    def get_state(self) -> ControllerState:
        data = self._get_json("/api/v1/state")
        return ControllerState.model_validate(data)

    def queue_tcp(
        self,
        target_tcp: list[float],
        source: str = "workspace",
        target_duration_sec: float | None = None,
        *,
        command_id: str | None = None,
        target_monotonic_time: float | None = None,
    ) -> None:
        command = TcpTargetCommand(
            arm_id=self.arm_id,
            command_id=command_id,
            target_tcp=target_tcp,
            target_duration_sec=target_duration_sec,
            target_monotonic_time=target_monotonic_time,
            source=source,
        )
        self._post_json("/api/v1/commands/tcp", command.model_dump(mode="json"))

    def move_gripper(
        self,
        width: float,
        velocity: float,
        force_limit: float,
        source: str = "workspace",
        blocking: bool = False,
    ) -> None:
        command = GripperWidthCommand(
            width=width,
            velocity=velocity,
            force_limit=force_limit,
            source=source,
            blocking=blocking,
        )
        timeout_sec = max(self.request_timeout_sec, 10.0) if blocking else None
        self._post_json("/api/v1/commands/gripper/width", command.model_dump(mode="json"), timeout_sec=timeout_sec)

    def grasp_gripper(
        self,
        velocity: float,
        force_limit: float,
        source: str = "workspace",
        blocking: bool = False,
    ) -> None:
        command = GripperGraspCommand(velocity=velocity, force_limit=force_limit, source=source, blocking=blocking)
        timeout_sec = max(self.request_timeout_sec, 10.0) if blocking else None
        self._post_json("/api/v1/commands/gripper/grasp", command.model_dump(mode="json"), timeout_sec=timeout_sec)

    def stop_gripper(self) -> None:
        self._post_json("/api/v1/commands/gripper/stop", {})

    def assert_identity(self) -> None:
        if self.arm_id is None:
            return
        health = self.health()
        actual = health.get("arm_id")
        if actual != self.arm_id:
            raise ControllerClientError(f"Expected controller arm_id={self.arm_id!r}, got {actual!r}")

    def home(self) -> None:
        self._post_json("/api/v1/actions/home", {}, timeout_sec=max(self.request_timeout_sec, 30.0))

    def ready(self) -> None:
        try:
            self._post_json("/api/v1/actions/ready", {}, timeout_sec=max(self.request_timeout_sec, 30.0))
        except ControllerClientError as exc:
            if "404" in str(exc):
                raise ControllerClientError(
                    "Controller API does not support /api/v1/actions/ready. "
                    "Restart vt-dual-franka-controller on the controller PC after updating it, or use --go-home / omit --go-ready."
                ) from exc
            raise

    def reset(self, command: ResetCommand) -> dict[str, Any]:
        try:
            return self._post_json(
                "/api/v1/actions/reset",
                command.model_dump(mode="json"),
                timeout_sec=max(self.request_timeout_sec, 30.0),
            )
        except ControllerClientError as exc:
            if "404" in str(exc):
                raise ControllerClientError(
                    "Controller API does not support /api/v1/actions/reset. "
                    "Restart vt-dual-franka-controller on the controller PC after updating it."
                ) from exc
            raise

    def _get_json(self, path: str) -> dict[str, Any]:
        last_exc: requests.RequestException | None = None
        for attempt in range(2):
            try:
                response = self._session.get(f"{self.base_url}{path}", timeout=self.request_timeout_sec)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt > 0:
                    break
                self._reset_thread_session()
                time.sleep(0.02)
            except ValueError as exc:
                raise ControllerClientError(f"GET {self.base_url}{path} returned invalid JSON") from exc
        raise ControllerClientError(f"GET {self.base_url}{path} failed: {last_exc}") from last_exc

    def _post_json(self, path: str, payload: dict[str, Any], timeout_sec: float | None = None) -> dict[str, Any]:
        timeout = self.request_timeout_sec if timeout_sec is None else timeout_sec
        last_exc: requests.RequestException | None = None
        last_detail: str | None = None
        for attempt in range(2):
            try:
                response = self._session.post(
                    f"{self.base_url}{path}",
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_exc = exc
                response = getattr(exc, "response", None)
                if response is not None:
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = None
                    if isinstance(payload, dict) and payload.get("detail"):
                        last_detail = str(payload["detail"])
                if attempt > 0:
                    break
                self._reset_thread_session()
                time.sleep(0.02)
            except ValueError as exc:
                raise ControllerClientError(f"POST {self.base_url}{path} returned invalid JSON") from exc
        detail_suffix = f" (detail: {last_detail})" if last_detail else ""
        raise ControllerClientError(f"POST {self.base_url}{path} failed: {last_exc}{detail_suffix}") from last_exc

    @property
    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def _reset_thread_session(self) -> None:
        try:
            self._session.close()
        finally:
            self._local.session = requests.Session()
