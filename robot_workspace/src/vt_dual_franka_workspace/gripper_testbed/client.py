from __future__ import annotations

from typing import Any

import requests

from vt_dual_franka_shared.models import ControllerState, GripperTestbedTargetCommand


class GripperTestbedControllerClient:
    def __init__(self, host: str, port: int, request_timeout_sec: float = 1.0) -> None:
        self.base_url = f"http://{host}:{port}"
        self.request_timeout_sec = request_timeout_sec
        self._session = requests.Session()

    def health(self) -> dict[str, Any]:
        return self._get_json("/api/v1/health")

    def get_state(self) -> ControllerState:
        return ControllerState.model_validate(self._get_json("/api/v1/state"))

    def get_status(self) -> dict[str, Any]:
        return self._get_json("/api/v1/gripper/status")

    def send_target(self, command: GripperTestbedTargetCommand) -> dict[str, Any]:
        return self._post_json("/api/v1/gripper/target", command.model_dump(mode="json"))

    def open_gripper(self, *, width: float | None = None, velocity: float | None = None, force_limit: float | None = None) -> dict[str, Any]:
        params: list[str] = []
        if width is not None:
            params.append(f"width={width}")
        if velocity is not None:
            params.append(f"velocity={velocity}")
        if force_limit is not None:
            params.append(f"force_limit={force_limit}")
        suffix = f"?{'&'.join(params)}" if params else ""
        return self._post_json(f"/api/v1/gripper/open{suffix}", {})

    def stop_gripper(self) -> dict[str, Any]:
        return self._post_json("/api/v1/gripper/stop", {})

    def _get_json(self, path: str) -> dict[str, Any]:
        response = self._session.get(f"{self.base_url}{path}", timeout=self.request_timeout_sec)
        response.raise_for_status()
        return response.json()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._session.post(f"{self.base_url}{path}", json=payload, timeout=self.request_timeout_sec)
        response.raise_for_status()
        return response.json()
