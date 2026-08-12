"""Client for the running GUI's loopback control endpoint."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 1024 * 1024
DEFAULT_REQUEST_TIMEOUT = 24 * 60 * 60


class LocalControlError(RuntimeError):
    """Raised when the local GUI cannot complete a control request."""


def default_endpoint_path() -> Path:
    override = os.environ.get("SHOOKLINK_CONTROL_ENDPOINT")
    if override:
        return Path(override).expanduser()

    from PySide6.QtCore import QCoreApplication, QStandardPaths

    QCoreApplication.setApplicationName("ShookLink")
    QCoreApplication.setOrganizationName("ShookLink")
    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    if location:
        return Path(location) / "control.json"
    return Path.home() / ".shooklink" / "control.json"


class LocalControlClient:
    def __init__(
        self,
        endpoint_path: str | Path | None = None,
        *,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self.endpoint_path = Path(endpoint_path or default_endpoint_path())
        self.timeout = timeout

    def request(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        endpoint = self._load_endpoint()
        request = {
            "token": endpoint["token"],
            "command": command,
            "payload": payload or {},
        }
        try:
            with socket.create_connection(
                (endpoint["host"], endpoint["port"]),
                timeout=self.timeout,
            ) as connection:
                connection.settimeout(self.timeout)
                connection.sendall(
                    json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n"
                )
                response_line = connection.makefile("rb").readline(
                    MAX_RESPONSE_BYTES + 1
                )
        except OSError as error:
            raise LocalControlError(
                "ShookLink is not running or its local control endpoint is unavailable"
            ) from error
        if not response_line:
            raise LocalControlError("ShookLink closed the local control request")
        if len(response_line) > MAX_RESPONSE_BYTES:
            raise LocalControlError("local control response is too large")
        try:
            response = json.loads(response_line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise LocalControlError("ShookLink returned an invalid control response") from error
        if not isinstance(response, dict) or type(response.get("ok")) is not bool:
            raise LocalControlError("ShookLink returned an invalid control response")
        if not response["ok"]:
            message = response.get("error")
            raise LocalControlError(
                message if isinstance(message, str) and message else "local control failed"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise LocalControlError("ShookLink returned an invalid control result")
        return result

    def _load_endpoint(self) -> dict[str, Any]:
        try:
            endpoint = json.loads(self.endpoint_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise LocalControlError("ShookLink is not running") from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LocalControlError("ShookLink control endpoint is invalid") from error
        if (
            not isinstance(endpoint, dict)
            or endpoint.get("version") != 1
            or endpoint.get("host") != "127.0.0.1"
            or type(endpoint.get("port")) is not int
            or not 1 <= endpoint["port"] <= 65535
            or not isinstance(endpoint.get("token"), str)
            or not endpoint["token"]
        ):
            raise LocalControlError("ShookLink control endpoint is invalid")
        return endpoint
