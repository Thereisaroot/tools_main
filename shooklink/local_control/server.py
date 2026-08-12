"""Loopback server owned by the running ShookLink GUI."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import socketserver
import tempfile
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

MAX_REQUEST_BYTES = 128 * 1024
DEFAULT_CHAT_TIMEOUT = 15.0
DEFAULT_FILE_TIMEOUT = 24 * 60 * 60.0
FILE_TERMINAL_STATES = frozenset(
    {"complete", "failed", "cancelled", "error", "rejected"}
)


class _ThreadingControlServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


class _ControlRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        if len(line) > MAX_REQUEST_BYTES:
            response = {"ok": False, "error": "control request is too large"}
        else:
            response = self.server.owner.handle_request_bytes(line)
        self.wfile.write(
            json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            + b"\n"
        )


class LocalControlServer:
    def __init__(
        self,
        core,
        endpoint_path: str | Path,
        *,
        chat_timeout: float = DEFAULT_CHAT_TIMEOUT,
        file_timeout: float = DEFAULT_FILE_TIMEOUT,
    ) -> None:
        self._core = core
        self.endpoint_path = Path(endpoint_path)
        self._chat_timeout = chat_timeout
        self._file_timeout = file_timeout
        self._token = secrets.token_urlsafe(32)
        self._server = None
        self._thread = None
        self._lock = threading.RLock()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._server is not None

    def start(self) -> None:
        with self._lock:
            if self._server is not None:
                return
            server = _ThreadingControlServer(
                ("127.0.0.1", 0), _ControlRequestHandler
            )
            server.owner = self
            try:
                self._publish_endpoint(server.server_address[1])
                thread = threading.Thread(
                    target=server.serve_forever,
                    name="shooklink-local-control",
                    daemon=True,
                )
                self._server = server
                self._thread = thread
                thread.start()
            except BaseException:
                server.server_close()
                self._remove_owned_endpoint()
                raise

    def close(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
        if server is None:
            return
        self._remove_owned_endpoint()
        server.shutdown()
        server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def handle_request_bytes(self, raw: bytes) -> dict[str, Any]:
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return {"ok": False, "error": "invalid control request"}
        if not isinstance(request, dict):
            return {"ok": False, "error": "invalid control request"}
        token = request.get("token")
        if not isinstance(token, str) or not hmac.compare_digest(token, self._token):
            return {"ok": False, "error": "authentication failed"}
        command = request.get("command")
        payload = request.get("payload")
        if not isinstance(command, str) or not isinstance(payload, dict):
            return {"ok": False, "error": "invalid control request"}
        try:
            result = self._dispatch(command, payload)
        except Exception as error:
            message = str(error).strip() or error.__class__.__name__
            return {"ok": False, "error": message}
        return {"ok": True, "result": result}

    def _dispatch(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "status":
            if payload:
                raise ValueError("status payload must be empty")
            snapshot = self._core.snapshot
            state = getattr(snapshot.state, "value", str(snapshot.state))
            return {
                "connection_id": snapshot.connection_id,
                "state": state,
                "peer_id": snapshot.peer_id,
                "trusted": bool(
                    snapshot.local_approved and snapshot.remote_approved
                ),
                "features": sorted(snapshot.features),
            }
        if command in {"send-plain", "send-secure"}:
            return self._send_text(command, payload)
        if command == "send-file":
            return self._send_file(payload)
        raise ValueError(f"unknown control command: {command}")

    def _send_text(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) != {"text"} or not isinstance(payload["text"], str):
            raise ValueError("text payload must contain one string field")
        completed = threading.Event()
        outcome: dict[str, Any] = {}

        def delivered() -> None:
            outcome["delivered"] = True
            completed.set()

        def failed(reason: str) -> None:
            outcome["error"] = reason
            completed.set()

        kind = "plain" if command == "send-plain" else "secure"
        sender = (
            self._core.chat.send_plain
            if kind == "plain"
            else self._core.chat.send_secure
        )
        sender(payload["text"], on_delivered=delivered, on_failed=failed)
        if not completed.wait(self._chat_timeout):
            raise TimeoutError("text delivery timed out")
        if "error" in outcome:
            raise RuntimeError(str(outcome["error"]))
        return {"kind": kind, "delivered": True}

    def _send_file(self, payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) != {"path"} or not isinstance(payload["path"], str):
            raise ValueError("file payload must contain one path string")
        path = Path(payload["path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"file does not exist: {path}")
        progress_condition = threading.Condition()
        terminal: dict[str, Any] = {}

        def progress_changed(progress) -> None:
            if (
                progress.direction == "outgoing"
                and progress.state in FILE_TERMINAL_STATES
            ):
                with progress_condition:
                    terminal[progress.transfer_id] = progress
                    progress_condition.notify_all()

        self._core.files.add_progress_listener(progress_changed)
        try:
            try:
                transfer_id = self._core.files.send_file(path).result(timeout=30)
            except FutureTimeoutError as error:
                raise TimeoutError("file preparation timed out") from error
            with progress_condition:
                available = progress_condition.wait_for(
                    lambda: transfer_id in terminal,
                    timeout=self._file_timeout,
                )
                if not available:
                    raise TimeoutError("file transfer timed out")
                progress = terminal.get(transfer_id)
            if progress.state != "complete":
                raise RuntimeError(f"file transfer failed: {progress.state}")
            return {
                "transfer_id": transfer_id,
                "name": progress.name,
                "bytes": progress.total,
                "state": progress.state,
            }
        finally:
            self._core.files.remove_progress_listener(progress_changed)

    def _publish_endpoint(self, port: int) -> None:
        self.endpoint_path.parent.mkdir(parents=True, exist_ok=True)
        endpoint = {
            "version": 1,
            "host": "127.0.0.1",
            "port": port,
            "token": self._token,
            "pid": os.getpid(),
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.endpoint_path.parent,
                prefix=f".{self.endpoint_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(endpoint, temporary_file, ensure_ascii=False)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            if os.name == "posix":
                temporary_path.chmod(0o600)
            os.replace(temporary_path, self.endpoint_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _remove_owned_endpoint(self) -> None:
        try:
            endpoint = json.loads(self.endpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        if isinstance(endpoint, dict) and endpoint.get("token") == self._token:
            self.endpoint_path.unlink(missing_ok=True)
