import json
import os
import socket
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from shooklink.local_control.client import LocalControlClient, LocalControlError
from shooklink.local_control.server import LocalControlServer


@dataclass
class FakeSnapshot:
    connection_id: int = 7
    state: object = None
    peer_id: str | None = "peer-1"
    fingerprint: str | None = "fingerprint"
    local_approved: bool = True
    remote_approved: bool = True
    features: frozenset[str] = frozenset({"chat", "files", "shell"})
    error: str | None = None

    def __post_init__(self):
        if self.state is None:
            self.state = type("State", (), {"value": "ready"})()


class FakeChat:
    def __init__(self):
        self.calls = []
        self.failure = None

    def _send(self, kind, text, *, on_delivered, on_failed, **_kwargs):
        self.calls.append((kind, text))
        if self.failure is None:
            on_delivered()
        else:
            on_failed(self.failure)

    def send_plain(self, text, **callbacks):
        self._send("plain", text, **callbacks)

    def send_secure(self, text, **callbacks):
        self._send("secure", text, **callbacks)


@dataclass
class FakeProgress:
    transfer_id: str
    name: str
    direction: str
    transferred: int
    total: int
    state: str
    path: Path | None = None


class ImmediateFuture:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        return self.value


class FakeFiles:
    def __init__(self):
        self.listeners = []
        self.paths = []
        self.terminal_state = "complete"

    def add_progress_listener(self, listener):
        self.listeners.append(listener)

    def remove_progress_listener(self, listener):
        self.listeners.remove(listener)

    def send_file(self, path):
        path = Path(path)
        self.paths.append(path)
        transfer_id = "a" * 32
        progress = FakeProgress(
            transfer_id,
            path.name,
            "outgoing",
            path.stat().st_size,
            path.stat().st_size,
            self.terminal_state,
            path,
        )
        for listener in tuple(self.listeners):
            listener(progress)
        return ImmediateFuture(transfer_id)


class FakeCore:
    def __init__(self):
        self.snapshot = FakeSnapshot()
        self.chat = FakeChat()
        self.files = FakeFiles()


@pytest.fixture
def running_server(tmp_path):
    core = FakeCore()
    endpoint_path = tmp_path / "control.json"
    server = LocalControlServer(core, endpoint_path)
    server.start()
    try:
        yield core, endpoint_path, LocalControlClient(endpoint_path)
    finally:
        server.close()


def test_status_reports_live_core_snapshot(running_server):
    _, _, client = running_server

    result = client.request("status")

    assert result == {
        "connection_id": 7,
        "state": "ready",
        "peer_id": "peer-1",
        "trusted": True,
        "features": ["chat", "files", "shell"],
    }


@pytest.mark.parametrize(
    ("command", "kind"),
    [("send-plain", "plain"), ("send-secure", "secure")],
)
def test_chat_commands_wait_for_delivery(running_server, command, kind):
    core, _, client = running_server

    result = client.request(command, {"text": "한글 message"})

    assert core.chat.calls == [(kind, "한글 message")]
    assert result == {"kind": kind, "delivered": True}


def test_chat_delivery_failure_is_returned_to_client(running_server):
    core, _, client = running_server
    core.chat.failure = "Delivery failed"

    with pytest.raises(LocalControlError, match="Delivery failed"):
        client.request("send-plain", {"text": "lost"})


def test_file_command_waits_for_terminal_completion(running_server, tmp_path):
    core, _, client = running_server
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")

    result = client.request("send-file", {"path": str(source)})

    assert core.files.paths == [source]
    assert result == {
        "transfer_id": "a" * 32,
        "name": "payload.bin",
        "bytes": 7,
        "state": "complete",
    }


def test_file_terminal_failure_is_returned_to_client(running_server, tmp_path):
    core, _, client = running_server
    core.files.terminal_state = "failed"
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")

    with pytest.raises(LocalControlError, match="file transfer failed"):
        client.request("send-file", {"path": str(source)})


def test_wrong_endpoint_token_is_rejected(running_server):
    _, endpoint_path, _ = running_server
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    with socket.create_connection((endpoint["host"], endpoint["port"])) as sock:
        request = {"token": "wrong", "command": "status", "payload": {}}
        sock.sendall(json.dumps(request).encode() + b"\n")
        response = json.loads(sock.makefile("rb").readline())

    assert response == {"ok": False, "error": "authentication failed"}


def test_endpoint_is_private_and_removed_on_close(tmp_path):
    endpoint_path = tmp_path / "control.json"
    server = LocalControlServer(FakeCore(), endpoint_path)

    server.start()

    assert endpoint_path.exists()
    if os.name == "posix":
        assert endpoint_path.stat().st_mode & 0o777 == 0o600
    server.close()
    assert not endpoint_path.exists()
