import json
import socket
import threading

import pytest

from shooklink.local_control.client import LocalControlClient, LocalControlError


def test_client_sends_endpoint_token_and_payload(tmp_path):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()
    endpoint = tmp_path / "control.json"
    endpoint.write_text(
        json.dumps({"version": 1, "host": host, "port": port, "token": "secret"}),
        encoding="utf-8",
    )
    received = []

    def serve():
        connection, _ = listener.accept()
        with connection:
            received.append(json.loads(connection.makefile("rb").readline()))
            connection.sendall(b'{"ok":true,"result":{"value":3}}\n')
        listener.close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        result = LocalControlClient(endpoint).request("example", {"item": 2})
    finally:
        thread.join(timeout=2)

    assert received == [
        {
            "token": "secret",
            "command": "example",
            "payload": {"item": 2},
        }
    ]
    assert result == {"value": 3}


def test_client_reports_missing_gui_endpoint(tmp_path):
    with pytest.raises(LocalControlError, match="ShookLink is not running"):
        LocalControlClient(tmp_path / "missing.json").request("status")
