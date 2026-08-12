from pathlib import Path

from shooklink.local_control import cli


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def request(self, command, payload=None):
        self.calls.append((command, payload or {}))
        if self.error is not None:
            raise self.error
        return self.result


def test_cli_send_file_prints_peer_confirmed_completion(tmp_path, capsys):
    source = tmp_path / "file.txt"
    source.write_text("hello", encoding="utf-8")
    client = FakeClient(
        {
            "transfer_id": "a" * 32,
            "name": "file.txt",
            "bytes": 5,
            "state": "complete",
        }
    )

    exit_code = cli.main(
        ["send-file", str(source)],
        client_factory=lambda _endpoint: client,
    )

    assert exit_code == 0
    assert client.calls == [("send-file", {"path": str(source.resolve())})]
    assert capsys.readouterr().out.strip() == "sent file: file.txt (5 bytes)"


def test_cli_send_text_selects_plain_or_secure(capsys):
    client = FakeClient({"kind": "secure", "delivered": True})

    exit_code = cli.main(
        ["send-secure", "secret text"],
        client_factory=lambda _endpoint: client,
    )

    assert exit_code == 0
    assert client.calls == [("send-secure", {"text": "secret text"})]
    assert capsys.readouterr().out.strip() == "sent secure text"


def test_cli_returns_nonzero_for_control_error(capsys):
    client = FakeClient(error=RuntimeError("not connected"))

    exit_code = cli.main(["status"], client_factory=lambda _endpoint: client)

    assert exit_code == 1
    assert capsys.readouterr().err.strip() == "shooklinkctl: not connected"
