"""Two actual offscreen GUI processes take turns on one simulated serial port."""

import json
import os
from pathlib import Path
import select
import shutil
import socket
import subprocess
import sys
import threading
import uuid

import pytest

from shooklink.live_update.enrollment import PublisherPin, local_signer
from shooklink.live_update.manager import MANAGED_ENV, State
from shooklink.live_update.release import Release
from shooklink.live_update.supervisor import supervise
from shooklink.live_update.wire import Unavailable
from shooklink.local_control import cli
from shooklink.local_control.server import LocalControlServer
from shooklink.protocol.crypto import IdentityStore, TrustStore
from shooklink.protocol.messages import MessageType
from shooklink.settings import AppSettings, SettingsStore
from tests.integration.test_peer_pair import build_core, wait_for


class SinglePort:
    """One permanent controller endpoint; a second concurrent receiver is refused."""
    def __init__(self, drop_candidate):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.port = f"socket://127.0.0.1:{self.listener.getsockname()[1]}"
        self.receiver = None
        self.opens = 0
        self.overlaps = 0
        self.drop_candidate = drop_candidate

    def read(self, size):
        sockets = [self.listener] if self.receiver is None else [self.receiver, self.listener]
        readable, _, _ = select.select(sockets, [], [], 0.03)
        if self.receiver is not None and self.receiver in readable:
            try:
                data = self.receiver.recv(size)
            except ConnectionResetError:
                data = b""
            if data:
                return b"" if self.drop_candidate and self.opens == 2 else data
            self.receiver.close()
            self.receiver = None
        if self.listener in readable:
            sock, _ = self.listener.accept()
            if self.receiver is not None:
                self.overlaps += 1
                sock.close()
            else:
                self.receiver = sock
                self.opens += 1
        return b""

    def write(self, data):
        sock = self.receiver
        if sock is not None and not (self.drop_candidate and self.opens == 2):
            try:
                sock.sendall(data)
            except OSError:
                pass
        return len(data)

    def flush(self):
        pass

    def close(self):
        self.listener.close()
        if self.receiver:
            self.receiver.close()


@pytest.mark.parametrize(("drop_candidate", "button_pairing"), [(False, False), (True, False),
                         (False, True)], ids=["installed", "rollback", "button-installed"])
def test_controller_observes_handoff_on_same_open_link(tmp_path, capsys, monkeypatch,
                                                     drop_candidate, button_pairing):
    source = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "bootstrap"
    checkout.mkdir()
    shutil.copytree(source / "shooklink", checkout / "shooklink",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in (".gitattributes", "README.md", "pyproject.toml", "requirements.txt",
                 "run_serial_text_chat.bat", "run_serial_text_chat.sh"):
        shutil.copyfile(source / name, checkout / name)
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "offline test release"],
                   cwd=checkout, check=True)
    if button_pairing:
        monkeypatch.setattr(cli, "application_data_dir", lambda: tmp_path / "mac-data")
        public_key = local_signer(tmp_path / "mac-data").public_bytes.hex()
        key_args = []
    else:
        private_key = tmp_path / "offline.key"
        assert cli.main(["release-keygen", str(private_key)]) == 0
        public_key = capsys.readouterr().out.strip().removeprefix("publisher public key: ")
        key_args = ["--key", str(private_key)]
    assert cli.main(["sign-update", "--repo", str(checkout), "--sequence", "1", *key_args]) == 0
    manifest, = (checkout / "dist").glob("*.release.json")
    release = Release.parse(json.loads(manifest.read_bytes()))
    release.verify(public_key)

    cable = SinglePort(drop_candidate)
    data = tmp_path / "receiver-data"
    home = data / "managed-updates"
    home.mkdir(parents=True)
    if not button_pairing:
        (home / "publisher").write_text(public_key)
    peer = uuid.uuid4().hex
    (data / "installation-id").write_text(peer)
    SettingsStore(data / "settings.json").save(AppSettings(
        last_port=cable.port, download_dir=str(data / "downloads"),
    ))
    (data / "downloads").mkdir()
    old_download = data / "downloads" / (release.name + ".zip")
    old_download.write_bytes(b"previous download")
    sender, _, _ = build_core(tmp_path, "controller", update_publisher=public_key if button_pairing else None)
    receiver_identity = IdentityStore(data / "identity.key").load_or_create()
    sender.trust_store.accept(peer, receiver_identity.fingerprint)
    TrustStore(data / "trusted-peers.json").accept(sender.local_peer_id, sender.identity.fingerprint)
    endpoint = tmp_path / "controller.json"
    server = LocalControlServer(sender, endpoint)
    children = []
    errors = []
    stop = threading.Event()
    # Inject only virtual hardware and the isolated application-data directory.
    child_main = """
import os
from pathlib import Path
from types import SimpleNamespace
from shooklink import app
from shooklink.ui import main_window
app.application_data_dir = lambda: Path(os.environ['TEST_DATA'])
app._build_input_backend = lambda: None
main_window.list_ports.comports = lambda: [SimpleNamespace(device=os.environ['TEST_PORT'])]
if os.environ.get('TEST_LOCAL_CLICK') == '1':
    from PySide6.QtCore import QTimer
    apply = app.MainWindow.apply_core_snapshot
    def simulate_local_operator(window, snapshot):
        apply(window, snapshot)
        if window.allow_updates_button.isVisible() and window.allow_updates_button.isEnabled():
            QTimer.singleShot(0, window.allow_updates_button.click)
    app.MainWindow.apply_core_snapshot = simulate_local_operator
raise SystemExit(app.main())
"""
    def launch(folder):
        if stop.is_set():
            raise OSError("test has stopped")
        assert all(child.poll() is not None for child in children)
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", TEST_DATA=str(data), TEST_PORT=cable.port)
        env[MANAGED_ENV] = str(home)
        env["TEST_LOCAL_CLICK"] = "1" if button_pairing and not children else "0"
        with (tmp_path / "gui.log").open("ab") as log:
            child = subprocess.Popen([sys.executable, "-E", "-c", child_main], cwd=folder,
                                     env=env, stdout=log, stderr=log)
        children.append(child)
        return child

    def run():
        try:
            supervise(home, checkout, launch, trial_timeout=4 if drop_candidate else 20)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=run, daemon=True)
    sender.connect_endpoint(cable)
    worker.start()
    try:
        server.start()
        assert wait_for(lambda: sender.trusted, timeout=20), (tmp_path / "gui.log").read_text()
        if button_pairing:
            assert wait_for(lambda: PublisherPin.load(home) == PublisherPin(public_key, sender.identity.fingerprint),
                            timeout=5), (tmp_path / "gui.log").read_text()
        connection = sender.snapshot.connection_id
        assert cli.main(["--endpoint", str(endpoint), "peer-update-status"]) == 0
        assert '"managed": true' in capsys.readouterr().out
        exchange = sender.updates.exchange
        dropped = False

        def lose_first_apply_reply(kind, *args, **kwargs):
            nonlocal dropped
            result = exchange(kind, *args, **kwargs)
            if kind is MessageType.UPDATE_APPLY and not dropped:
                dropped = True
                raise Unavailable("simulated lost apply acknowledgement")
            return result

        sender.updates.exchange = lose_first_apply_reply
        result = cli.main(["--endpoint", str(endpoint), "live-update", str(manifest),
                           "--timeout", "50"])
        captured = capsys.readouterr()
        assert dropped
        if drop_candidate:
            assert result == 1 and "rolled_back" in captured.err, captured
            assert State.load(home).phase == "rolled_back" and not State.load(home).active
            assert len(children) == cable.opens == 3
        else:
            assert result == 0 and f"verified: {release.revision}" in captured.out, captured
            assert State.load(home).phase == "installed"
            assert len(children) == cable.opens == 2
        assert cable.overlaps == 0 and sender.snapshot.connection_id == connection
        assert old_download.read_bytes() == b"previous download"
        assert children[0].poll() == 75 and children[-1].poll() is None
    finally:
        stop.set()
        for child in children:
            if child.poll() is None:
                child.terminate()
        worker.join(10)
        server.close()
        sender.close()
    assert not worker.is_alive() and not errors
