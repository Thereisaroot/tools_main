import os
from concurrent.futures import Future

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal

from shooklink import app
from shooklink.input.macos_backend import MacOSInputBackend
from shooklink.input.windows_backend import WindowsInputBackend
from shooklink.settings import AppSettings, SettingsStore


class ImmediateExecutor:
    def __init__(self):
        self.shutdown_calls = []

    def submit(self, callback, *args):
        future = Future()
        try:
            future.set_result(callback(*args))
        except BaseException as error:
            future.set_exception(error)
        return future

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_calls.append((wait, cancel_futures))


class FakeCore:
    def __init__(self):
        self.snapshot = object()
        self.listeners = []
        self.connect_calls = []
        self.disconnect_calls = 0
        self.approvals = []
        self.close_calls = 0

    def add_state_listener(self, listener):
        self.listeners.append(listener)

    def remove_state_listener(self, listener):
        self.listeners.remove(listener)

    def connect_port(self, port, baud_rate):
        self.connect_calls.append((port, baud_rate))

    def disconnect(self):
        self.disconnect_calls += 1

    def approve_peer(self, connection_id, fingerprint):
        self.approvals.append((connection_id, fingerprint))

    def close(self):
        self.close_calls += 1


class FakeWindow(QObject):
    connect_requested = Signal(str, int)
    disconnect_requested = Signal()
    trust_requested = Signal(int, str)

    def __init__(self, ports):
        super().__init__()
        self._ports = tuple(ports)
        self.defaults = None
        self.snapshots = []
        self.pending = []
        self.errors = []
        self.operation_errors = []
        self.preferences = ("COM9", 460800, "left", True)

    def available_ports(self):
        return self._ports

    def set_serial_defaults(self, port, baud_rate):
        self.defaults = (port, baud_rate)

    def apply_core_snapshot(self, snapshot):
        self.snapshots.append(snapshot)

    def set_connection_pending(self, port):
        self.pending.append(port)

    def show_connection_error(self, message):
        self.errors.append(message)

    def show_operation_error(self, message):
        self.operation_errors.append(message)

    def persistent_preferences(self):
        return self.preferences


def test_input_backend_factory_selects_supported_platform(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "darwin")
    assert isinstance(app._build_input_backend(), MacOSInputBackend)

    monkeypatch.setattr(app.sys, "platform", "win32")
    assert isinstance(app._build_input_backend(), WindowsInputBackend)


def test_input_backend_factory_disables_unsupported_platform(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "linux")
    assert app._build_input_backend() is None


def test_installation_id_is_stable_and_recovers_invalid_data(tmp_path):
    path = tmp_path / "installation-id"
    first = app._load_or_create_peer_id(path)
    second = app._load_or_create_peer_id(path)
    assert second == first
    assert len(first) == 32
    assert set(first) <= set("0123456789abcdef")

    path.write_text("not a peer id", encoding="ascii")
    replacement = app._load_or_create_peer_id(path)
    assert replacement != first
    assert len(replacement) == 32


def test_controller_autoconnects_only_when_saved_port_is_present(qtbot, tmp_path):
    settings = AppSettings(
        last_port="COM9",
        baud_rate=460800,
        peer_side="left",
        auto_edge_enabled=True,
        download_dir=str(tmp_path / "downloads"),
    )
    core = FakeCore()
    window = FakeWindow(("COM3", "COM9"))
    executor = ImmediateExecutor()
    controller = app.ApplicationController(
        core,
        window,
        SettingsStore(tmp_path / "settings.json"),
        settings,
        executor=executor,
    )

    controller.start()
    qtbot.waitUntil(lambda: core.connect_calls == [("COM9", 460800)])
    assert window.defaults == ("COM9", 460800)
    assert window.pending == ["COM9"]

    controller.close()
    assert core.close_calls == 1
    assert executor.shutdown_calls == [(True, True)]
    assert SettingsStore(tmp_path / "settings.json").load().peer_side == "left"


def test_controller_does_not_autoconnect_a_missing_saved_port(qtbot, tmp_path):
    settings = AppSettings(last_port="COM9", baud_rate=460800)
    core = FakeCore()
    window = FakeWindow(("COM3",))
    controller = app.ApplicationController(
        core,
        window,
        SettingsStore(tmp_path / "settings.json"),
        settings,
        executor=ImmediateExecutor(),
    )

    controller.start()
    qtbot.wait(10)
    assert core.connect_calls == []
    controller.close()


def test_trust_failure_preserves_the_active_connection_snapshot(qtbot, tmp_path):
    core = FakeCore()
    active_snapshot = object()
    core.snapshot = active_snapshot

    def fail_approval(_connection_id, _fingerprint):
        raise OSError("trust store unavailable")

    core.approve_peer = fail_approval
    window = FakeWindow(("COM9",))
    controller = app.ApplicationController(
        core,
        window,
        SettingsStore(tmp_path / "settings.json"),
        AppSettings(last_port="COM9"),
        executor=ImmediateExecutor(),
    )

    window.trust_requested.emit(7, "SHA256:peer")
    qtbot.waitUntil(
        lambda: window.operation_errors == ["trust store unavailable"]
    )

    assert window.errors == []
    assert window.snapshots[-1] is active_snapshot
    controller.close()
