import os
import threading
from concurrent.futures import Future
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QRect, Qt, QUrl
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QScrollArea

from shooklink.chat.service import ChatService
from shooklink.core import CoreSnapshot, CoreState
from shooklink.files.service import FileProgress
from shooklink.input.backend import PermissionStatus
from shooklink.input.service import InputSessionState, InputStateChange
from shooklink.input.topology import Side
from shooklink.protocol.crypto import TrustStatus
from shooklink.protocol.messages import Message, MessageType
from shooklink.shell.service import ShellOutput, ShellState
from shooklink.ui.main_window import FileDropZone, MainWindow


class FakeBus:
    def __init__(
        self,
        trusted=True,
        decrypted_body=None,
        *,
        chat_ack_available=False,
        chat_connection_id=1,
    ):
        self.trusted = trusted
        self.decrypted_body = decrypted_body
        self.chat_ack_available = chat_ack_available
        self.chat_connection_id = chat_connection_id
        self.sent = []
        self.on_written_callbacks = []

    def send(
        self,
        message,
        *,
        secure=False,
        on_written=None,
        expected_connection_id=None,
    ):
        if (
            expected_connection_id is not None
            and expected_connection_id != self.chat_connection_id
        ):
            raise RuntimeError("serial connection changed")
        self.sent.append((message, secure))
        self.on_written_callbacks.append(on_written)

    def decrypt_secure(self, _message):
        if not self.trusted or self.decrypted_body is None:
            raise RuntimeError("secure payload was not authenticated")
        return self.decrypted_body


class FakeFileService:
    def __init__(self, download_dir):
        self.download_dir = Path(download_dir)
        self.listeners = []
        self.sent_paths = []
        self.cancelled = []

    def add_progress_listener(self, listener):
        self.listeners.append(listener)

    def remove_progress_listener(self, listener):
        if listener in self.listeners:
            self.listeners.remove(listener)

    def send_file(self, path):
        self.sent_paths.append(Path(path))
        future = Future()
        future.set_result("f" * 32)
        return future

    def cancel(self, transfer_id):
        self.cancelled.append(transfer_id)

    def emit(self, progress):
        for listener in tuple(self.listeners):
            listener(progress)


class FakeShellService:
    def __init__(self, *, allowed=False):
        self.allowed = allowed
        self.outputs = []
        self.states = []
        self.opened = []
        self.inputs = []
        self.resizes = []
        self.closed = []

    def add_output_listener(self, listener):
        self.outputs.append(listener)

    def remove_output_listener(self, listener):
        self.outputs.remove(listener)

    def add_state_listener(self, listener):
        self.states.append(listener)

    def remove_state_listener(self, listener):
        self.states.remove(listener)

    def set_allow_remote_shell(self, allowed):
        self.allowed = allowed

    @property
    def allow_remote_shell(self):
        return self.allowed

    def open_remote(self, *, columns, rows):
        self.opened.append((columns, rows))
        return "a" * 32

    def send_input(self, session_id, data):
        self.inputs.append((session_id, data))

    def resize(self, session_id, columns, rows):
        self.resizes.append((session_id, columns, rows))

    def close_session(self, session_id):
        self.closed.append(session_id)

    def emit_output(self, output):
        for listener in tuple(self.outputs):
            listener(output)

    def emit_state(self, state):
        for listener in tuple(self.states):
            listener(state)


class FakeInputService:
    def __init__(self, *, allowed=False):
        self.state = InputSessionState.IDLE
        self.peer_side = Side.RIGHT
        self.auto_edge_enabled = False
        self.allowed = allowed
        self.listeners = []
        self.emergency_listeners = []
        self.requests = 0
        self.stops = []
        self.connection_changes = []

    def permission_status(self):
        return PermissionStatus(True, True, "ready")

    def add_state_listener(self, listener):
        self.listeners.append(listener)

    def remove_state_listener(self, listener):
        self.listeners.remove(listener)

    def add_emergency_listener(self, listener):
        self.emergency_listeners.append(listener)

    def remove_emergency_listener(self, listener):
        self.emergency_listeners.remove(listener)

    def set_allow_remote_input(self, allowed):
        self.allowed = allowed

    @property
    def allow_remote_input(self):
        return self.allowed

    def set_peer_side(self, side):
        self.peer_side = side if isinstance(side, Side) else Side(side)

    def set_auto_edge_enabled(self, enabled):
        self.auto_edge_enabled = enabled

    def request_control(self):
        self.requests += 1
        self.emit(InputStateChange(InputSessionState.REQUESTING, "1" * 32))
        return "1" * 32

    def stop_control(self, *, reason="manual"):
        self.stops.append(reason)
        self.emit(InputStateChange(InputSessionState.IDLE, reason=reason))

    def connection_changed(self, connected):
        self.connection_changes.append(connected)

    def emit(self, change):
        self.state = change.state
        for listener in tuple(self.listeners):
            listener(change)

    def emit_emergency(self, action):
        for listener in tuple(self.emergency_listeners):
            listener(action)


def test_main_window_sends_korean_and_punctuation(qtbot):
    bus = FakeBus()
    window = MainWindow(ChatService(bus))
    qtbot.addWidget(window)
    window.show()
    text = "한글 ./?=+-_0)"

    window.message_editor.setPlainText(text)
    qtbot.mouseClick(window.send_plain_button, Qt.MouseButton.LeftButton)
    window.message_editor.setPlainText(text)
    qtbot.mouseClick(window.send_secure_button, Qt.MouseButton.LeftButton)

    assert [item[0].body.decode() for item in bus.sent] == [text, text]
    assert [item[1] for item in bus.sent] == [False, True]


def test_send_status_changes_to_sent_only_after_peer_ack(qtbot):
    bus = FakeBus(chat_ack_available=True)
    service = ChatService(bus)
    window = MainWindow(service)
    qtbot.addWidget(window)
    window.show()
    window.message_editor.setPlainText("after file")

    qtbot.mouseClick(window.send_plain_button, Qt.MouseButton.LeftButton)

    assert window.action_status.text() == "Queued"
    callback = bus.on_written_callbacks[-1]
    assert callback is not None
    writer = threading.Thread(target=callback)
    writer.start()
    writer.join(timeout=1)
    assert window.action_status.text() == "Queued"

    message_id = bus.sent[-1][0].metadata["message_id"]
    reader = threading.Thread(
        target=service.handle_message,
        args=(Message(MessageType.CHAT_PLAIN, {"ack_id": message_id}, b""),),
    )
    reader.start()
    reader.join(timeout=1)
    qtbot.waitUntil(lambda: window.action_status.text() == "Sent")


def test_copy_actions_target_last_received_text(qtbot):
    bus = FakeBus(decrypted_body=b"secure first")
    service = ChatService(bus)
    window = MainWindow(service)
    qtbot.addWidget(window)
    window.show()
    clipboard = QApplication.clipboard()

    service.handle_message(Message(MessageType.CHAT_SECURE, {}, b"ciphertext"))
    service.handle_message(Message(MessageType.CHAT_PLAIN, {}, b"plain last"))
    qtbot.mouseClick(window.copy_secure_button, Qt.MouseButton.LeftButton)
    assert clipboard.text() == "plain last"

    qtbot.mouseClick(window.copy_last_button, Qt.MouseButton.LeftButton)
    assert clipboard.text() == "plain last"


def test_editor_uses_native_copy_and_paste_shortcuts(qtbot):
    window = MainWindow(ChatService(FakeBus()))
    qtbot.addWidget(window)
    window.show()
    window.message_editor.setFocus()
    QApplication.clipboard().setText("한글 붙여넣기")

    qtbot.keyClick(window.message_editor, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert window.message_editor.toPlainText() == "한글 붙여넣기"

    window.message_editor.selectAll()
    qtbot.keyClick(window.message_editor, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert QApplication.clipboard().text() == "한글 붙여넣기"


def test_connection_shell_contains_expected_controls(qtbot):
    window = MainWindow(ChatService(FakeBus()))
    qtbot.addWidget(window)

    assert window.port_combo.isEditable()
    assert window.baud_combo.isEditable()
    assert window.baud_combo.findText("750000") >= 0
    assert window.connect_button.text() == "Connect"
    assert window.connection_status.text() == "Disconnected"


def test_default_window_height_fits_full_content_on_tall_screen(
    qtbot,
    tmp_path,
    monkeypatch,
):
    screen = type(
        "TallScreen",
        (),
        {"availableGeometry": lambda self: QRect(0, 0, 2560, 1440)},
    )()
    monkeypatch.setattr(
        QApplication,
        "primaryScreen",
        staticmethod(lambda: screen),
    )
    window = MainWindow(
        ChatService(FakeBus()),
        FakeFileService(tmp_path),
        FakeShellService(),
        FakeInputService(),
    )
    qtbot.addWidget(window)

    scroll_area = window.centralWidget()
    assert isinstance(scroll_area, QScrollArea)
    assert window.height() >= scroll_area.widget().sizeHint().height()
    assert window.height() > 760


def test_short_screen_keeps_window_visible_and_scrolls_content(
    qtbot,
    tmp_path,
    monkeypatch,
):
    screen = type(
        "ShortScreen",
        (),
        {"availableGeometry": lambda self: QRect(0, 0, 1280, 800)},
    )()
    monkeypatch.setattr(
        QApplication,
        "primaryScreen",
        staticmethod(lambda: screen),
    )
    window = MainWindow(
        ChatService(FakeBus()),
        FakeFileService(tmp_path),
        FakeShellService(),
        FakeInputService(),
    )
    qtbot.addWidget(window)
    window.show()
    QApplication.processEvents()

    scroll_area = window.centralWidget()
    assert isinstance(scroll_area, QScrollArea)
    assert window.height() <= 800 - 64
    assert scroll_area.verticalScrollBar().maximum() > 0


def test_light_surface_text_remains_dark_with_a_dark_system_palette(qtbot):
    application = QApplication.instance()
    original_palette = QPalette(application.palette())
    dark_palette = QPalette(original_palette)
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
    ):
        dark_palette.setColor(group, QPalette.ColorRole.Text, QColor("#ffffff"))
        dark_palette.setColor(group, QPalette.ColorRole.WindowText, QColor("#ffffff"))
        dark_palette.setColor(group, QPalette.ColorRole.ButtonText, QColor("#ffffff"))
    application.setPalette(dark_palette)

    try:
        window = MainWindow(ChatService(FakeBus()))
        qtbot.addWidget(window)
        window.show()
        window.ensurePolished()
        widgets_and_roles = (
            (window.port_combo, QPalette.ColorRole.Text),
            (window.port_combo.lineEdit(), QPalette.ColorRole.Text),
            (window.message_editor, QPalette.ColorRole.Text),
            (window.allow_input_checkbox, QPalette.ColorRole.WindowText),
            (window.copy_last_button, QPalette.ColorRole.ButtonText),
        )

        for widget, role in widgets_and_roles:
            widget.ensurePolished()
            color = widget.palette().color(QPalette.ColorGroup.Active, role)
            assert color == QColor("#172421")
    finally:
        application.setPalette(original_palette)


def test_connection_state_exposes_trust_approval_and_disconnect(qtbot):
    bus = FakeBus(trusted=False)
    window = MainWindow(ChatService(bus))
    qtbot.addWidget(window)
    trust_requests = []
    disconnects = []
    window.trust_requested.connect(
        lambda connection_id, fingerprint: trust_requests.append(
            (connection_id, fingerprint)
        )
    )
    window.disconnect_requested.connect(lambda: disconnects.append(True))

    fingerprint = "SHA256:peer-fingerprint"
    window.apply_core_snapshot(
        CoreSnapshot(
            7,
            CoreState.UNTRUSTED,
            "peer-installation",
            fingerprint,
            TrustStatus.UNKNOWN,
            False,
            False,
            frozenset({"chat"}),
        )
    )

    assert window.connection_status.text() == "Untrusted"
    assert fingerprint in window.peer_fingerprint.text()
    assert window.trust_button.isEnabled()
    qtbot.mouseClick(window.trust_button, Qt.MouseButton.LeftButton)
    assert trust_requests == [(7, fingerprint)]

    window.apply_core_snapshot(
        CoreSnapshot(
            7,
            CoreState.UNTRUSTED,
            "peer-installation",
            fingerprint,
            TrustStatus.TRUSTED,
            True,
            False,
            frozenset({"chat"}),
        )
    )
    assert window.connection_status.text() == "Awaiting peer approval"
    assert not window.trust_button.isEnabled()

    qtbot.mouseClick(window.connect_button, Qt.MouseButton.LeftButton)
    assert disconnects == [True]


def test_changed_peer_identity_is_blocked_in_connection_state(qtbot):
    window = MainWindow(ChatService(FakeBus(trusted=False)))
    qtbot.addWidget(window)

    window.apply_core_snapshot(
        CoreSnapshot(
            9,
            CoreState.CHANGED,
            "peer-installation",
            "SHA256:changed",
            TrustStatus.CHANGED,
        )
    )

    assert window.connection_status.text() == "Identity changed"
    assert not window.trust_button.isEnabled()


def test_core_snapshot_gates_features_by_readiness_and_negotiation(qtbot, tmp_path):
    bus = FakeBus(trusted=False)
    files = FakeFileService(tmp_path / "downloads")
    shell = FakeShellService()
    input_service = FakeInputService()
    window = MainWindow(ChatService(bus), files, shell, input_service)
    qtbot.addWidget(window)

    window.apply_core_snapshot(CoreSnapshot(0, CoreState.DISCONNECTED))
    assert not window.send_plain_button.isEnabled()
    assert not window.send_secure_button.isEnabled()
    assert not window.file_select_button.isEnabled()
    assert not window.open_shell_button.isEnabled()
    assert not window.toggle_input_button.isEnabled()

    window.apply_core_snapshot(
        CoreSnapshot(
            1,
            CoreState.UNTRUSTED,
            "peer",
            "SHA256:peer",
            TrustStatus.UNKNOWN,
            features=frozenset({"chat", "files", "shell", "input"}),
        )
    )
    assert window.send_plain_button.isEnabled()
    assert not window.send_secure_button.isEnabled()
    assert not window.file_select_button.isEnabled()

    bus.trusted = True
    window.apply_core_snapshot(
        CoreSnapshot(
            1,
            CoreState.READY,
            "peer",
            "SHA256:peer",
            TrustStatus.TRUSTED,
            True,
            True,
            frozenset({"chat", "input"}),
        )
    )
    assert window.send_plain_button.isEnabled()
    assert window.send_secure_button.isEnabled()
    assert not window.file_select_button.isEnabled()
    assert not window.open_shell_button.isEnabled()
    assert window.toggle_input_button.isEnabled()


def test_disconnecting_snapshot_disables_peer_features(qtbot, tmp_path):
    bus = FakeBus(trusted=True)
    window = MainWindow(
        ChatService(bus),
        FakeFileService(tmp_path / "downloads"),
        FakeShellService(),
        FakeInputService(),
    )
    qtbot.addWidget(window)

    window.apply_core_snapshot(
        CoreSnapshot(
            1,
            CoreState.DISCONNECTING,
            "peer",
            "SHA256:peer",
            TrustStatus.TRUSTED,
            True,
            True,
            frozenset({"chat", "files", "shell", "input"}),
        )
    )

    assert window.connection_status.text() == "Disconnecting"
    assert window.connect_button.text() == "Disconnect"
    assert not window.send_plain_button.isEnabled()
    assert not window.send_secure_button.isEnabled()
    assert not window.file_select_button.isEnabled()
    assert not window.open_shell_button.isEnabled()
    assert not window.toggle_input_button.isEnabled()


def test_file_drop_zone_accepts_local_files_only(tmp_path):
    local_path = tmp_path / "drop.txt"
    local_path.write_text("drop", encoding="utf-8")
    local = QMimeData()
    local.setUrls([QUrl.fromLocalFile(str(local_path))])
    remote = QMimeData()
    remote.setUrls([QUrl("https://example.com/file.txt")])

    assert FileDropZone.local_file_path(local) == local_path
    assert FileDropZone.local_file_path(remote) is None


def test_file_drop_progress_and_cancel_leave_chat_enabled(qtbot, tmp_path):
    path = tmp_path / "send.bin"
    path.write_bytes(b"payload")
    files = FakeFileService(tmp_path / "downloads")
    window = MainWindow(ChatService(FakeBus()), files)
    qtbot.addWidget(window)
    window.show()

    window.file_drop_zone.file_dropped.emit(str(path))
    assert files.sent_paths == [path]

    files.emit(
        FileProgress(
            "f" * 32,
            path.name,
            "outgoing",
            5,
            10,
            "sending",
            path,
            2048.0,
        )
    )
    assert window.file_progress_bar.value() == 50
    assert "50%" in window.file_progress_label.text()
    assert "2.0 KiB/s" in window.file_progress_label.text()
    assert window.send_plain_button.isEnabled()
    assert window.send_secure_button.isEnabled()

    qtbot.mouseClick(window.file_cancel_button, Qt.MouseButton.LeftButton)
    assert files.cancelled == ["f" * 32]

    final_path = tmp_path / "downloads" / path.name
    files.emit(
        FileProgress(
            "f" * 32,
            path.name,
            "incoming",
            10,
            10,
            "complete",
            final_path,
            1024.0,
        )
    )
    assert path.name in window.file_progress_label.text()
    assert "complete" in window.file_progress_label.text()
    assert str(final_path) not in window.file_progress_label.text()


def test_unrelated_completion_does_not_clear_active_transfer(qtbot, tmp_path):
    files = FakeFileService(tmp_path / "downloads")
    window = MainWindow(ChatService(FakeBus()), files)
    qtbot.addWidget(window)
    window.show()
    first = FileProgress("1" * 32, "first", "outgoing", 1, 10, "sending")
    second = FileProgress("2" * 32, "second", "outgoing", 1, 10, "sending")

    files.emit(first)
    files.emit(second)
    files.emit(
        FileProgress(
            first.transfer_id,
            first.name,
            first.direction,
            10,
            10,
            "complete",
            tmp_path / "first",
        )
    )
    qtbot.mouseClick(window.file_cancel_button, Qt.MouseButton.LeftButton)

    assert files.cancelled == [second.transfer_id]


def test_cancelling_current_transfer_selects_remaining_active_transfer(qtbot, tmp_path):
    files = FakeFileService(tmp_path / "downloads")
    window = MainWindow(ChatService(FakeBus()), files)
    qtbot.addWidget(window)
    window.show()
    first = FileProgress("1" * 32, "first", "outgoing", 1, 10, "sending")
    second = FileProgress("2" * 32, "second", "outgoing", 1, 10, "sending")
    files.emit(first)
    files.emit(second)

    qtbot.mouseClick(window.file_cancel_button, Qt.MouseButton.LeftButton)

    assert files.cancelled == [second.transfer_id]
    assert window._active_transfer_id == first.transfer_id
    assert window.file_cancel_button.isEnabled()


def test_terminal_file_progress_history_is_bounded(qtbot, tmp_path):
    files = FakeFileService(tmp_path / "downloads")
    window = MainWindow(ChatService(FakeBus()), files)
    qtbot.addWidget(window)

    for index in range(200):
        files.emit(
            FileProgress(
                f"{index + 1:032x}",
                f"{index}.bin",
                "incoming",
                1,
                1,
                "complete",
                tmp_path / f"{index}.bin",
            )
        )

    assert len(window._file_transfers) <= 128


def test_remote_shell_controls_open_terminal_and_forward_output(qtbot):
    shell = FakeShellService()
    window = MainWindow(ChatService(FakeBus()), None, shell)
    qtbot.addWidget(window)
    window.show()

    qtbot.mouseClick(window.allow_shell_checkbox, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.open_shell_button, Qt.MouseButton.LeftButton)
    session_id = "a" * 32
    shell.emit_state(ShellState(session_id, "outgoing", "active"))
    shell.emit_output(ShellOutput(session_id, b"SHELL_OK"))

    assert shell.allowed is True
    assert shell.opened == [(100, 30)]
    assert window.terminal_window is not None
    qtbot.waitUntil(
        lambda: "SHELL_OK" in window.terminal_window.terminal_view.toPlainText()
    )

    qtbot.mouseClick(window.terminate_shell_button, Qt.MouseButton.LeftButton)
    assert shell.closed[-1] == session_id


def test_input_share_controls_state_and_leave_chat_file_actions_enabled(qtbot, tmp_path):
    files = FakeFileService(tmp_path / "downloads")
    input_service = FakeInputService()
    window = MainWindow(
        ChatService(FakeBus()),
        files,
        None,
        input_service,
    )
    qtbot.addWidget(window)
    window.show()

    assert window.input_permission_status.text() == "ready"
    assert window.peer_side_combo.currentText() == "Right"
    qtbot.mouseClick(window.allow_input_checkbox, Qt.MouseButton.LeftButton)
    window.peer_side_combo.setCurrentText("Left")
    qtbot.mouseClick(window.auto_edge_checkbox, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.toggle_input_button, Qt.MouseButton.LeftButton)

    assert input_service.allowed is True
    assert input_service.peer_side is Side.LEFT
    assert input_service.auto_edge_enabled is True
    assert input_service.requests == 1
    assert window.input_status.text() == "Requesting remote control"

    input_service.emit(
        InputStateChange(InputSessionState.CONTROLLING, "1" * 32)
    )
    qtbot.waitUntil(lambda: window.input_status.text() == "Controlling remote")
    assert window.send_plain_button.isEnabled()
    assert window.send_secure_button.isEnabled()
    assert window.file_select_button.isEnabled()

    window.set_connected(True)
    window.set_connected(False)
    assert input_service.connection_changes == []

    qtbot.mouseClick(window.toggle_input_button, Qt.MouseButton.LeftButton)
    assert input_service.stops == ["manual"]


def test_authorization_checkboxes_restore_service_state_and_emit_preferences(qtbot):
    shell = FakeShellService(allowed=True)
    input_service = FakeInputService(allowed=True)
    window = MainWindow(
        ChatService(FakeBus()),
        None,
        shell,
        input_service,
    )
    qtbot.addWidget(window)
    window.show()
    changes = []
    window.preferences_changed.connect(lambda: changes.append(True))

    assert window.allow_shell_checkbox.isChecked()
    assert window.allow_input_checkbox.isChecked()
    assert window.persistent_preferences()[-2:] == (True, True)

    qtbot.mouseClick(window.allow_shell_checkbox, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.allow_input_checkbox, Qt.MouseButton.LeftButton)

    assert shell.allowed is False
    assert input_service.allowed is False
    assert len(changes) == 2


def test_auto_edge_control_explains_its_local_direction(qtbot):
    window = MainWindow(
        ChatService(FakeBus()),
        None,
        None,
        FakeInputService(),
    )
    qtbot.addWidget(window)

    assert "This Computer -> Peer" in window.auto_edge_checkbox.text()
    assert "Return works" in window.auto_edge_checkbox.toolTip()


def test_being_controlled_state_is_visible_and_listener_is_removed_on_close(qtbot):
    input_service = FakeInputService()
    window = MainWindow(ChatService(FakeBus()), None, None, input_service)
    qtbot.addWidget(window)
    window.show()

    input_service.emit(
        InputStateChange(InputSessionState.BEING_CONTROLLED, "2" * 32)
    )
    qtbot.waitUntil(lambda: window.input_status.text() == "Being controlled")
    assert "Ctrl+Alt+Shift" in window.input_emergency_help.text()

    exits = []
    window.emergency_exit_requested.connect(lambda: exits.append(True))
    input_service.emit_emergency("exit")
    qtbot.waitUntil(lambda: exits == [True])

    window.close()
    assert input_service.listeners == []
    assert input_service.emergency_listeners == []
