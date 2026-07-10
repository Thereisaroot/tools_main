import os
from concurrent.futures import Future
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, Qt, QUrl
from PySide6.QtWidgets import QApplication

from shooklink.chat.service import ChatService
from shooklink.files.service import FileProgress
from shooklink.protocol.messages import Message, MessageType
from shooklink.shell.service import ShellOutput, ShellState
from shooklink.ui.main_window import FileDropZone, MainWindow


class FakeBus:
    def __init__(self, trusted=True, decrypted_body=None):
        self.trusted = trusted
        self.decrypted_body = decrypted_body
        self.sent = []

    def send(self, message, *, secure=False):
        self.sent.append((message, secure))

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
    def __init__(self):
        self.allowed = False
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
    assert window.connect_button.text() == "Connect"
    assert window.connection_status.text() == "Disconnected"


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
    assert str(final_path) in window.file_progress_label.text()


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
