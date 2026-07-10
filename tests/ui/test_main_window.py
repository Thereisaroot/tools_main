import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from shooklink.chat.service import ChatService
from shooklink.protocol.messages import Message, MessageType
from shooklink.ui.main_window import MainWindow


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
