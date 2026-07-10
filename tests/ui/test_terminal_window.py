import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from shooklink.ui.terminal_window import TerminalWindow


def test_terminal_renders_vt_output_and_bounds_history(qtbot):
    sent = []
    resized = []
    closed = []
    window = TerminalWindow(
        "a" * 32,
        sent.append,
        lambda columns, rows: resized.append((columns, rows)),
        lambda: closed.append(True),
    )
    qtbot.addWidget(window)
    window.show()

    window.feed_output(b"\x1b[2J\x1b[HPTY_OK")

    qtbot.waitUntil(lambda: "PTY_OK" in window.terminal_view.toPlainText())
    assert window.screen.history.size == 10_000
    window.close()
    assert closed == [True]


def test_terminal_forwards_ctrl_c_arrows_text_and_paste(qtbot):
    sent = []
    window = TerminalWindow(
        "b" * 32,
        sent.append,
        lambda columns, rows: None,
        lambda: None,
    )
    qtbot.addWidget(window)
    window.show()
    window.terminal_view.setFocus()

    qtbot.keyClick(
        window.terminal_view,
        Qt.Key.Key_C,
        Qt.KeyboardModifier.ControlModifier,
    )
    qtbot.keyClick(window.terminal_view, Qt.Key.Key_Up)
    qtbot.keyClicks(window.terminal_view, "hello")
    QApplication.clipboard().setText("한글 paste")
    qtbot.keyClick(
        window.terminal_view,
        Qt.Key.Key_V,
        Qt.KeyboardModifier.ControlModifier
        | Qt.KeyboardModifier.ShiftModifier,
    )

    assert sent[0] == b"\x03"
    assert sent[1] == b"\x1b[A"
    assert b"".join(sent[2:7]) == b"hello"
    assert sent[-1] == "한글 paste".encode()


def test_remote_exit_does_not_send_a_second_close(qtbot):
    closed = []
    window = TerminalWindow(
        "c" * 32,
        lambda data: None,
        lambda columns, rows: None,
        lambda: closed.append(True),
    )
    qtbot.addWidget(window)
    window.show()

    window.mark_exited(0)
    window.close()

    assert closed == []


def test_large_paste_is_emitted_in_bounded_chunks(qtbot):
    sent = []
    window = TerminalWindow(
        "d" * 32,
        sent.append,
        lambda columns, rows: None,
        lambda: None,
    )
    qtbot.addWidget(window)
    QApplication.clipboard().setText("x" * 100_000)

    window.terminal_view.paste_to_remote()

    assert all(len(chunk) <= 16 * 1024 for chunk in sent)
    assert b"".join(sent) == b"x" * 100_000


def test_rendered_qt_cursor_tracks_pyte_cursor(qtbot):
    window = TerminalWindow(
        "e" * 32,
        lambda data: None,
        lambda columns, rows: None,
        lambda: None,
    )
    qtbot.addWidget(window)
    window.show()

    window.feed_output(b"abc\x1b[2D")

    qtbot.waitUntil(lambda: window.terminal_view.toPlainText().startswith("abc"))
    assert window.screen.cursor.x == 1
    assert window.terminal_view.textCursor().positionInBlock() == 1
