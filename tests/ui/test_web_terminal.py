import os
import shutil
import subprocess

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QUrl
from PySide6.QtTest import QTest

from shooklink.ui.web_terminal import ASSETS, OutputQueue, asset_url_allowed, WebTerminal


def test_local_page_can_load_only_the_bundled_ui_and_qt_channel():
    assert asset_url_allowed(QUrl.fromLocalFile(str(ASSETS / "index.html")))
    assert asset_url_allowed(QUrl("qrc:///qtwebchannel/qwebchannel.js"))
    for url in (QUrl("https://example.com/"), QUrl("data:text/html,other"),
                QUrl.fromLocalFile(str(ASSETS.parent / "main_window.py")),
                QUrl("qrc:///elsewhere.js")):
        assert not asset_url_allowed(url)


def test_render_queue_preserves_bytes_until_one_batch_is_acknowledged():
    q = OutputQueue(limit=10, batch=4)
    q.append(b"123456")
    assert q.take() == b"1234" and not q.take()
    q.append(b"7890")
    with pytest.raises(BufferError):
        q.append(b"x")
    assert q.pending() == b"1234567890"
    q.ack()
    assert q.take() == b"5678"


def test_real_offline_xterm_tab_search_unicode_and_alternate_screen(qtbot):
    w = WebTerminal()
    qtbot.addWidget(w)
    w.resize(760, 410)
    w.show()
    sent, failures = [], []
    w.input_bytes.connect(sent.append)
    w.failed.connect(lambda why, _: failures.append(why))
    w.feed_output(b"original prompt> ")
    qtbot.waitUntil(lambda: w.is_ready or bool(failures), timeout=15000)
    assert not failures

    def js(script):
        result = []
        w.page().runJavaScript(script, 0, result.append)
        qtbot.waitUntil(lambda: bool(result), timeout=3000)
        return result[0]

    def contains(text):
        return text in (js("window.shookTerm.screenText()") or "")

    qtbot.waitUntil(lambda: contains("original prompt"))
    w.focus_terminal()
    target = w.focusProxy() or w
    QTest.keyClick(target, Qt.Key.Key_Tab)
    QTest.keyClick(target, Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
    qtbot.waitUntil(lambda: len(sent) >= 2)
    assert sent[:2] == [b"\t", b"\x1b[Z"]
    w.feed_output(b"\x1b[?1049h\x1b[H")
    data = "한글🙂".encode()
    w.feed_output(data[:2])
    w.feed_output(data[2:])
    qtbot.waitUntil(lambda: contains("한글🙂"))
    assert not contains("original prompt")
    w.feed_output(b"\x1b[?1049l")
    qtbot.waitUntil(lambda: contains("original prompt"))
    assert js("window.shookTerm.find('original', false)")
    assert js("window.shookTerm.selection()") == "original"
    w.feed_output(b"\x1b[?2004h")
    qtbot.waitUntil(lambda: not w.output_pending)
    w.paste_text("alpha\nbeta")
    qtbot.waitUntil(lambda: any(b"alpha" in data for data in sent))
    assert b"\x1b[200~" in b"".join(sent)
    w.close()


@pytest.mark.skipif(os.name != "posix" or not shutil.which("bash"), reason="requires a local Bash PTY")
def test_tab_completes_in_a_real_interactive_bash_without_executing_a_command(qtbot, tmp_path):
    import pty
    import select

    (tmp_path / "completion-example.txt").touch()
    master, slave = pty.openpty()
    bash = subprocess.Popen([shutil.which("bash"), "--noprofile", "--norc", "-i"],
                            stdin=slave, stdout=slave, stderr=slave, cwd=tmp_path,
                            env=dict(os.environ, PS1="SHOOK> ", INPUTRC=os.devnull, TERM="xterm-256color"))
    os.close(slave)
    w = WebTerminal()
    qtbot.addWidget(w)
    w.show()
    w.input_bytes.connect(lambda data: os.write(master, data))
    transcript = bytearray()

    def has(marker):
        if select.select([master], [], [], 0)[0]:
            data = os.read(master, 8192)
            transcript.extend(data)
            w.feed_output(data)
        return marker in transcript

    try:
        qtbot.waitUntil(lambda: has(b"SHOOK> ") and w.is_ready, timeout=15000)
        w.focus_terminal()
        w.paste_text("cat completion-ex")
        qtbot.waitUntil(lambda: has(b"cat completion-ex"))
        QTest.keyClick(w.focusProxy() or w, Qt.Key.Key_Tab)
        qtbot.waitUntil(lambda: has(b"completion-example.txt"))
    finally:
        w.close()
        bash.kill()
        bash.wait(timeout=3)
        os.close(master)


def test_rich_popup_falls_back_without_reopening_or_losing_pending_bytes(qtbot, monkeypatch):
    from shooklink.ui.terminal_window import TerminalWindow

    monkeypatch.delenv("SHOOKLINK_TERMINAL_RENDERER", raising=False)
    closed = []
    w = TerminalWindow("f" * 32, lambda _: None, lambda *_: None, lambda: closed.append(True), rich=True)
    qtbot.addWidget(w)
    assert w._web is not None
    w.feed_output(b"pending prompt> ")
    w._web._fail("test renderer failure")
    qtbot.waitUntil(lambda: "pending prompt" in w.terminal_view.toPlainText())
    assert w._web is None and not closed
    w.close()
    assert closed == [True]


def test_integrated_popup_uses_xterm_find_and_keeps_command_editor_optional(qtbot, monkeypatch):
    from shooklink.ui.terminal_window import TerminalWindow
    from PySide6.QtWidgets import QApplication

    monkeypatch.delenv("SHOOKLINK_TERMINAL_RENDERER", raising=False)
    sent = []
    w = TerminalWindow("e" * 32, sent.append, lambda *_: None, lambda: None, rich=True)
    qtbot.addWidget(w)
    w.show()
    w.feed_output(b"Completed step 135")
    qtbot.waitUntil(lambda: w.renderer_label.text() == "xterm.js" and not w._web.output_pending, timeout=15000)
    assert not w.line_mode.isChecked()
    w._web.focus_terminal()
    QTest.keyClick(w._web.focusProxy(), Qt.Key.Key_F,
                   Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert w.find_bar.isVisible()
    w.find_field.setText("step 135")
    with qtbot.waitSignal(w._web.search_result, check_params_cb=lambda matched: matched):
        qtbot.keyClick(w.find_field, Qt.Key.Key_Return)
    w._copy()
    qtbot.waitUntil(lambda: QApplication.clipboard().text() == "step 135")
    assert not sent
    w._hide_find()
    QTest.keyClick(w._web.focusProxy(), Qt.Key.Key_Minus, Qt.KeyboardModifier.ControlModifier)
    qtbot.waitUntil(lambda: w._zoom_steps == -1)
    QTest.keyClick(w._web.focusProxy(), Qt.Key.Key_0, Qt.KeyboardModifier.ControlModifier)
    qtbot.waitUntil(lambda: w._zoom_steps == 0)
    w.close()
