"""ANSI terminal popup for remote PTY and ConPTY sessions."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from itertools import groupby

import pyte
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QContextMenuEvent,
    QFont,
    QKeyEvent,
    QKeySequence,
    QResizeEvent,
    QShortcut,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)
_KEY_SEQUENCES = {
    Qt.Key.Key_Up: b"\x1b[A",
    Qt.Key.Key_Down: b"\x1b[B",
    Qt.Key.Key_Right: b"\x1b[C",
    Qt.Key.Key_Left: b"\x1b[D",
    Qt.Key.Key_Home: b"\x1b[H",
    Qt.Key.Key_End: b"\x1b[F",
    Qt.Key.Key_Insert: b"\x1b[2~",
    Qt.Key.Key_Delete: b"\x1b[3~",
    Qt.Key.Key_PageUp: b"\x1b[5~",
    Qt.Key.Key_PageDown: b"\x1b[6~",
    Qt.Key.Key_F1: b"\x1bOP",
    Qt.Key.Key_F2: b"\x1bOQ",
    Qt.Key.Key_F3: b"\x1bOR",
    Qt.Key.Key_F4: b"\x1bOS",
    Qt.Key.Key_F5: b"\x1b[15~",
    Qt.Key.Key_F6: b"\x1b[17~",
    Qt.Key.Key_F7: b"\x1b[18~",
    Qt.Key.Key_F8: b"\x1b[19~",
    Qt.Key.Key_F9: b"\x1b[20~",
    Qt.Key.Key_F10: b"\x1b[21~",
    Qt.Key.Key_F11: b"\x1b[23~",
    Qt.Key.Key_F12: b"\x1b[24~",
    Qt.Key.Key_Return: b"\r",
    Qt.Key.Key_Enter: b"\r",
    Qt.Key.Key_Backspace: b"\x7f",
    Qt.Key.Key_Tab: b"\t",
    Qt.Key.Key_Backtab: b"\x1b[Z",
    Qt.Key.Key_Escape: b"\x1b",
}
MAX_TERMINAL_INPUT_CHUNK = 16 * 1024
_ANSI_COLORS = {
    "black": "#1a1f1e",
    "red": "#d75f5f",
    "green": "#5faf87",
    "brown": "#d7af5f",
    "blue": "#5f87d7",
    "magenta": "#af87d7",
    "cyan": "#5fafaf",
    "white": "#d8e7df",
    "brightblack": "#68736f",
    "brightred": "#ff8787",
    "brightgreen": "#87d7af",
    "brightbrown": "#ffdf87",
    "brightblue": "#87afff",
    "brightmagenta": "#d7afff",
    "brightcyan": "#87d7d7",
    "brightwhite": "#ffffff",
}


class TerminalView(QPlainTextEdit):
    input_bytes = Signal(bytes)
    terminal_resized = Signal(int, int)
    history_page = Signal(int)
    selection_finished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.selecting = False
        font = QFont("Menlo" if sys.platform == "darwin" else "Consolas", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self._last_size: tuple[int, int] | None = None

    def event(self, event) -> bool:
        if event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress) and (
            event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab)
            and event.modifiers() in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier)
            and self.focusPolicy() != Qt.FocusPolicy.NoFocus
        ):
            if event.type() == QEvent.Type.KeyPress:
                reverse = event.key() == Qt.Key.Key_Backtab or bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                self._send_keys(b"\x1b[Z" if reverse else b"\t")
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        modifiers = event.modifiers()
        control = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        alt = bool(modifiers & Qt.KeyboardModifier.AltModifier)

        if control and event.key() == Qt.Key.Key_C:
            if shift or self.textCursor().hasSelection():
                self.copy()
            else:
                self._send_keys(b"\x03")
            event.accept()
            return
        if control and shift and event.key() == Qt.Key.Key_V:
            self.paste_to_remote()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste_to_remote()
            event.accept()
            return
        if shift and event.key() == Qt.Key.Key_PageUp:
            self.history_page.emit(-1)
            event.accept()
            return
        if shift and event.key() == Qt.Key.Key_PageDown:
            self.history_page.emit(1)
            event.accept()
            return
        if control and Qt.Key.Key_A <= event.key() <= Qt.Key.Key_Z:
            self._send_keys(bytes((event.key() - Qt.Key.Key_A + 1,)))
            event.accept()
            return

        payload = _KEY_SEQUENCES.get(event.key())
        if payload is None and event.text():
            payload = event.text().encode("utf-8")
        if payload is not None:
            if alt:
                payload = b"\x1b" + payload
            self._send_keys(payload)
            event.accept()
            return
        event.ignore()

    def _send_keys(self, data: bytes) -> None:
        cursor = self.textCursor()
        if cursor.hasSelection():
            cursor.clearSelection()
            self.setTextCursor(cursor)
            self.selection_finished.emit()
        self.input_bytes.emit(data)

    def inputMethodEvent(self, event) -> None:
        if event.commitString():
            self._send_keys(event.commitString().encode("utf-8"))
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.selecting = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.selecting = False
            self.selection_finished.emit()

    def paste_to_remote(self) -> None:
        text = QApplication.clipboard().text()
        if text:
            encoded = text.encode("utf-8")
            for offset in range(0, len(encoded), MAX_TERMINAL_INPUT_CHUNK):
                self._send_keys(
                    encoded[offset : offset + MAX_TERMINAL_INPUT_CHUNK]
                )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.sync_size()

    def sync_size(self) -> None:
        metrics = self.fontMetrics()
        cell_width = max(1, metrics.horizontalAdvance("M"))
        cell_height = max(1, metrics.lineSpacing())
        columns = max(2, min(500, self.viewport().width() // cell_width))
        rows = max(2, min(500, self.viewport().height() // cell_height))
        size = (columns, rows)
        if size != self._last_size:
            self._last_size = size
            self.terminal_resized.emit(columns, rows)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        menu = QMenu(self)
        copy_action = menu.addAction("Copy")
        copy_action.setEnabled(bool(self.textCursor().selectedText()))
        paste_action = menu.addAction("Paste to Remote")
        selected = menu.exec(event.globalPos())
        if selected is copy_action:
            self.copy()
        elif selected is paste_action:
            self.paste_to_remote()


class TerminalWindow(QMainWindow):
    output_received = Signal(bytes)
    exit_received = Signal(object, str)

    def __init__(
        self,
        session_id: str,
        send_input: Callable[[bytes], None],
        resize_remote: Callable[[int, int], None],
        close_remote: Callable[[], None],
        *,
        rich: bool = False,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self._send_input = send_input
        self._resize_remote = resize_remote
        self._close_remote = close_remote
        self._remote_exited = False
        self._connected = True
        self._web = None
        self._zoom_steps = 0
        self.screen = pyte.HistoryScreen(80, 24, history=10_000)
        self._stream = pyte.ByteStream(self.screen)
        self._rows = {}
        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(12)
        self._frame_timer.timeout.connect(self._render)
        self.setWindowTitle("ShookLink Remote Shell")
        self.resize(920, 620)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.status_label = QLabel("Remote shell connected")
        self.status_label.setObjectName("terminalStatus")
        self.status_label.setContentsMargins(12, 7, 12, 7)
        self.terminal_view = TerminalView()
        layout.addWidget(self.status_label)
        layout.addWidget(self.terminal_view, 1)
        self.find_bar = QWidget()
        find_layout = QHBoxLayout(self.find_bar)
        find_layout.setContentsMargins(8, 2, 8, 2)
        self.find_field = QLineEdit()
        self.find_field.setPlaceholderText("Find in terminal output")
        find_layout.addWidget(self.find_field, 1)
        self.find_result = QLabel()
        find_layout.addWidget(self.find_result)
        for label, callback in (("Previous", lambda: self._find(True)), ("Next", self._find),
                                ("Close", self._hide_find)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, fn=callback: fn())
            find_layout.addWidget(button)
        self.find_field.returnPressed.connect(self._find)
        self.find_bar.hide()
        layout.addWidget(self.find_bar)
        controls = QHBoxLayout()
        self.line_mode = QCheckBox("Edit command locally")
        self.line_mode.setToolTip("Off: send keys immediately, including Tab and Ctrl+C.")
        self.command_line = QLineEdit()
        self.command_line.setPlaceholderText("Command (Enter to send)")
        self.command_line.hide()
        controls.addWidget(self.line_mode)
        controls.addWidget(self.command_line, 1)
        self.renderer_label = QLabel("Native terminal")
        controls.addWidget(self.renderer_label)
        controls.addStretch(1)
        for label, callback in (("Find", self._open_find), ("Copy", self._copy), ("Paste", self._paste)):
            button = QPushButton(label)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _=False, fn=callback: fn())
            controls.addWidget(button)
        layout.addLayout(controls)
        self.line_mode.toggled.connect(self._set_line_mode)
        self.command_line.returnPressed.connect(self._submit_line)
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QLabel#terminalStatus { background: #15312c; color: #b9d8ce; }
            QPlainTextEdit {
                background: #0d1715; color: #d8e7df; border: none;
                selection-background-color: #2a7768; padding: 8px;
            }
            """
        )

        self.output_received.connect(self._consume_output)
        self.exit_received.connect(self._mark_exited)
        self.terminal_view.input_bytes.connect(self._send_live)
        self.terminal_view.terminal_resized.connect(self._resize)
        self.terminal_view.history_page.connect(self._page_history)
        self.terminal_view.selection_finished.connect(self._render)
        self.terminal_view.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled)
        self.terminal_view.setFocus()
        QShortcut(QKeySequence("Ctrl+Shift+F"), self).activated.connect(self._open_find)
        QShortcut(QKeySequence(QKeySequence.StandardKey.ZoomIn), self).activated.connect(lambda: self._zoom(1))
        QShortcut(QKeySequence(QKeySequence.StandardKey.ZoomOut), self).activated.connect(lambda: self._zoom(-1))
        QShortcut(QKeySequence("Ctrl+0"), self).activated.connect(lambda: self._zoom(-self._zoom_steps))
        if sys.platform == "darwin":
            QShortcut(QKeySequence(QKeySequence.StandardKey.Find), self).activated.connect(self._open_find)
        QShortcut(QKeySequence("Esc"), self.find_field,
                  context=Qt.ShortcutContext.WidgetShortcut).activated.connect(self._hide_find)
        if rich and os.environ.get("SHOOKLINK_TERMINAL_RENDERER", "").lower() != "native":
            try:
                from shooklink.ui.web_terminal import WebTerminal
                web = WebTerminal(self)
            except (ImportError, OSError, RuntimeError) as error:
                self.renderer_label.setToolTip(str(error))
                logger.info("Using native terminal: %s", error)
            else:
                self._web = web
                self.terminal_view.hide()
                layout.insertWidget(1, web, 1)
                self.renderer_label.setText("Starting xterm.js")
                web.input_bytes.connect(self._send_live)
                web.terminal_resized.connect(self._resize_remote)
                web.ready.connect(self._web_ready)
                web.failed.connect(self._web_failed)
                web.search_result.connect(self._show_match)
                web.find_shortcut.connect(self._open_find)
                web.copy_shortcut.connect(self._copy)
                web.paste_shortcut.connect(self._paste)

    def _zoom(self, change):
        self._zoom_steps = max(-5, min(12, self._zoom_steps + change))
        if self._web:
            self._web.set_font_size(15 + self._zoom_steps)
        else:
            font = self.terminal_view.font()
            font.setPointSize(12 + self._zoom_steps)
            self.terminal_view.setFont(font)
            self.terminal_view.sync_size()

    def _web_ready(self):
        self.renderer_label.setText("xterm.js")
        self._web.set_font_size(15 + self._zoom_steps)
        if not self.line_mode.isChecked() and not self.find_bar.isVisible():
            self._web.focus_terminal()

    def _web_failed(self, reason: str, pending: bytes):
        web, self._web = self._web, None
        if web is None:
            return
        logger.warning("Falling back to native terminal: %s", reason)
        web.hide()
        web.shutdown()
        web.deleteLater()
        self.renderer_label.setText("Native fallback")
        self.renderer_label.setToolTip(reason)
        self.terminal_view.show()
        self._set_line_mode(self.line_mode.isChecked())
        if pending:
            self._consume_output(pending)

    def _focus_terminal(self):
        if self.line_mode.isChecked():
            self.command_line.setFocus()
        elif self._web:
            self._web.focus_terminal()
        else:
            self.terminal_view.setFocus()

    def _open_find(self):
        self.find_bar.show()
        self.find_field.setFocus()
        self.find_field.selectAll()

    def _hide_find(self):
        self.find_bar.hide()
        self.find_result.clear()
        if self._web:
            self._web.find_text("")
        self._focus_terminal()

    def _show_match(self, matched):
        self.find_result.setText("" if matched or not self.find_field.text() else "No match")

    def _find(self, previous=False):
        query = self.find_field.text()
        if self._web:
            self._web.find_text(query, previous)
        elif query:
            flags = QTextDocument.FindFlag.FindBackward if previous else QTextDocument.FindFlag(0)
            if not self.terminal_view.find(query, flags):
                self.terminal_view.moveCursor(QTextCursor.MoveOperation.End if previous else QTextCursor.MoveOperation.Start)
                self._show_match(self.terminal_view.find(query, flags))
            else:
                self._show_match(True)

    def _copy(self):
        if self._web:
            self._web.copy_selection()
        else:
            self.terminal_view.copy()

    def _paste(self):
        if self.line_mode.isChecked():
            self.command_line.paste()
        elif self._web:
            self._web.paste_text(QApplication.clipboard().text())
        else:
            self.terminal_view.paste_to_remote()
        self._focus_terminal()

    def _set_line_mode(self, enabled: bool) -> None:
        self.command_line.setVisible(enabled)
        self.terminal_view.setFocusPolicy(Qt.FocusPolicy.NoFocus if enabled else Qt.FocusPolicy.StrongFocus)
        self._focus_terminal()

    def _send_live(self, data: bytes) -> None:
        if self._connected and not self._remote_exited and not self.line_mode.isChecked():
            self._send_input(data)

    def _submit_line(self) -> None:
        if not self._connected or self._remote_exited:
            return
        data = self.command_line.text().encode("utf-8") + b"\r"
        self._send_input(data)
        self.command_line.clear()

    def feed_output(self, data: bytes) -> None:
        if isinstance(data, bytes):
            self.output_received.emit(data)

    def set_requesting(self) -> None:
        self._connected = False
        self.status_label.setText("Requesting remote shell")

    def set_connected(self) -> None:
        self._connected = True
        self.status_label.setText("Remote shell connected")

    def mark_exited(self, exit_code: int | None, reason: str = "") -> None:
        self.exit_received.emit(exit_code, reason)

    def _consume_output(self, data: bytes) -> None:
        if self._web:
            self._web.feed_output(data)
            return
        self._stream.feed(data)
        if not self._frame_timer.isActive():
            self._frame_timer.start()

    def _resize(self, columns: int, rows: int) -> None:
        if self._web:
            return
        self.screen.resize(lines=rows, columns=columns)
        self._resize_remote(columns, rows)
        self._render()

    def _page_history(self, direction: int) -> None:
        if direction < 0:
            self.screen.prev_page()
        else:
            self.screen.next_page()
        self._render()

    def _render(self) -> None:
        view, screen = self.terminal_view, self.screen
        if view.selecting or view.textCursor().hasSelection():
            return  # Keep copied text stable; the VT parser continues to consume output.
        document = view.document()
        if document.blockCount() != screen.lines:
            view.setPlainText("\n" * (screen.lines - 1))
            self._rows.clear()
            screen.dirty.update(range(screen.lines))
        edit = QTextCursor(document)
        edit.beginEditBlock()
        for y in sorted(screen.dirty):
            if not 0 <= y < screen.lines:
                continue
            line = screen.buffer[y]
            cells = tuple(line[x] for x in range(screen.columns))
            if cells == self._rows.get(y):
                continue
            self._rows[y] = cells
            block = document.findBlockByNumber(y)
            edit.setPosition(block.position())
            edit.setPosition(block.position() + block.length() - 1, QTextCursor.MoveMode.KeepAnchor)
            for _, run in groupby(cells, key=lambda cell: cell[1:]):
                first = next(run)
                text = first.data + "".join(cell.data for cell in run)
                if text:
                    edit.insertText(text, _character_format(first) or QTextCharFormat())
        edit.endEditBlock()
        screen.dirty.clear()
        y = min(screen.cursor.y, screen.lines - 1)
        block = document.findBlockByNumber(y)
        cells = self._rows.get(y, ())
        # VT positions count display cells; Qt positions count UTF-16 units.
        prefix = "".join(cell.data for cell in cells[:screen.cursor.x])
        caret = QTextCursor(block)
        caret.setPosition(block.position() + min(len(prefix.encode("utf-16-le")) // 2, block.length() - 1))
        view.setTextCursor(caret)
        marker = QTextEdit.ExtraSelection()
        marker.cursor = QTextCursor(caret)
        marker.cursor.movePosition(QTextCursor.MoveOperation.NextCharacter, QTextCursor.MoveMode.KeepAnchor)
        marker.format.setBackground(QColor("#b9d8ce"))
        marker.format.setForeground(QColor("#15312c"))
        view.setExtraSelections([] if screen.cursor.hidden else [marker])

    def _mark_exited(self, exit_code: int | None, reason: str) -> None:
        self._remote_exited = True
        self.line_mode.setEnabled(False)
        self.command_line.setEnabled(False)
        suffix = "" if exit_code is None else f" ({exit_code})"
        self.status_label.setText({
            "permission": "Denied: enable Allow Remote Shell on the computer running the shell.",
            "timeout": "No shell reply. Check the peer connection and Allow Remote Shell, then try again.",
            "busy": "Denied: another remote shell is active.",
            "start_failed": "The remote terminal process could not start.",
        }.get(reason, f"Remote shell exited{suffix}"))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._web:
            web, self._web = self._web, None
            web.shutdown()
        if not self._remote_exited:
            self._remote_exited = True
            self._close_remote()
        super().closeEvent(event)


def _character_format(cell) -> QTextCharFormat | None:
    if not any(
        (
            cell.fg != "default",
            cell.bg != "default",
            cell.bold,
            cell.italics,
            cell.underscore,
            cell.strikethrough,
            cell.reverse,
        )
    ):
        return None
    foreground = _terminal_color(cell.fg, "#d8e7df")
    background = _terminal_color(cell.bg, "#0d1715")
    if cell.reverse:
        foreground, background = background, foreground
    character_format = QTextCharFormat()
    character_format.setForeground(foreground)
    character_format.setBackground(background)
    character_format.setFontWeight(QFont.Weight.Bold if cell.bold else QFont.Weight.Normal)
    character_format.setFontItalic(cell.italics)
    character_format.setFontUnderline(cell.underscore)
    character_format.setFontStrikeOut(cell.strikethrough)
    return character_format


def _terminal_color(value: str, fallback: str) -> QColor:
    if value == "default":
        return QColor(fallback)
    if value in _ANSI_COLORS:
        return QColor(_ANSI_COLORS[value])
    if len(value) == 6 and all(character in "0123456789abcdef" for character in value):
        return QColor(f"#{value}")
    return QColor(fallback)


__all__ = [
    "MAX_TERMINAL_INPUT_CHUNK",
    "TerminalView",
    "TerminalWindow",
]
