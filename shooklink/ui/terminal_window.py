"""ANSI terminal popup for remote PTY and ConPTY sessions."""

from __future__ import annotations

import sys
from collections.abc import Callable

import pyte
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QContextMenuEvent,
    QFont,
    QKeyEvent,
    QKeySequence,
    QResizeEvent,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByKeyboard
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        font = QFont("Menlo" if sys.platform == "darwin" else "Consolas", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self._last_size: tuple[int, int] | None = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        modifiers = event.modifiers()
        control = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        alt = bool(modifiers & Qt.KeyboardModifier.AltModifier)

        if control and event.key() == Qt.Key.Key_C:
            if shift:
                self.copy()
            else:
                self.input_bytes.emit(b"\x03")
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
            self.input_bytes.emit(bytes((event.key() - Qt.Key.Key_A + 1,)))
            event.accept()
            return

        payload = _KEY_SEQUENCES.get(event.key())
        if payload is None and event.text():
            payload = event.text().encode("utf-8")
        if payload is not None:
            if alt:
                payload = b"\x1b" + payload
            self.input_bytes.emit(payload)
            event.accept()
            return
        event.ignore()

    def paste_to_remote(self) -> None:
        text = QApplication.clipboard().text()
        if text:
            encoded = text.encode("utf-8")
            for offset in range(0, len(encoded), MAX_TERMINAL_INPUT_CHUNK):
                self.input_bytes.emit(
                    encoded[offset : offset + MAX_TERMINAL_INPUT_CHUNK]
                )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        metrics = self.fontMetrics()
        cell_width = max(1, metrics.horizontalAdvance("M"))
        cell_height = max(1, metrics.lineSpacing())
        columns = max(2, self.viewport().width() // cell_width)
        rows = max(2, self.viewport().height() // cell_height)
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
    exit_received = Signal(object)

    def __init__(
        self,
        session_id: str,
        send_input: Callable[[bytes], None],
        resize_remote: Callable[[int, int], None],
        close_remote: Callable[[], None],
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self._send_input = send_input
        self._resize_remote = resize_remote
        self._close_remote = close_remote
        self._remote_exited = False
        self.screen = pyte.HistoryScreen(80, 24, history=10_000)
        self._stream = pyte.ByteStream(self.screen)
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
        self.terminal_view.input_bytes.connect(self._send_input)
        self.terminal_view.terminal_resized.connect(self._resize)
        self.terminal_view.history_page.connect(self._page_history)

    def feed_output(self, data: bytes) -> None:
        if isinstance(data, bytes):
            self.output_received.emit(data)

    def mark_exited(self, exit_code: int | None) -> None:
        self.exit_received.emit(exit_code)

    def _consume_output(self, data: bytes) -> None:
        self._stream.feed(data)
        self._render()

    def _resize(self, columns: int, rows: int) -> None:
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
        self.terminal_view.setPlainText("\n".join(self.screen.display))
        document = self.terminal_view.document()
        for row, cells in self.screen.buffer.items():
            block = document.findBlockByNumber(row)
            if not block.isValid():
                continue
            for column, cell in cells.items():
                if column >= self.screen.columns:
                    continue
                character_format = _character_format(cell)
                if character_format is None:
                    continue
                cursor = QTextCursor(document)
                cursor.setPosition(block.position() + column)
                cursor.movePosition(
                    QTextCursor.MoveOperation.Right,
                    QTextCursor.MoveMode.KeepAnchor,
                    1,
                )
                cursor.mergeCharFormat(character_format)
        cursor_row = min(self.screen.cursor.y, max(0, document.blockCount() - 1))
        block = document.findBlockByNumber(cursor_row)
        if block.isValid():
            cursor = QTextCursor(block)
            cursor.movePosition(
                QTextCursor.MoveOperation.Right,
                QTextCursor.MoveMode.MoveAnchor,
                min(self.screen.cursor.x, max(0, block.length() - 1)),
            )
            self.terminal_view.setTextCursor(cursor)

    def _mark_exited(self, exit_code: int | None) -> None:
        self._remote_exited = True
        suffix = "" if exit_code is None else f" ({exit_code})"
        self.status_label.setText(f"Remote shell exited{suffix}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._remote_exited:
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
