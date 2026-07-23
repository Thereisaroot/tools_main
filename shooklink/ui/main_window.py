"""Primary ShookLink application window."""

from __future__ import annotations

import sys
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from serial.tools import list_ports

from shooklink.chat.service import ChatMessage, ChatService
from shooklink.core import CoreSnapshot, CoreState
from shooklink.files.service import FileProgress
from shooklink.input.backend import PermissionStatus
from shooklink.input.service import InputSessionState, InputStateChange
from shooklink.input.topology import Side
from shooklink.shell.service import ShellOutput, ShellState
from shooklink.ui.terminal_window import TerminalWindow

COMMON_BAUD_RATES = (
    115_200,
    230_400,
    460_800,
    921_600,
    1_000_000,
    1_500_000,
    2_000_000,
)


class FileUiService(Protocol):
    download_dir: Path

    def send_file(self, path: str | Path) -> Future[str]: ...

    def cancel(self, transfer_id: str) -> None: ...

    def add_progress_listener(self, listener) -> None: ...

    def remove_progress_listener(self, listener) -> None: ...


class ShellUiService(Protocol):
    allow_remote_shell: bool

    def add_output_listener(self, listener) -> None: ...

    def remove_output_listener(self, listener) -> None: ...

    def add_state_listener(self, listener) -> None: ...

    def remove_state_listener(self, listener) -> None: ...

    def add_emergency_listener(self, listener) -> None: ...

    def remove_emergency_listener(self, listener) -> None: ...

    def set_allow_remote_shell(self, allowed: bool) -> None: ...

    def open_remote(self, *, columns: int, rows: int) -> str: ...

    def send_input(self, session_id: str, data: bytes) -> None: ...

    def resize(self, session_id: str, columns: int, rows: int) -> None: ...

    def close_session(self, session_id: str) -> None: ...


class InputUiService(Protocol):
    state: InputSessionState
    peer_side: Side
    auto_edge_enabled: bool
    allow_remote_input: bool

    def permission_status(self) -> PermissionStatus: ...

    def add_state_listener(self, listener) -> None: ...

    def remove_state_listener(self, listener) -> None: ...

    def set_allow_remote_input(self, allowed: bool) -> None: ...

    def set_peer_side(self, side: Side | str) -> None: ...

    def set_auto_edge_enabled(self, enabled: bool) -> None: ...

    def request_control(self) -> str: ...

    def stop_control(self, *, reason: str = "manual") -> None: ...

    def connection_changed(self, connected: bool) -> None: ...


class FileDropZone(QFrame):
    """Drop target that accepts one existing local file URL."""

    file_dropped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("fileDropZone")
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        label = QLabel("DROP A FILE HERE")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    @staticmethod
    def local_file_path(mime_data) -> Path | None:
        if not mime_data.hasUrls():
            return None
        urls = mime_data.urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = Path(urls[0].toLocalFile())
        return path if path.is_file() else None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self.local_file_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        path = self.local_file_path(event.mimeData())
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.file_dropped.emit(str(path))


class MainWindow(QMainWindow):
    """Connection shell and chat interface shared by macOS and Windows."""

    connect_requested = Signal(str, int)
    disconnect_requested = Signal()
    trust_requested = Signal(int, str)
    emergency_exit_requested = Signal()
    preferences_changed = Signal()
    incoming_message = Signal(object)
    file_progress = Signal(object)
    file_prepared = Signal(object)
    shell_output = Signal(object)
    shell_state = Signal(object)
    input_state = Signal(object)

    def __init__(
        self,
        chat_service: ChatService,
        file_service: FileUiService | None = None,
        shell_service: ShellUiService | None = None,
        input_service: InputUiService | None = None,
    ) -> None:
        super().__init__()
        self._chat_service = chat_service
        self._file_service = file_service
        self._shell_service = shell_service
        self._input_service = input_service
        self._shortcuts: list[QShortcut] = []
        self._connected = False
        self._connecting = False
        self._connection_id = 0
        self._peer_fingerprint: str | None = None
        self._secure_feature_available = True
        self._input_feature_available = input_service is not None
        self._active_transfer_id: str | None = None
        self._file_transfers: OrderedDict[str, FileProgress] = OrderedDict()
        self._active_shell_session: str | None = None
        self.terminal_window: TerminalWindow | None = None
        self._chat_listener = self.incoming_message.emit
        self._file_listener = self.file_progress.emit
        self._shell_output_listener = self.shell_output.emit
        self._shell_state_listener = self.shell_state.emit
        self._input_state_listener = self.input_state.emit
        self._input_emergency_listener = self._show_input_emergency
        self.setWindowTitle("ShookLink")
        self.setMinimumSize(760, 640)
        self.resize(920, 760)
        self._build_ui()
        self._install_shortcuts()
        self._refresh_ports()
        self.incoming_message.connect(self._show_received_message)
        self.file_progress.connect(self._show_file_progress)
        self.file_prepared.connect(self._file_was_prepared)
        self.shell_output.connect(self._show_shell_output)
        self.shell_state.connect(self._show_shell_state)
        self.input_state.connect(self._show_input_state)
        self._chat_service.add_message_listener(self._chat_listener)
        if self._file_service is not None:
            self._file_service.add_progress_listener(self._file_listener)
        if self._shell_service is not None:
            self._shell_service.add_output_listener(self._shell_output_listener)
            self._shell_service.add_state_listener(self._shell_state_listener)
        if self._input_service is not None:
            self._input_service.add_state_listener(self._input_state_listener)
            self._input_service.add_emergency_listener(
                self._input_emergency_listener
            )

    def _build_ui(self) -> None:
        root = QWidget(self)
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(18)

        heading_row = QHBoxLayout()
        title = QLabel("SHOOKLINK")
        title.setObjectName("title")
        subtitle = QLabel("SERIAL PEER LINK")
        subtitle.setObjectName("subtitle")
        heading_row.addWidget(title)
        heading_row.addSpacing(12)
        heading_row.addWidget(subtitle)
        heading_row.addItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setObjectName("connectionStatus")
        heading_row.addWidget(self.connection_status)
        layout.addLayout(heading_row)

        connection = QFrame()
        connection.setObjectName("panel")
        connection_layout = QGridLayout(connection)
        connection_layout.setContentsMargins(18, 16, 18, 16)
        connection_layout.setHorizontalSpacing(12)
        connection_layout.addWidget(QLabel("SERIAL PORT"), 0, 0)
        connection_layout.addWidget(QLabel("BAUD"), 0, 1)
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.baud_combo = QComboBox()
        self.baud_combo.setEditable(True)
        self.baud_combo.addItems([str(rate) for rate in COMMON_BAUD_RATES])
        self.baud_combo.setCurrentText("115200")
        self.connect_button = QPushButton("Connect")
        self.connect_button.setObjectName("connectButton")
        self.connect_button.clicked.connect(self._request_connection)
        connection_layout.addWidget(self.port_combo, 1, 0)
        connection_layout.addWidget(self.baud_combo, 1, 1)
        connection_layout.addWidget(self.connect_button, 1, 2)
        self.peer_fingerprint = QLabel("No authenticated peer")
        self.peer_fingerprint.setObjectName("peerFingerprint")
        self.trust_button = QPushButton("Trust Peer")
        self.trust_button.clicked.connect(self._request_trust)
        self.trust_button.setEnabled(False)
        connection_layout.addWidget(self.peer_fingerprint, 2, 0, 1, 2)
        connection_layout.addWidget(self.trust_button, 2, 2)
        connection_layout.setColumnStretch(0, 3)
        connection_layout.setColumnStretch(1, 1)
        layout.addWidget(connection)

        file_panel = QFrame()
        file_panel.setObjectName("panel")
        file_layout = QGridLayout(file_panel)
        file_layout.setContentsMargins(18, 14, 18, 14)
        file_layout.setHorizontalSpacing(10)
        self.file_drop_zone = FileDropZone()
        self.file_drop_zone.file_dropped.connect(self._queue_file)
        self.file_select_button = QPushButton("Choose File")
        self.file_select_button.clicked.connect(self._select_file)
        self.file_cancel_button = QPushButton("Cancel Transfer")
        self.file_cancel_button.clicked.connect(self._cancel_file)
        self.open_download_button = QPushButton("Open Downloads")
        self.open_download_button.clicked.connect(self._open_download_folder)
        self.file_progress_bar = QProgressBar()
        self.file_progress_bar.setRange(0, 100)
        self.file_progress_bar.setValue(0)
        self.file_progress_bar.setTextVisible(False)
        self.file_progress_label = QLabel("No active file transfer")
        self.file_progress_label.setObjectName("fileProgress")
        file_layout.addWidget(self.file_drop_zone, 0, 0, 2, 1)
        file_layout.addWidget(self.file_select_button, 0, 1)
        file_layout.addWidget(self.file_cancel_button, 0, 2)
        file_layout.addWidget(self.open_download_button, 0, 3)
        file_layout.addWidget(self.file_progress_bar, 1, 1, 1, 2)
        file_layout.addWidget(self.file_progress_label, 1, 3)
        file_layout.setColumnStretch(0, 2)
        file_layout.setColumnStretch(3, 1)
        files_enabled = self._file_service is not None
        self.file_drop_zone.setEnabled(files_enabled)
        self.file_select_button.setEnabled(files_enabled)
        self.file_cancel_button.setEnabled(False)
        self.open_download_button.setEnabled(files_enabled)
        layout.addWidget(file_panel)

        input_panel = QFrame()
        input_panel.setObjectName("panel")
        input_layout = QGridLayout(input_panel)
        input_layout.setContentsMargins(18, 13, 18, 13)
        input_layout.setHorizontalSpacing(10)
        input_layout.setVerticalSpacing(8)
        input_title = QLabel("INPUT SHARE")
        input_title.setObjectName("sectionLabel")
        self.allow_input_checkbox = QCheckBox("Allow Remote Input")
        peer_side_label = QLabel("PEER SIDE")
        self.peer_side_combo = QComboBox()
        self.peer_side_combo.addItems(["Right", "Left", "Top", "Bottom"])
        self.auto_edge_checkbox = QCheckBox("Auto Edge (This Computer -> Peer)")
        self.auto_edge_checkbox.setToolTip(
            "Enable this only on the computer whose mouse starts control. "
            "Return works automatically without enabling Auto Edge on the peer."
        )
        self.toggle_input_button = QPushButton("Toggle Remote Control")
        self.toggle_input_button.clicked.connect(self._toggle_input_control)
        self.input_status = QLabel("Idle")
        self.input_status.setObjectName("inputStatus")
        self.input_permission_status = QLabel("Unavailable")
        self.input_permission_status.setObjectName("inputPermissionStatus")
        self.input_emergency_help = QLabel(
            "Ctrl+Alt+Shift+Backspace: emergency stop · Ctrl+Alt+Shift+Escape: exit"
        )
        self.input_emergency_help.setObjectName("inputEmergencyHelp")
        input_layout.addWidget(input_title, 0, 0)
        input_layout.addWidget(self.allow_input_checkbox, 0, 1)
        input_layout.addWidget(peer_side_label, 0, 2)
        input_layout.addWidget(self.peer_side_combo, 0, 3)
        input_layout.addWidget(self.auto_edge_checkbox, 0, 4)
        input_layout.addWidget(self.input_status, 0, 5)
        input_layout.addWidget(self.toggle_input_button, 0, 6)
        input_layout.addWidget(self.input_permission_status, 1, 0, 1, 2)
        input_layout.addWidget(self.input_emergency_help, 1, 2, 1, 5)
        input_layout.setColumnStretch(5, 1)
        input_enabled = self._input_service is not None
        self.allow_input_checkbox.setEnabled(input_enabled)
        self.peer_side_combo.setEnabled(input_enabled)
        self.auto_edge_checkbox.setEnabled(input_enabled)
        self.toggle_input_button.setEnabled(input_enabled)
        if self._input_service is not None:
            self.allow_input_checkbox.setChecked(
                self._input_service.allow_remote_input
            )
            self.peer_side_combo.setCurrentText(
                self._input_service.peer_side.value.title()
            )
            self.auto_edge_checkbox.setChecked(
                self._input_service.auto_edge_enabled
            )
            try:
                permission = self._input_service.permission_status()
                self.input_permission_status.setText(permission.detail)
            except Exception as error:
                self.input_permission_status.setText(str(error))
            self._show_input_state(InputStateChange(self._input_service.state))
        self.allow_input_checkbox.toggled.connect(self._set_input_permission)
        self.peer_side_combo.currentTextChanged.connect(self._set_input_side)
        self.auto_edge_checkbox.toggled.connect(self._set_auto_edge)
        layout.addWidget(input_panel)

        shell_panel = QFrame()
        shell_panel.setObjectName("panel")
        shell_layout = QHBoxLayout(shell_panel)
        shell_layout.setContentsMargins(18, 13, 18, 13)
        shell_layout.setSpacing(10)
        shell_title = QLabel("REMOTE SHELL")
        shell_title.setObjectName("sectionLabel")
        self.allow_shell_checkbox = QCheckBox("Allow Remote Shell")
        self.open_shell_button = QPushButton("Open Remote Shell")
        self.open_shell_button.clicked.connect(self._open_remote_shell)
        self.terminate_shell_button = QPushButton("Terminate Session")
        self.terminate_shell_button.clicked.connect(self._terminate_shell)
        self.shell_status = QLabel("Disabled")
        self.shell_status.setObjectName("shellStatus")
        shell_layout.addWidget(shell_title)
        shell_layout.addWidget(self.allow_shell_checkbox)
        shell_layout.addStretch(1)
        shell_layout.addWidget(self.shell_status)
        shell_layout.addWidget(self.open_shell_button)
        shell_layout.addWidget(self.terminate_shell_button)
        shell_enabled = self._shell_service is not None
        self.allow_shell_checkbox.setEnabled(shell_enabled)
        self.open_shell_button.setEnabled(shell_enabled)
        self.terminate_shell_button.setEnabled(False)
        if self._shell_service is not None:
            self.allow_shell_checkbox.setChecked(
                self._shell_service.allow_remote_shell
            )
            self.shell_status.setText(
                "Armed" if self._shell_service.allow_remote_shell else "Disabled"
            )
        self.allow_shell_checkbox.toggled.connect(self._set_shell_permission)
        layout.addWidget(shell_panel)

        received_label = QLabel("LAST RECEIVED")
        received_label.setObjectName("sectionLabel")
        layout.addWidget(received_label)
        self.received_view = QPlainTextEdit()
        self.received_view.setObjectName("receivedView")
        self.received_view.setReadOnly(True)
        self.received_view.setPlaceholderText("Incoming text appears here")
        self.received_view.setMinimumHeight(145)
        layout.addWidget(self.received_view, 1)

        copy_row = QHBoxLayout()
        self.received_kind = QLabel("No message")
        self.received_kind.setObjectName("messageKind")
        copy_row.addWidget(self.received_kind)
        copy_row.addStretch(1)
        self.copy_last_button = QPushButton("Copy Last Message")
        self.copy_secure_button = QPushButton("Copy Last Secure Message")
        self.copy_last_button.clicked.connect(self._copy_last)
        self.copy_secure_button.clicked.connect(self._copy_last_secure)
        copy_row.addWidget(self.copy_last_button)
        copy_row.addWidget(self.copy_secure_button)
        layout.addLayout(copy_row)

        editor_label = QLabel("MESSAGE")
        editor_label.setObjectName("sectionLabel")
        layout.addWidget(editor_label)
        self.message_editor = QPlainTextEdit()
        self.message_editor.setObjectName("messageEditor")
        self.message_editor.setPlaceholderText("Write or paste text")
        self.message_editor.setMinimumHeight(180)
        self.message_editor.setTabChangesFocus(False)
        layout.addWidget(self.message_editor, 2)

        send_row = QHBoxLayout()
        self.action_status = QLabel("")
        self.action_status.setObjectName("actionStatus")
        send_row.addWidget(self.action_status)
        send_row.addStretch(1)
        self.send_plain_button = QPushButton("Send Plain")
        self.send_plain_button.setObjectName("plainButton")
        self.send_secure_button = QPushButton("Send Secure")
        self.send_secure_button.setObjectName("secureButton")
        self.send_secure_button.setEnabled(self._chat_service.secure_available)
        self.send_plain_button.clicked.connect(self._send_plain)
        self.send_secure_button.clicked.connect(self._send_secure)
        send_row.addWidget(self.send_plain_button)
        send_row.addWidget(self.send_secure_button)
        layout.addLayout(send_row)

        self.setCentralWidget(root)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#root { background: #f2efe7; color: #172421; }
            QWidget#root QLabel, QWidget#root QCheckBox { color: #172421; }
            QLabel#title { color: #0b3d36; font-size: 23px; font-weight: 800; }
            QLabel#subtitle { color: #b4482b; font-size: 11px; font-weight: 700; }
            QLabel#sectionLabel, QFrame#panel QLabel {
                color: #4f625d; font-size: 10px; font-weight: 700;
            }
            QLabel#connectionStatus {
                background: #d9dfd4; color: #27443e; border-radius: 10px;
                padding: 5px 11px; font-weight: 700;
            }
            QFrame#panel {
                background: #e7e3d9; border: 1px solid #c8c1b2; border-radius: 8px;
            }
            QFrame#fileDropZone {
                background: #dce6df; border: 1px dashed #6f8c82; border-radius: 6px;
            }
            QFrame#fileDropZone QLabel { color: #31564e; font-size: 10px; font-weight: 750; }
            QComboBox, QPlainTextEdit {
                background: #fffdf8; color: #172421;
                border: 1px solid #b9b2a5; border-radius: 6px;
                selection-background-color: #176b5b; selection-color: white;
            }
            QComboBox { min-height: 31px; padding: 0 9px; }
            QComboBox QLineEdit { color: #172421; background: transparent; }
            QComboBox QAbstractItemView {
                background: #fffdf8; color: #172421;
                selection-background-color: #176b5b; selection-color: white;
            }
            QPlainTextEdit { padding: 12px; font-size: 14px; }
            QPlainTextEdit#receivedView { background: #172421; color: #e9f0e8; }
            QPushButton {
                min-height: 31px; padding: 0 14px; border: 1px solid #8f988f;
                border-radius: 6px; background: #f8f5ed; color: #172421;
                font-weight: 650;
            }
            QPushButton:hover { background: #ebe5d9; }
            QPushButton:disabled { color: #9b9a93; background: #e1ded6; }
            QPushButton#connectButton, QPushButton#plainButton {
                background: #176b5b; color: white; border-color: #176b5b;
            }
            QPushButton#secureButton {
                background: #b4482b; color: white; border-color: #b4482b;
            }
            QLabel#messageKind, QLabel#actionStatus { color: #66756f; }
            QLabel#peerFingerprint { color: #526761; font-size: 11px; }
            QLabel#fileProgress { color: #526761; font-size: 11px; }
            QLabel#shellStatus { color: #526761; font-size: 11px; }
            QLabel#inputStatus, QLabel#inputPermissionStatus, QLabel#inputEmergencyHelp {
                color: #526761; font-size: 11px;
            }
            QProgressBar {
                min-height: 8px; max-height: 8px; border: none; border-radius: 4px;
                background: #cbc7bd;
            }
            QProgressBar::chunk { background: #176b5b; border-radius: 4px; }
            """
        )
        fixed_font = QFont("Menlo" if sys.platform == "darwin" else "Consolas")
        fixed_font.setStyleHint(QFont.StyleHint.Monospace)
        self.message_editor.setFont(fixed_font)
        self.received_view.setFont(fixed_font)

    def _install_shortcuts(self) -> None:
        self._add_shortcut("Ctrl+Alt+V", self._send_plain)
        self._add_shortcut("Ctrl+Shift+Alt+V", self._send_secure)
        self._add_shortcut("Ctrl+Alt+C", self._copy_last)
        self._add_shortcut("Ctrl+Shift+Alt+C", self._copy_last_secure)
        if sys.platform == "darwin":
            self._add_shortcut("Ctrl+V", self.message_editor.paste, self.message_editor)
            self._add_shortcut("Ctrl+C", self.message_editor.copy, self.message_editor)
            self._add_shortcut("Ctrl+A", self.message_editor.selectAll, self.message_editor)

    def _add_shortcut(self, sequence: str, callback, parent=None) -> None:
        shortcut = QShortcut(QKeySequence(sequence), parent or self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)

    def _refresh_ports(self) -> None:
        current = self.port_combo.currentText()
        ports = [port.device for port in list_ports.comports()]
        self.port_combo.clear()
        self.port_combo.addItems(ports)
        if current:
            self.port_combo.setCurrentText(current)

    def _request_connection(self) -> None:
        if self._connected or self._connecting:
            self.disconnect_requested.emit()
            return
        port = self.port_combo.currentText().strip()
        try:
            baud = int(self.baud_combo.currentText().strip())
        except ValueError:
            self.action_status.setText("Baud must be an integer")
            return
        if not port:
            self.action_status.setText("Select a serial port")
            return
        self.action_status.clear()
        self.connect_requested.emit(port, baud)

    def _request_trust(self) -> None:
        if self._connection_id and self._peer_fingerprint is not None:
            self.trust_requested.emit(
                self._connection_id,
                self._peer_fingerprint,
            )

    def _select_file(self) -> None:
        if self._file_service is None:
            return
        path, _selected_filter = QFileDialog.getOpenFileName(self, "Choose a file")
        if path:
            self._queue_file(path)

    def _queue_file(self, path: str) -> None:
        if self._file_service is None:
            return
        try:
            future = self._file_service.send_file(path)
        except Exception as error:
            self.action_status.setText(str(error))
            return
        self.file_progress_label.setText(f"Preparing {Path(path).name}")
        future.add_done_callback(self.file_prepared.emit)

    def _file_was_prepared(self, future: Future[str]) -> None:
        try:
            self._active_transfer_id = future.result()
        except Exception as error:
            self.file_progress_label.setText(str(error))
            self.file_cancel_button.setEnabled(False)
            return
        self.file_cancel_button.setEnabled(True)

    def _cancel_file(self) -> None:
        if self._file_service is None or self._active_transfer_id is None:
            return
        transfer_id = self._active_transfer_id
        self._file_service.cancel(transfer_id)
        if self._active_transfer_id == transfer_id:
            remaining = [
                item
                for item in self._file_transfers.values()
                if item.transfer_id != transfer_id
                and item.state not in {"complete", "failed", "cancelled"}
            ]
            if remaining:
                replacement = remaining[-1]
                self._active_transfer_id = replacement.transfer_id
                self._render_file_progress(replacement)
            else:
                self._active_transfer_id = None
                self.file_progress_label.setText("Cancellation requested")
                self.file_cancel_button.setEnabled(False)

    def _open_download_folder(self) -> None:
        if self._file_service is None:
            return
        self._file_service.download_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self._file_service.download_dir.resolve()))
        )

    def _set_shell_permission(self, allowed: bool) -> None:
        if self._shell_service is None:
            return
        self._shell_service.set_allow_remote_shell(allowed)
        if self._active_shell_session is None:
            self.shell_status.setText("Armed" if allowed else "Disabled")
        self.preferences_changed.emit()

    def _set_input_permission(self, allowed: bool) -> None:
        if self._input_service is None:
            return
        try:
            self._input_service.set_allow_remote_input(allowed)
        except Exception as error:
            self.input_status.setText(str(error))
            return
        self.preferences_changed.emit()

    def _set_input_side(self, label: str) -> None:
        if self._input_service is None:
            return
        try:
            self._input_service.set_peer_side(label.lower())
        except Exception as error:
            self.input_status.setText(str(error))
            self.peer_side_combo.setCurrentText(
                self._input_service.peer_side.value.title()
            )
            return
        self.preferences_changed.emit()

    def _set_auto_edge(self, enabled: bool) -> None:
        if self._input_service is None:
            return
        try:
            self._input_service.set_auto_edge_enabled(enabled)
        except Exception as error:
            self.input_status.setText(str(error))
            self.auto_edge_checkbox.setChecked(
                self._input_service.auto_edge_enabled
            )
            return
        self.preferences_changed.emit()

    def _toggle_input_control(self) -> None:
        if self._input_service is None:
            return
        try:
            if self._input_service.state is InputSessionState.IDLE:
                self._input_service.request_control()
            else:
                self._input_service.stop_control(reason="manual")
        except Exception as error:
            self.input_status.setText(str(error))

    def _show_input_state(self, change: InputStateChange) -> None:
        labels = {
            InputSessionState.IDLE: "Idle",
            InputSessionState.REQUESTING: "Requesting remote control",
            InputSessionState.CONTROLLING: "Controlling remote",
            InputSessionState.BEING_CONTROLLED: "Being controlled",
        }
        self.input_status.setText(labels[change.state])
        active = change.state is not InputSessionState.IDLE
        self.toggle_input_button.setText(
            "Stop Input Share" if active else "Toggle Remote Control"
        )
        self.peer_side_combo.setEnabled(
            self._input_service is not None and not active
        )
        self.auto_edge_checkbox.setEnabled(
            self._input_service is not None and not active
        )
        self.toggle_input_button.setEnabled(self._input_feature_available)

    def _show_input_emergency(self, action: str) -> None:
        if action == "exit":
            self.emergency_exit_requested.emit()

    def _open_remote_shell(self) -> None:
        if self._shell_service is None:
            return
        try:
            session_id = self._shell_service.open_remote(columns=100, rows=30)
        except Exception as error:
            self.shell_status.setText(str(error))
            return
        self._active_shell_session = session_id
        self.shell_status.setText("Requesting")
        terminal = TerminalWindow(
            session_id,
            lambda data, sid=session_id: self._shell_service.send_input(sid, data),
            lambda columns, rows, sid=session_id: self._shell_service.resize(
                sid, columns, rows
            ),
            lambda sid=session_id: self._shell_service.close_session(sid),
        )
        terminal.status_label.setText("Requesting remote shell")
        self.terminal_window = terminal
        terminal.show()
        self.terminate_shell_button.setEnabled(True)

    def _terminate_shell(self) -> None:
        if self._shell_service is None or self._active_shell_session is None:
            return
        session_id = self._active_shell_session
        self._shell_service.close_session(session_id)

    def _show_shell_output(self, output: ShellOutput) -> None:
        if (
            self.terminal_window is not None
            and self.terminal_window.session_id == output.session_id
        ):
            self.terminal_window.feed_output(output.data)

    def _show_shell_state(self, state: ShellState) -> None:
        if state.state in {"requesting", "active"}:
            self._active_shell_session = state.session_id
        if state.state == "active":
            self.shell_status.setText(
                "Executing remote shell"
                if state.direction == "incoming"
                else "Remote shell active"
            )
            self.terminate_shell_button.setEnabled(True)
            if (
                state.direction == "outgoing"
                and self.terminal_window is not None
                and self.terminal_window.session_id == state.session_id
            ):
                self.terminal_window.status_label.setText("Remote shell connected")
            return
        if state.state in {"denied", "exited"}:
            self.shell_status.setText(
                f"Denied: {state.reason}" if state.state == "denied" else "Shell exited"
            )
            if (
                self.terminal_window is not None
                and self.terminal_window.session_id == state.session_id
            ):
                self.terminal_window.mark_exited(state.exit_code)
            if self._active_shell_session == state.session_id:
                self._active_shell_session = None
                self.terminate_shell_button.setEnabled(False)

    def _show_file_progress(self, progress: FileProgress) -> None:
        self._file_transfers[progress.transfer_id] = progress
        self._file_transfers.move_to_end(progress.transfer_id)
        self._prune_file_progress_history()
        finished = progress.state in {"complete", "failed", "cancelled"}
        if not finished:
            self._active_transfer_id = progress.transfer_id
        elif self._active_transfer_id not in {None, progress.transfer_id}:
            return
        elif self._active_transfer_id == progress.transfer_id:
            remaining = [
                item
                for item in self._file_transfers.values()
                if item.state not in {"complete", "failed", "cancelled"}
            ]
            if remaining:
                replacement = remaining[-1]
                self._active_transfer_id = replacement.transfer_id
                self._render_file_progress(replacement)
                return
            self._active_transfer_id = None
        self._render_file_progress(progress)

    def _prune_file_progress_history(self) -> None:
        while len(self._file_transfers) > 128:
            removable = next(
                (
                    transfer_id
                    for transfer_id, progress in self._file_transfers.items()
                    if transfer_id != self._active_transfer_id
                    and progress.state in {"complete", "failed", "cancelled"}
                ),
                None,
            )
            if removable is None:
                break
            self._file_transfers.pop(removable, None)

    def _render_file_progress(self, progress: FileProgress) -> None:
        percent = 100 if progress.total == 0 else round(
            100 * progress.transferred / progress.total
        )
        self.file_progress_bar.setValue(max(0, min(100, percent)))
        self.file_progress_label.setText(
            f"{progress.name} · {percent}% · {_format_bytes(progress.transferred)} / "
            f"{_format_bytes(progress.total)} · "
            f"{_format_bytes(progress.throughput_bps)}/s · {progress.state}"
            + (
                f" · {progress.path}"
                if progress.state == "complete" and progress.path is not None
                else ""
            )
        )
        finished = progress.state in {"complete", "failed", "cancelled"}
        self.file_cancel_button.setEnabled(not finished)

    def set_connected(self, connected: bool) -> None:
        self._connected = connected
        self._connecting = False
        self.connection_status.setText("Connected" if connected else "Disconnected")
        self.connect_button.setText("Disconnect" if connected else "Connect")
        self.port_combo.setEnabled(not connected)
        self.baud_combo.setEnabled(not connected)

    def available_ports(self) -> tuple[str, ...]:
        return tuple(
            self.port_combo.itemText(index)
            for index in range(self.port_combo.count())
        )

    def set_serial_defaults(self, port: str, baud_rate: int) -> None:
        self.baud_combo.setCurrentText(str(baud_rate))
        if port and port in self.available_ports():
            self.port_combo.setCurrentText(port)

    def persistent_preferences(self) -> tuple[str, int, str, bool, bool, bool]:
        try:
            baud_rate = int(self.baud_combo.currentText().strip())
        except ValueError:
            baud_rate = 115_200
        peer_side = (
            self._input_service.peer_side.value
            if self._input_service is not None
            else "right"
        )
        auto_edge_enabled = bool(
            self._input_service is not None
            and self._input_service.auto_edge_enabled
        )
        allow_remote_shell = bool(
            self._shell_service is not None
            and self._shell_service.allow_remote_shell
        )
        allow_input = bool(
            self._input_service is not None
            and self._input_service.allow_remote_input
        )
        return (
            self.port_combo.currentText().strip(),
            baud_rate,
            peer_side,
            auto_edge_enabled,
            allow_remote_shell,
            allow_input,
        )

    def set_connection_pending(self, port: str) -> None:
        self._connecting = True
        self.connection_status.setText(f"Connecting {port}")
        self.connect_button.setText("Disconnect")
        self.port_combo.setEnabled(False)
        self.baud_combo.setEnabled(False)

    def show_connection_error(self, message: str) -> None:
        self.set_connected(False)
        self.connection_status.setText("Error")
        self.action_status.setText(message)

    def show_operation_error(self, message: str) -> None:
        self.action_status.setText(message)

    def apply_core_snapshot(self, snapshot: CoreSnapshot) -> None:
        self._connection_id = snapshot.connection_id
        self._peer_fingerprint = snapshot.fingerprint
        connected = snapshot.state not in {
            CoreState.DISCONNECTED,
            CoreState.ERROR,
        }
        self.set_connected(connected)

        if snapshot.state is CoreState.UNTRUSTED and snapshot.local_approved:
            status = "Awaiting peer approval"
        else:
            status = {
                CoreState.DISCONNECTED: "Disconnected",
                CoreState.DISCONNECTING: "Disconnecting",
                CoreState.HANDSHAKING: "Handshaking",
                CoreState.UNTRUSTED: "Untrusted",
                CoreState.CHANGED: "Identity changed",
                CoreState.READY: "Ready",
                CoreState.ERROR: "Error",
            }[snapshot.state]
        self.connection_status.setText(status)
        if snapshot.error:
            self.action_status.setText(snapshot.error)

        if snapshot.fingerprint is None:
            self.peer_fingerprint.setText("No authenticated peer")
        else:
            peer = snapshot.peer_id or "Unknown peer"
            self.peer_fingerprint.setText(
                f"{peer} · {snapshot.fingerprint}"
            )
        self.trust_button.setEnabled(
            snapshot.state is CoreState.UNTRUSTED
            and not snapshot.local_approved
            and snapshot.fingerprint is not None
        )
        features = snapshot.features
        plain_available = (
            snapshot.state
            in {CoreState.UNTRUSTED, CoreState.CHANGED, CoreState.READY}
            and "chat" in features
        )
        protected_available = snapshot.state is CoreState.READY
        self._secure_feature_available = (
            protected_available and "chat" in features
        )
        file_available = bool(
            self._file_service is not None
            and protected_available
            and "files" in features
        )
        shell_available = bool(
            self._shell_service is not None
            and protected_available
            and "shell" in features
        )
        self._input_feature_available = bool(
            self._input_service is not None
            and protected_available
            and "input" in features
        )
        self.send_plain_button.setEnabled(plain_available)
        self.file_drop_zone.setEnabled(file_available)
        self.file_select_button.setEnabled(file_available)
        if not file_available:
            self.file_cancel_button.setEnabled(False)
        self.open_shell_button.setEnabled(shell_available)
        self.terminate_shell_button.setEnabled(
            shell_available and self._active_shell_session is not None
        )
        self.toggle_input_button.setEnabled(self._input_feature_available)
        self.refresh_secure_state()

    def refresh_secure_state(self) -> None:
        self.send_secure_button.setEnabled(
            self._secure_feature_available
            and self._chat_service.secure_available
        )

    def _send_plain(self) -> None:
        self._run_send(self._chat_service.send_plain)

    def _send_secure(self) -> None:
        if not self.send_secure_button.isEnabled():
            return
        self._run_send(self._chat_service.send_secure)

    def _run_send(self, sender) -> None:
        try:
            sender(self.message_editor.toPlainText())
        except Exception as error:
            self.action_status.setText(str(error))
            return
        self.action_status.setText("Queued")

    def _show_received_message(self, message: ChatMessage) -> None:
        self.received_view.setPlainText(message.text)
        self.received_kind.setText("Secure" if message.secure else "Plain")

    def _copy_last(self) -> None:
        QApplication.clipboard().setText(self._chat_service.last_text)

    def _copy_last_secure(self) -> None:
        QApplication.clipboard().setText(self._chat_service.last_text)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._chat_service.remove_message_listener(self._chat_listener)
        if self._file_service is not None:
            self._file_service.remove_progress_listener(self._file_listener)
        if self._shell_service is not None:
            self._shell_service.remove_output_listener(self._shell_output_listener)
            self._shell_service.remove_state_listener(self._shell_state_listener)
        if self._input_service is not None:
            self._input_service.remove_state_listener(self._input_state_listener)
            self._input_service.remove_emergency_listener(
                self._input_emergency_listener
            )
        if self.terminal_window is not None:
            self.terminal_window.close()
        super().closeEvent(event)


def _format_bytes(value: float | int) -> str:
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or suffix == "TiB":
            return f"{amount:.0f} {suffix}" if suffix == "B" else f"{amount:.1f} {suffix}"
        amount /= 1024
    raise AssertionError("unreachable")


__all__ = ["COMMON_BAUD_RATES", "FileDropZone", "MainWindow"]
