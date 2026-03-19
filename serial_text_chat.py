from __future__ import annotations

import base64
import binascii
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import uuid
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import BinaryIO
from urllib.parse import unquote, urlparse

import serial
from serial.tools import list_ports

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    TkinterDnD = None

try:
    from pynput import keyboard as pynput_keyboard
    from pynput import mouse as pynput_mouse

    PYNPUT_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    pynput_keyboard = None
    pynput_mouse = None
    PYNPUT_IMPORT_ERROR = exc

if sys.platform == "darwin":
    try:
        import Quartz
    except Exception:  # pragma: no cover - environment dependent
        Quartz = None
    try:
        import AppKit
    except Exception:  # pragma: no cover - environment dependent
        AppKit = None
else:
    Quartz = None
    AppKit = None

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
else:
    ctypes = None
    wintypes = None

MESSAGE_TERMINATOR = b"\0"
CONTROL_PREFIX = "\x1eSTCFILE"
CONTROL_SEPARATOR = "\x1f"
FRAME_KIND_TEXT = "text"
FRAME_KIND_CONTROL = "control"
OBFUSCATION_MARKERS = "ABC"
FILE_CHUNK_SIZE = 1024
HIGH_SPEED_FRAME_DELAY_SECONDS = 0.002
HIGH_SPEED_BAUD_THRESHOLD = 460800
FILE_ACK_TIMEOUT_SECONDS = 5.0
INPUT_REQUEST_TIMEOUT_SECONDS = 3.0
INPUT_NOTICE_SECONDS = 2.5
INPUT_SENDER_POLL_SECONDS = 0.01
INPUT_WARP_IGNORE_EVENTS = 4
REMOTE_KEY_REPEAT_INITIAL_DELAY_SECONDS = 0.35
REMOTE_KEY_REPEAT_INTERVAL_SECONDS = 0.05
REMOTE_KEY_REPEAT_POLL_SECONDS = 0.01
REMOTE_MOUSE_SCROLL_MULTIPLIER = 5
AUTO_EDGE_ENTRY_ANCHOR_INSET_PIXELS = 48
AUTO_EDGE_HOLD_SECONDS = 0.5
AUTO_EDGE_EXIT_STALE_SECONDS = 0.2
AUX_DISPLAY_REFRESH_SECONDS = 1.0
RECEIVED_FILES_DIR = Path(__file__).resolve().parent / "received_files"
BAUD_RATE_OPTIONS = [
    "9600",
    "19200",
    "38400",
    "57600",
    "115200",
    "230400",
    "460800",
    "921600",
    "1000000",
    "1500000",
    "2000000",
]
INPUT_STATE_IDLE = "idle"
INPUT_STATE_REQUESTING = "requesting_remote_control"
INPUT_STATE_CONTROLLING = "controlling_remote"
INPUT_STATE_CONTROLLED = "being_controlled"
WINDOWS_SCROLL_LOCK_VK = 145
WINDOWS_KEY_FLAG_INJECTED = 0x10
WINDOWS_MOUSE_FLAG_INJECTED = 0x01
DARWIN_F8_KEYCODE = 100
EMERGENCY_MODIFIER_TOKENS = frozenset({"special:ctrl", "special:alt", "special:shift"})
EMERGENCY_STOP_KEY_TOKEN = "special:backspace"
EMERGENCY_EXIT_KEY_TOKEN = "special:esc"
APP_HOTKEY_MODIFIER_TOKENS = frozenset({"special:ctrl", "special:alt"})
APP_COPY_TRIGGER_CHARS = frozenset({"c", "ㅊ"})
APP_SEND_TRIGGER_CHARS = frozenset({"v", "ㅍ"})
AUTO_EDGE_DISABLED = "off"
AUTO_EDGE_MODE_ENTER = "enter"
AUTO_EDGE_MODE_EXIT = "exit"
AUTO_EDGE_SIDE_RIGHT = "Right"
AUTO_EDGE_SIDE_LEFT = "Left"
AUTO_EDGE_SIDE_TOP = "Top"
AUTO_EDGE_SIDE_BOTTOM = "Bottom"
AUTO_EDGE_SIDE_OPTIONS = [
    AUTO_EDGE_SIDE_RIGHT,
    AUTO_EDGE_SIDE_LEFT,
    AUTO_EDGE_SIDE_TOP,
    AUTO_EDGE_SIDE_BOTTOM,
]
DARWIN_SHORTCUT_KEYCODES = {
    0: "select_all",
    7: "cut",
    8: "copy",
    9: "paste",
    6: "undo",
    16: "redo",
}
DARWIN_CONTROL_CHAR_SHORTCUTS = {
    "\x01": "select_all",
    "\x03": "copy",
    "\x16": "paste",
    "\x18": "cut",
    "\x19": "redo",
    "\x1a": "undo",
}
DARWIN_EDITOR_SHORTCUT_CHARS = {
    "a": "select_all",
    "c": "copy",
    "x": "cut",
    "v": "paste",
    "y": "redo",
    "z": "undo",
    "ㅁ": "select_all",
    "ㅊ": "copy",
    "ㅌ": "cut",
    "ㅍ": "paste",
    "ㅛ": "redo",
    "ㅋ": "undo",
}
EDITOR_SHORTCUT_MODIFIER_KEYSYMS = {
    "control_l": "control",
    "control_r": "control",
    "meta_l": "command",
    "meta_r": "command",
    "command": "command",
    "command_l": "command",
    "command_r": "command",
    "super_l": "command",
    "super_r": "command",
    "shift_l": "shift",
    "shift_r": "shift",
}
APP_STATE_PATH = Path.home() / ".serial_text_chat_state.json"


def obfuscate_text(text: str) -> str:
    encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
    chunks = [encoded[index : index + 4] for index in range(0, len(encoded), 4)]
    return "".join(chunk + OBFUSCATION_MARKERS[index % len(OBFUSCATION_MARKERS)] for index, chunk in enumerate(chunks))


def deobfuscate_text(text: str) -> str:
    if not text:
        return ""

    if len(text) % 5 != 0:
        raise ValueError("Encoded payload length is invalid.")

    restored_chunks: list[str] = []
    for index in range(0, len(text), 5):
        block = text[index : index + 5]
        expected_marker = OBFUSCATION_MARKERS[(index // 5) % len(OBFUSCATION_MARKERS)]
        if len(block) != 5 or block[-1] != expected_marker:
            raise ValueError("Encoded payload markers do not match.")

        restored_chunks.append(block[:4])

    try:
        decoded = base64.b64decode("".join(restored_chunks), altchars=b"-_", validate=True)
    except binascii.Error as exc:
        raise ValueError("Encoded payload base64 is invalid.") from exc

    return decoded.decode("utf-8")


def encode_control_text(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def decode_control_text(text: str) -> str:
    try:
        decoded = base64.urlsafe_b64decode(text.encode("ascii"))
    except binascii.Error as exc:
        raise ValueError("Control field base64 is invalid.") from exc

    return decoded.decode("utf-8")


def build_control_message(command: str, *parts: str) -> str:
    return json.dumps(
        {"t": FRAME_KIND_CONTROL, "c": command, "p": list(parts)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_legacy_control_message(message: str) -> tuple[str, list[str]] | None:
    prefix = CONTROL_PREFIX + CONTROL_SEPARATOR
    if not message.startswith(prefix):
        return None

    parts = message[len(prefix) :].split(CONTROL_SEPARATOR)
    if not parts or not parts[0]:
        raise ValueError("Control message is malformed.")

    return parts[0], parts[1:]


def build_text_message(payload: str) -> str:
    return json.dumps(
        {"t": FRAME_KIND_TEXT, "p": payload},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_serial_message(message: str) -> tuple[str, str | tuple[str, list[str]]]:
    try:
        frame = json.loads(message)
    except json.JSONDecodeError:
        legacy_control = parse_legacy_control_message(message)
        if legacy_control is not None:
            return FRAME_KIND_CONTROL, legacy_control
        return FRAME_KIND_TEXT, message

    if not isinstance(frame, dict):
        raise ValueError("Serialized frame is malformed.")

    frame_type = frame.get("t")
    if frame_type == FRAME_KIND_TEXT:
        payload = frame.get("p")
        if not isinstance(payload, str):
            raise ValueError("Serialized text frame is malformed.")
        return FRAME_KIND_TEXT, payload

    if frame_type == FRAME_KIND_CONTROL:
        command = frame.get("c")
        parts = frame.get("p")
        if not isinstance(command, str) or not isinstance(parts, list) or not all(isinstance(part, str) for part in parts):
            raise ValueError("Serialized control frame is malformed.")
        return FRAME_KIND_CONTROL, (command, parts)

    raise ValueError("Serialized frame type is unsupported.")


def normalize_dropped_path(raw_path: str) -> Path:
    if raw_path.startswith("file://"):
        parsed = urlparse(raw_path)
        path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            path = f"//{parsed.netloc}{path}"
        return Path(path)

    return Path(raw_path)


def supported_input_platform() -> bool:
    return sys.platform in {"darwin", "win32"}


def encode_key_token(key: object) -> str:
    if pynput_keyboard is None:
        raise RuntimeError("pynput keyboard support is unavailable.")

    if isinstance(key, pynput_keyboard.Key):
        return f"special:{key.name}"

    if isinstance(key, pynput_keyboard.KeyCode):
        if key.char is not None:
            return f"char:{encode_control_text(key.char)}"
        if key.vk is not None:
            return f"vk:{key.vk}"

    raise ValueError("Unsupported key event.")


def decode_key_token(token: str) -> object:
    if pynput_keyboard is None:
        raise RuntimeError("pynput keyboard support is unavailable.")

    kind, separator, value = token.partition(":")
    if not separator or not value:
        raise ValueError("Key token is malformed.")

    if kind == "special":
        key = getattr(pynput_keyboard.Key, value, None)
        if key is None:
            raise ValueError(f"Unknown special key: {value}")
        return key

    if kind == "char":
        return decode_control_text(value)

    if kind == "vk":
        return pynput_keyboard.KeyCode.from_vk(int(value))

    raise ValueError(f"Unknown key token type: {kind}")


def encode_mouse_button_token(button: object) -> str:
    if pynput_mouse is None:
        raise RuntimeError("pynput mouse support is unavailable.")

    if isinstance(button, pynput_mouse.Button):
        return button.name

    raise ValueError("Unsupported mouse button.")


def decode_mouse_button_token(token: str) -> object:
    if pynput_mouse is None:
        raise RuntimeError("pynput mouse support is unavailable.")

    button = getattr(pynput_mouse.Button, token, None)
    if button is None:
        raise ValueError(f"Unknown mouse button: {token}")
    return button


def canonical_key_token(key_token: str) -> str:
    modifier_aliases = {
        "special:ctrl_l": "special:ctrl",
        "special:ctrl_r": "special:ctrl",
        "special:alt_l": "special:alt",
        "special:alt_r": "special:alt",
        "special:alt_gr": "special:alt",
        "special:shift_l": "special:shift",
        "special:shift_r": "special:shift",
        "special:cmd_l": "special:cmd",
        "special:cmd_r": "special:cmd",
    }
    return modifier_aliases.get(key_token, key_token)


@dataclass
class ReceiveFileState:
    transfer_id: str
    original_name: str
    path: Path
    expected_size: int
    file_handle: BinaryIO
    received_size: int = 0
    next_chunk_index: int = 0


@dataclass(frozen=True)
class DisplayRect:
    min_x: int
    min_y: int
    max_x: int
    max_y: int


def serialize_display_rects(rects: tuple[DisplayRect, ...]) -> str:
    return json.dumps([[rect.min_x, rect.min_y, rect.max_x, rect.max_y] for rect in rects], separators=(",", ":"))


def deserialize_display_rects(serialized: str) -> tuple[DisplayRect, ...]:
    try:
        raw_rects = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError("Display rect snapshot is malformed.") from exc

    if not isinstance(raw_rects, list):
        raise ValueError("Display rect snapshot is malformed.")

    rects: list[DisplayRect] = []
    for raw_rect in raw_rects:
        if (
            not isinstance(raw_rect, list)
            or len(raw_rect) != 4
            or not all(isinstance(value, int) for value in raw_rect)
        ):
            raise ValueError("Display rect snapshot is malformed.")
        rects.append(DisplayRect(*raw_rect))
    return tuple(rects)


class SerialChatApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Serial Text Chat v1")
        self.root.geometry("900x820")

        self.serial_port: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.file_send_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.ui_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.transfer_ack_queue: queue.Queue[tuple[str, list[str]]] = queue.Queue()
        self.input_outbound_queue: queue.Queue[tuple[str, str, tuple[str, ...]] | None] = queue.Queue()
        self.receive_buffer = bytearray()
        self.receive_transfers: dict[str, ReceiveFileState] = {}
        self.last_received_raw = ""
        self.connected_port_name = ""
        self.connected_baudrate = 0
        self.file_send_active = False
        self.write_lock = threading.Lock()
        self.input_lock = threading.Lock()

        self.keyboard_listener = None
        self.mouse_listener = None
        self.keyboard_controller = None
        self.mouse_controller = None
        self.input_backend_ready = False
        self.input_receive_ready = False
        self.input_permission_required = False
        self.input_support_message = self.build_default_input_support_message()
        self.input_state = INPUT_STATE_IDLE
        self.local_input_session_id = ""
        self.remote_input_session_id = ""
        self.local_input_request_deadline = 0.0
        self.input_notice_message = ""
        self.input_notice_deadline = 0.0
        self.local_pressed_key_tokens: set[str] = set()
        self.local_pressed_mouse_tokens: set[str] = set()
        self.hotkey_pressed_key_tokens: set[str] = set()
        self.active_app_hotkey_key_tokens: set[str] = set()
        self.mac_hotkey_backend_ready = False
        self.mac_hotkey_global_monitor = None
        self.mac_hotkey_local_monitor = None
        self.mac_hotkey_global_handler = None
        self.mac_hotkey_local_handler = None
        self.mac_mouse_global_monitor = None
        self.mac_mouse_local_monitor = None
        self.mac_mouse_global_handler = None
        self.mac_mouse_local_handler = None
        self.remote_pressed_key_tokens: set[str] = set()
        self.remote_physical_key_tokens: set[str] = set()
        self.remote_pressed_mouse_tokens: set[str] = set()
        self.remote_repeat_deadlines: dict[str, float] = {}
        self.pending_mouse_dx = 0
        self.pending_mouse_dy = 0
        self.mouse_anchor: tuple[int, int] | None = None
        self.mouse_warp_events_to_ignore = 0
        self.last_idle_pointer_position: tuple[int, int, str] | None = None
        self.edge_hold_direction = ""
        self.edge_hold_mode = ""
        self.edge_hold_started_at = 0.0
        self.edge_hold_last_progress_at = 0.0
        self.display_rects_cache: tuple[DisplayRect, ...] = ()
        self.display_rects_cache_mode = ""
        self.display_rects_cache_deadline = 0.0
        self.remote_pointer_position: tuple[int, int] | None = None
        self.remote_pointer_coordinate_mode = ""
        self.remote_display_rects: tuple[DisplayRect, ...] = ()
        self.arm_auto_edge_anchor_on_next_request = False
        self.pending_auto_edge_anchor_side = ""
        self.pending_auto_edge_anchor_session_id = ""
        self.editor_modifier_state: set[str] = set()
        self.app_state = self.load_app_state()
        self.auto_edge_enabled = bool(self.app_state.get("auto_edge_enabled", False))
        self.peer_side = str(self.app_state.get("peer_side", AUTO_EDGE_SIDE_RIGHT))

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="115200")
        self.status_var = tk.StringVar(value="Disconnected")
        self.input_state_var = tk.StringVar(value="State: Idle")
        self.input_hotkey_var = tk.StringVar(value=self.build_hotkey_hint())
        self.input_support_var = tk.StringVar(value=self.input_support_message)
        self.auto_edge_enabled_var = tk.BooleanVar(value=self.auto_edge_enabled)
        self.peer_side_var = tk.StringVar(value=self.peer_side)

        self._build_ui()
        self.refresh_ports()
        self.refresh_input_ui()
        self.update_controls()
        self.update_windows_listener_suppression()

        self.input_sender_thread = threading.Thread(target=self.input_sender_loop, daemon=True)
        self.input_sender_thread.start()
        self.remote_key_repeat_thread = threading.Thread(target=self.remote_key_repeat_loop, daemon=True)
        self.remote_key_repeat_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.process_ui_events)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(fill="x")

        ttk.Label(controls, text="Port").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(controls, textvariable=self.port_var, width=18, state="normal")
        self.port_combo.grid(row=1, column=0, padx=(0, 8), sticky="ew")

        ttk.Label(controls, text="Baud").grid(row=0, column=1, sticky="w")
        self.baud_combo = ttk.Combobox(
            controls,
            textvariable=self.baud_var,
            values=BAUD_RATE_OPTIONS,
            width=12,
            state="normal",
        )
        self.baud_combo.grid(row=1, column=1, padx=(0, 8), sticky="ew")

        ttk.Button(controls, text="Refresh Ports", command=self.refresh_ports).grid(row=1, column=2, padx=(0, 8))

        self.connect_button = ttk.Button(controls, text="Connect", command=self.toggle_connection)
        self.connect_button.grid(row=1, column=3)

        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=0)

        log_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        receive_actions = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        receive_actions.pack(fill="x")

        self.copy_button = ttk.Button(
            receive_actions,
            text="Copy to Clipboard",
            command=self.copy_last_received,
            state="disabled",
        )
        self.copy_button.pack(side="left", padx=(0, 8))

        self.copy_decoded_button = ttk.Button(
            receive_actions,
            text="Copy to Clipboard After Decode",
            command=self.copy_last_received_after_decode,
            state="disabled",
        )
        self.copy_decoded_button.pack(side="left")

        input_share_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        input_share_frame.pack(fill="x")

        ttk.Label(input_share_frame, text="Input Share").pack(anchor="w")

        input_actions = ttk.Frame(input_share_frame)
        input_actions.pack(fill="x", pady=(6, 0))

        self.toggle_remote_button = ttk.Button(
            input_actions,
            text="Toggle Remote Control",
            command=self.toggle_remote_control,
            state="disabled",
        )
        self.toggle_remote_button.pack(side="left", padx=(0, 12))

        ttk.Label(input_actions, textvariable=self.input_state_var).pack(side="left", anchor="w")
        edge_actions = ttk.Frame(input_share_frame)
        edge_actions.pack(fill="x", pady=(6, 0))

        self.auto_edge_toggle = ttk.Checkbutton(
            edge_actions,
            text="Auto edge toggle",
            variable=self.auto_edge_enabled_var,
            command=self.on_auto_edge_settings_changed,
        )
        self.auto_edge_toggle.pack(side="left", padx=(0, 12))

        ttk.Label(edge_actions, text="Peer side").pack(side="left")
        self.peer_side_combo = ttk.Combobox(
            edge_actions,
            textvariable=self.peer_side_var,
            values=AUTO_EDGE_SIDE_OPTIONS,
            state="readonly",
            width=8,
        )
        self.peer_side_combo.pack(side="left", padx=(8, 0))
        self.peer_side_combo.bind("<<ComboboxSelected>>", self.on_peer_side_selected)

        ttk.Label(input_share_frame, textvariable=self.input_hotkey_var, wraplength=840, justify="left").pack(anchor="w", pady=(6, 0))
        ttk.Label(input_share_frame, textvariable=self.input_support_var, wraplength=840, justify="left").pack(anchor="w", pady=(2, 0))

        file_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        file_frame.pack(fill="x")

        ttk.Label(file_frame, text="File Transfer").pack(anchor="w")

        file_actions = ttk.Frame(file_frame)
        file_actions.pack(fill="x", pady=(6, 0))

        self.select_file_button = ttk.Button(
            file_actions,
            text="Select Files",
            command=self.select_files_for_send,
            state="disabled",
        )
        self.select_file_button.pack(side="left", padx=(0, 8))

        self.open_download_folder_button = ttk.Button(
            file_actions,
            text="Open Download Folder",
            command=self.open_received_files_folder,
        )
        self.open_download_folder_button.pack(side="left", padx=(0, 8))

        self.drop_zone = tk.Label(
            file_actions,
            text="Drop files here to send",
            relief="groove",
            bd=1,
            padx=14,
            pady=14,
            anchor="center",
        )
        self.drop_zone.pack(side="left", fill="x", expand=True)
        self.configure_drop_zone()

        send_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        send_frame.pack(fill="x")

        ttk.Label(send_frame, text="Message").pack(anchor="w")

        input_frame = ttk.Frame(send_frame)
        input_frame.pack(fill="both", expand=True)
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(0, weight=1)

        text_frame = ttk.Frame(input_frame)
        text_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.message_text = tk.Text(text_frame, wrap="word", height=10, undo=True, autoseparators=True, maxundo=-1)
        self.message_text.pack(side="left", fill="both", expand=True)

        input_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.message_text.yview)
        input_scrollbar.pack(side="right", fill="y")
        self.message_text.configure(yscrollcommand=input_scrollbar.set)
        self.bind_editor_shortcuts()
        self.message_text.bind("<Control-Return>", self.send_plain_shortcut)
        self.message_text.bind("<Control-Shift-Return>", self.send_encoded_shortcut)

        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.grid(row=0, column=1, sticky="ns")

        self.send_plain_button = ttk.Button(
            buttons_frame,
            text="Send Plain",
            command=lambda: self.send_message("plain"),
            state="disabled",
        )
        self.send_plain_button.pack(fill="x", pady=(0, 8))

        self.send_encoded_button = ttk.Button(
            buttons_frame,
            text="Send Encoded",
            command=lambda: self.send_message("encoded"),
            state="disabled",
        )
        self.send_encoded_button.pack(fill="x")

        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(8, 4))
        status_bar.pack(fill="x", side="bottom")

    def build_hotkey_hint(self) -> str:
        if sys.platform == "win32":
            return (
                "Hotkey: Scroll Lock | Global app hotkeys: Ctrl+Alt+C copy, Ctrl+Alt+V send plain, "
                "Ctrl+Alt+Shift+C decode copy, Ctrl+Alt+Shift+V send encoded | "
                "Emergency stop: Ctrl+Alt+Shift+Backspace | Emergency exit: Ctrl+Alt+Shift+Esc"
            )
        if sys.platform == "darwin":
            return (
                "Hotkey: F8 (Shift/Ctrl modifiers are also accepted) | Global app hotkeys: Ctrl+Alt+C copy, "
                "Ctrl+Alt+V send plain, Ctrl+Alt+Shift+C decode copy, Ctrl+Alt+Shift+V send encoded | "
                "Emergency stop: Ctrl+Alt+Shift+Backspace | Emergency exit: Ctrl+Alt+Shift+Esc"
            )
        return "Hotkey: unsupported on this platform"

    def build_default_input_support_message(self) -> str:
        if not supported_input_platform():
            return "Input sharing v1 supports macOS and Windows only."
        if PYNPUT_IMPORT_ERROR is not None:
            return "Input sharing unavailable: install pynput and restart the app."
        if sys.platform == "darwin":
            return (
                "Connect first. Global keyboard hotkeys initialize after connect. Auto edge toggle uses whole-desktop "
                "screen edges, and mouse capture starts only when remote control begins. macOS also needs Accessibility "
                "and Input Monitoring."
            )
        return (
            "Connect first. Global keyboard hotkeys initialize after connect. Auto edge toggle uses whole-desktop "
            "screen edges, and mouse capture starts only when remote control begins."
        )

    def build_input_backend_error_message(self, exc: Exception) -> str:
        if not supported_input_platform():
            return "Input sharing v1 supports macOS and Windows only."
        if sys.platform == "darwin":
            return "Permission required: enable Accessibility and Input Monitoring for the Python app running this program."
        return f"Input sharing unavailable: {exc}"

    def refresh_input_ui(self) -> None:
        with self.input_lock:
            state = self.input_state
            permission_required = self.input_permission_required
            support_message = self.input_support_message
            notice_message = self.input_notice_message
            notice_deadline = self.input_notice_deadline

        state_text = {
            INPUT_STATE_IDLE: "Idle",
            INPUT_STATE_REQUESTING: "Requesting remote control",
            INPUT_STATE_CONTROLLING: "Controlling remote",
            INPUT_STATE_CONTROLLED: "Being controlled",
        }.get(state, state)

        now = time.monotonic()
        if permission_required:
            state_text = "Permission required"
        elif notice_message and notice_deadline > now:
            state_text = notice_message

        self.input_state_var.set(f"State: {state_text}")
        self.input_hotkey_var.set(self.build_hotkey_hint())
        self.input_support_var.set(support_message)

    def queue_refresh_input_ui(self) -> None:
        self.ui_events.put(("refresh-input-ui", None))

    def set_input_notice(self, message: str, duration: float = INPUT_NOTICE_SECONDS) -> None:
        with self.input_lock:
            self.input_notice_message = message
            self.input_notice_deadline = time.monotonic() + duration
        self.queue_refresh_input_ui()

    def clear_input_notice_locked(self) -> None:
        self.input_notice_message = ""
        self.input_notice_deadline = 0.0

    def serial_connected(self) -> bool:
        return bool(self.serial_port and self.serial_port.is_open)

    def input_state_snapshot(self) -> tuple[str, bool]:
        with self.input_lock:
            return self.input_state, self.input_backend_ready

    def input_feature_available(self) -> bool:
        return supported_input_platform() and PYNPUT_IMPORT_ERROR is None and pynput_keyboard is not None and pynput_mouse is not None

    def refresh_input_support_for_connection(self) -> None:
        with self.input_lock:
            self.input_backend_ready = False
            self.input_receive_ready = False
            self.input_permission_required = False
            if not self.input_feature_available():
                self.input_support_message = self.build_default_input_support_message()
            elif sys.platform == "darwin":
                self.input_support_message = (
                    "Connected. Preparing global keyboard hotkeys and auto edge monitors. Mouse capture will start only "
                    "when remote control begins."
                )
            else:
                self.input_support_message = (
                    "Connected. Preparing global keyboard hotkeys. Auto edge can use the mouse listener while idle, and "
                    "mouse capture will start when remote control begins."
                )

    def prepare_input_receive_backend(self) -> bool:
        if not self.input_feature_available():
            with self.input_lock:
                self.input_receive_ready = False
                self.input_permission_required = False
                self.input_support_message = self.build_default_input_support_message()
            self.refresh_input_ui()
            self.update_controls()
            return False

        try:
            if self.keyboard_controller is None:
                self.keyboard_controller = pynput_keyboard.Controller()
            if self.mouse_controller is None:
                self.mouse_controller = pynput_mouse.Controller()

            with self.input_lock:
                self.input_receive_ready = True
                self.input_permission_required = False
                if sys.platform == "darwin":
                    self.input_support_message = (
                        "Connected. Remote receive is ready. Global keyboard hotkeys will be enabled after listener startup."
                    )
                else:
                    self.input_support_message = (
                        "Connected. Remote receive is ready. Global keyboard hotkeys will be enabled after listener startup."
                    )
        except Exception as exc:
            with self.input_lock:
                self.input_receive_ready = False
                self.input_permission_required = sys.platform == "darwin"
                self.input_support_message = self.build_input_backend_error_message(exc)
            self.append_log(f"[input share] {self.input_support_message}")

        self.refresh_input_ui()
        self.update_controls()
        self.update_windows_listener_suppression()
        with self.input_lock:
            return self.input_receive_ready

    def start_mac_hotkey_backend(self) -> bool:
        if sys.platform != "darwin":
            return False

        if AppKit is None:
            with self.input_lock:
                self.input_permission_required = True
                self.input_support_message = "Input sharing unavailable: AppKit global hotkey support is unavailable."
            self.refresh_input_ui()
            self.update_controls()
            return False

        if self.mac_hotkey_backend_ready:
            return True

        try:
            if self.mac_hotkey_global_monitor is None:
                self.mac_hotkey_global_handler = self.handle_mac_hotkey_event
                self.mac_hotkey_global_monitor = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                    AppKit.NSEventMaskKeyDown,
                    self.mac_hotkey_global_handler,
                )

            if self.mac_hotkey_local_monitor is None:
                def local_handler(event: object) -> object:
                    self.handle_mac_hotkey_event(event)
                    return event

                self.mac_hotkey_local_handler = local_handler
                self.mac_hotkey_local_monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                    AppKit.NSEventMaskKeyDown,
                    self.mac_hotkey_local_handler,
                )

            mouse_mask = (
                AppKit.NSEventMaskMouseMoved
                | AppKit.NSEventMaskLeftMouseDragged
                | AppKit.NSEventMaskRightMouseDragged
                | AppKit.NSEventMaskOtherMouseDragged
            )

            if self.mac_mouse_global_monitor is None:
                self.mac_mouse_global_handler = self.handle_mac_mouse_event
                self.mac_mouse_global_monitor = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                    mouse_mask,
                    self.mac_mouse_global_handler,
                )

            if self.mac_mouse_local_monitor is None:
                def local_mouse_handler(event: object) -> object:
                    self.handle_mac_mouse_event(event)
                    return event

                self.mac_mouse_local_handler = local_mouse_handler
                self.mac_mouse_local_monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                    mouse_mask,
                    self.mac_mouse_local_handler,
                )

            self.mac_hotkey_backend_ready = True
            with self.input_lock:
                self.input_permission_required = False
                self.input_support_message = (
                    "Global keyboard hotkeys are ready. F8 and Ctrl+Alt(+Shift)+C/V work while connected. "
                    "Mouse capture starts only when remote control begins."
                )
        except Exception as exc:
            self.stop_mac_hotkey_backend()
            with self.input_lock:
                self.input_permission_required = True
                self.input_support_message = self.build_input_backend_error_message(exc)
            self.append_log(f"[input share] {self.input_support_message}")
            self.refresh_input_ui()
            self.update_controls()
            return False

        self.refresh_input_ui()
        self.update_controls()
        return True

    def stop_mac_hotkey_backend(self) -> None:
        if sys.platform != "darwin" or AppKit is None:
            self.mac_hotkey_backend_ready = False
            return

        for attr_name in (
            "mac_hotkey_global_monitor",
            "mac_hotkey_local_monitor",
            "mac_mouse_global_monitor",
            "mac_mouse_local_monitor",
        ):
            monitor = getattr(self, attr_name)
            if monitor is None:
                continue
            try:
                AppKit.NSEvent.removeMonitor_(monitor)
            except Exception:
                pass
            setattr(self, attr_name, None)

        self.mac_hotkey_global_handler = None
        self.mac_hotkey_local_handler = None
        self.mac_mouse_global_handler = None
        self.mac_mouse_local_handler = None
        self.mac_hotkey_backend_ready = False

    def handle_mac_hotkey_event(self, event: object) -> None:
        if not self.serial_connected() or AppKit is None:
            return

        try:
            if bool(event.isARepeat()):
                return
        except Exception:
            pass

        try:
            keycode = int(event.keyCode())
            modifier_flags = int(event.modifierFlags())
        except Exception:
            return

        control_active = bool(modifier_flags & AppKit.NSEventModifierFlagControl)
        alt_active = bool(modifier_flags & AppKit.NSEventModifierFlagOption)
        shift_active = bool(modifier_flags & AppKit.NSEventModifierFlagShift)

        if keycode == DARWIN_F8_KEYCODE:
            self.ui_events.put(("toggle-input", None))
            return

        if not (control_active and alt_active):
            return

        if keycode == 8:
            self.ui_events.put(("copy-last-decoded" if shift_active else "copy-last", None))
        elif keycode == 9:
            self.ui_events.put(("send-encoded-global" if shift_active else "send-plain-global", None))

    def handle_mac_mouse_event(self, event: object) -> None:
        del event
        if sys.platform != "darwin" or AppKit is None:
            return

        if not self.serial_connected():
            return

        point = AppKit.NSEvent.mouseLocation()
        self.handle_idle_edge_pointer_move(int(point.x), int(point.y), coordinate_mode="appkit")

    def build_keyboard_listener_options(self) -> dict[str, object]:
        keyboard_options: dict[str, object] = {}
        if sys.platform == "darwin":
            if Quartz is None:
                raise PermissionError("Quartz support is unavailable.")
            keyboard_options["darwin_intercept"] = self.darwin_keyboard_intercept
        elif sys.platform == "win32":
            keyboard_options["win32_event_filter"] = self.win32_keyboard_filter
        return keyboard_options

    def build_mouse_listener_options(self) -> dict[str, object]:
        mouse_options: dict[str, object] = {}
        if sys.platform == "darwin":
            if Quartz is None:
                raise PermissionError("Quartz support is unavailable.")
            mouse_options["darwin_intercept"] = self.darwin_mouse_intercept
        elif sys.platform == "win32":
            mouse_options["win32_event_filter"] = self.win32_mouse_filter
        return mouse_options

    def ensure_keyboard_listener_started(self) -> None:
        if self.keyboard_listener is not None:
            self.raise_if_input_listener_failed(self.keyboard_listener)
            return

        self.keyboard_listener = pynput_keyboard.Listener(
            on_press=self.handle_global_key_press,
            on_release=self.handle_global_key_release,
            suppress=False,
            **self.build_keyboard_listener_options(),
        )
        self.keyboard_listener.start()
        self.keyboard_listener.wait()
        self.raise_if_input_listener_failed(self.keyboard_listener)

    def ensure_mouse_listener_started(self) -> None:
        if self.mouse_listener is not None:
            self.raise_if_input_listener_failed(self.mouse_listener)
            return

        self.mouse_listener = pynput_mouse.Listener(
            on_move=self.handle_global_mouse_move,
            on_click=self.handle_global_mouse_click,
            on_scroll=self.handle_global_mouse_scroll,
            suppress=False,
            **self.build_mouse_listener_options(),
        )
        self.mouse_listener.start()
        self.mouse_listener.wait()
        self.raise_if_input_listener_failed(self.mouse_listener)

    def start_hotkey_backend(self) -> bool:
        if not self.input_feature_available():
            return False

        if sys.platform == "darwin":
            return self.start_mac_hotkey_backend()

        try:
            self.ensure_keyboard_listener_started()
            with self.input_lock:
                self.input_permission_required = False
                if sys.platform == "darwin":
                    self.input_support_message = (
                        "Global keyboard hotkeys are ready. F8 and Ctrl+Alt(+Shift)+C/V work while connected. "
                        "Mouse capture starts only when remote control begins."
                    )
                else:
                    self.input_support_message = (
                        "Global keyboard hotkeys are ready. Scroll Lock and Ctrl+Alt(+Shift)+C/V work while connected. "
                        "Mouse capture starts only when remote control begins."
                    )
        except Exception as exc:
            try:
                self.stop_input_capture_backend()
            except Exception:
                pass
            with self.input_lock:
                self.input_permission_required = sys.platform == "darwin"
                self.input_support_message = self.build_input_backend_error_message(exc)
            self.append_log(f"[input share] {self.input_support_message}")
            self.refresh_input_ui()
            self.update_controls()
            return False

        self.refresh_input_ui()
        self.update_controls()
        self.update_windows_listener_suppression()
        return True

    def sync_auto_edge_runtime(self) -> None:
        if not self.serial_connected() or not self.auto_edge_enabled_var.get():
            return

        if sys.platform == "win32":
            try:
                self.ensure_mouse_listener_started()
            except Exception as exc:
                self.append_log(f"[input share] auto edge mouse listener unavailable: {exc}")
        elif sys.platform == "darwin" and not self.mac_hotkey_backend_ready:
            self.start_hotkey_backend()

    def update_controls(self) -> None:
        connected = self.serial_connected()
        input_state, input_backend_ready = self.input_state_snapshot()
        input_supported = supported_input_platform()

        self.connect_button.configure(text="Disconnect" if connected else "Connect")
        self.send_plain_button.configure(state="normal" if connected else "disabled")
        self.send_encoded_button.configure(state="normal" if connected else "disabled")
        self.select_file_button.configure(state="normal" if connected and not self.file_send_active and input_state == INPUT_STATE_IDLE else "disabled")
        self.auto_edge_toggle.configure(state="normal" if input_supported else "disabled")
        self.peer_side_combo.configure(state="readonly" if input_supported else "disabled")

        if not connected or self.file_send_active:
            toggle_state = "disabled"
        elif input_state in {INPUT_STATE_REQUESTING, INPUT_STATE_CONTROLLING}:
            toggle_state = "normal"
        elif input_state == INPUT_STATE_CONTROLLED:
            toggle_state = "disabled"
        elif input_backend_ready or self.input_feature_available():
            toggle_state = "normal"
        else:
            toggle_state = "disabled"

        button_text = "Stop Remote Control" if input_state in {INPUT_STATE_REQUESTING, INPUT_STATE_CONTROLLING} else "Toggle Remote Control"
        self.toggle_remote_button.configure(text=button_text, state=toggle_state)
        self.update_drop_zone_state()

    def configure_drop_zone(self) -> None:
        self.drop_zone.configure(bg="#f7f7f7", fg="#222222")

        if DND_FILES is None:
            self.drop_zone.configure(text="Drag and drop unavailable. Install tkinterdnd2 or use Select Files.")
            return

        try:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<Drop>>", self.handle_file_drop)
            self.drop_zone.dnd_bind("<<DragEnter>>", self.handle_drag_enter)
            self.drop_zone.dnd_bind("<<DragLeave>>", self.handle_drag_leave)
        except tk.TclError:
            self.drop_zone.configure(text="Drag and drop unavailable in this Tk build. Use Select Files.")

    def update_drop_zone_state(self) -> None:
        connected = self.serial_connected()
        input_state, _ = self.input_state_snapshot()

        if DND_FILES is None:
            self.drop_zone.configure(text="Drag and drop unavailable. Install tkinterdnd2 or use Select Files.")
            return

        if not connected:
            self.drop_zone.configure(text="Connect first, then drop files here to send", bg="#f7f7f7")
        elif input_state == INPUT_STATE_CONTROLLED:
            self.drop_zone.configure(text="Peer is controlling this machine. File send is disabled.", bg="#fff1d6")
        elif input_state in {INPUT_STATE_REQUESTING, INPUT_STATE_CONTROLLING}:
            self.drop_zone.configure(text="Remote control is active. Stop it before sending files.", bg="#fff1d6")
        elif self.file_send_active:
            self.drop_zone.configure(text="Sending files...", bg="#e7f0ff")
        else:
            self.drop_zone.configure(text="Drop files here to send", bg="#f7f7f7")

    def refresh_ports(self) -> None:
        ports = [port.device for port in list_ports.comports()]
        self.port_combo["values"] = ports
        preferred_port = self.app_state.get("last_port", "")
        current_port = self.port_var.get().strip()

        if not ports:
            return

        if current_port in ports:
            return

        if preferred_port in ports:
            self.port_var.set(preferred_port)
            return

        self.port_var.set(ports[0])

    def toggle_connection(self) -> None:
        if self.serial_connected():
            self.disconnect()
        else:
            self.connect()

    def connect(self) -> None:
        port_name = self.port_var.get().strip()
        baudrate_text = self.baud_var.get().strip()

        if not port_name:
            messagebox.showerror("Connection Error", "Select or type a serial port first.")
            return

        try:
            baudrate = int(baudrate_text)
        except ValueError:
            messagebox.showerror("Connection Error", "Baud rate must be a number.")
            return

        try:
            self.serial_port = serial.serial_for_url(
                port_name,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.2,
                write_timeout=1,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
        except serial.SerialException as exc:
            messagebox.showerror("Connection Error", str(exc))
            self.serial_port = None
            return

        self.stop_event.clear()
        self.receive_buffer.clear()
        self.connected_port_name = port_name
        self.connected_baudrate = baudrate
        self.app_state["last_port"] = port_name
        self.save_app_state()
        self.refresh_input_support_for_connection()
        self.prepare_input_receive_backend()
        self.start_hotkey_backend()
        self.sync_auto_edge_runtime()
        self.reader_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.reader_thread.start()
        self.refresh_connection_status()
        self.refresh_input_ui()
        self.update_controls()
        self.message_text.focus_set()

    def disconnect(self) -> None:
        self.stop_event.set()
        self.stop_input_sessions_for_disconnect()

        if self.serial_port:
            try:
                if self.serial_port.is_open:
                    self.serial_port.close()
            except serial.SerialException:
                pass

            self.serial_port = None

        self.close_receive_transfers(interrupted=True)
        self.stop_input_backend()
        self.connected_port_name = ""
        self.connected_baudrate = 0
        self.file_send_active = False
        self.receive_buffer.clear()
        self.refresh_connection_status()
        self.refresh_input_ui()
        self.update_controls()

    def refresh_connection_status(self) -> None:
        if self.serial_connected() and self.connected_port_name:
            self.status_var.set(f"Connected to {self.connected_port_name} @ {self.connected_baudrate}")
        else:
            self.status_var.set("Disconnected")

    def load_app_state(self) -> dict[str, object]:
        try:
            raw_state = json.loads(APP_STATE_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

        if not isinstance(raw_state, dict):
            return {}

        loaded_state: dict[str, object] = {}
        last_port = raw_state.get("last_port")
        if isinstance(last_port, str):
            loaded_state["last_port"] = last_port

        auto_edge_enabled = raw_state.get("auto_edge_enabled")
        if isinstance(auto_edge_enabled, bool):
            loaded_state["auto_edge_enabled"] = auto_edge_enabled

        peer_side = raw_state.get("peer_side")
        if isinstance(peer_side, str) and peer_side in AUTO_EDGE_SIDE_OPTIONS:
            loaded_state["peer_side"] = peer_side

        return loaded_state

    def save_app_state(self) -> None:
        try:
            APP_STATE_PATH.write_text(
                json.dumps(self.app_state, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
        except OSError:
            pass

    def persist_auto_edge_settings(self) -> None:
        self.auto_edge_enabled = bool(self.auto_edge_enabled_var.get())
        self.peer_side = self.peer_side_var.get().strip() or AUTO_EDGE_SIDE_RIGHT
        self.app_state["auto_edge_enabled"] = self.auto_edge_enabled
        self.app_state["peer_side"] = self.peer_side
        self.save_app_state()

    def on_peer_side_selected(self, event: tk.Event) -> None:
        del event
        self.on_auto_edge_settings_changed()

    def on_auto_edge_settings_changed(self) -> None:
        side = self.peer_side_var.get().strip()
        if side not in AUTO_EDGE_SIDE_OPTIONS:
            side = AUTO_EDGE_SIDE_RIGHT
            self.peer_side_var.set(side)

        self.persist_auto_edge_settings()
        self.reset_auto_edge_hold()
        self.refresh_input_ui()
        self.update_controls()
        self.sync_auto_edge_runtime()

    def current_peer_side(self) -> str:
        side = self.peer_side
        if side in AUTO_EDGE_SIDE_OPTIONS:
            return side
        return AUTO_EDGE_SIDE_RIGHT

    def current_exit_side(self) -> str:
        return {
            AUTO_EDGE_SIDE_RIGHT: AUTO_EDGE_SIDE_LEFT,
            AUTO_EDGE_SIDE_LEFT: AUTO_EDGE_SIDE_RIGHT,
            AUTO_EDGE_SIDE_TOP: AUTO_EDGE_SIDE_BOTTOM,
            AUTO_EDGE_SIDE_BOTTOM: AUTO_EDGE_SIDE_TOP,
        }[self.current_peer_side()]

    def reset_auto_edge_hold_locked(self) -> None:
        self.edge_hold_direction = ""
        self.edge_hold_mode = ""
        self.edge_hold_started_at = 0.0
        self.edge_hold_last_progress_at = 0.0

    def reset_auto_edge_hold(self) -> None:
        with self.input_lock:
            self.reset_auto_edge_hold_locked()

    def clear_remote_pointer_context_locked(self) -> None:
        self.remote_pointer_position = None
        self.remote_pointer_coordinate_mode = ""
        self.remote_display_rects = ()

    def clear_pending_auto_edge_anchor_locked(self) -> None:
        self.arm_auto_edge_anchor_on_next_request = False
        self.pending_auto_edge_anchor_side = ""
        self.pending_auto_edge_anchor_session_id = ""

    def coordinate_mode_uses_bottom_left_origin(self, coordinate_mode: str) -> bool:
        return coordinate_mode in {"appkit", "quartz"}

    def refresh_display_rects(self, coordinate_mode: str) -> tuple[DisplayRect, ...]:
        if coordinate_mode == "appkit" and sys.platform == "darwin" and AppKit is not None:
            screens = AppKit.NSScreen.screens()
            return tuple(
                DisplayRect(
                    int(screen.frame().origin.x),
                    int(screen.frame().origin.y),
                    int(screen.frame().origin.x + screen.frame().size.width),
                    int(screen.frame().origin.y + screen.frame().size.height),
                )
                for screen in screens
            )

        if coordinate_mode == "quartz" and sys.platform == "darwin" and Quartz is not None:
            error, display_ids, display_count = Quartz.CGGetActiveDisplayList(32, None, None)
            if error != 0:
                return ()

            rects: list[DisplayRect] = []
            for display_id in display_ids[:display_count]:
                bounds = Quartz.CGDisplayBounds(display_id)
                rects.append(
                    DisplayRect(
                        int(bounds.origin.x),
                        int(bounds.origin.y),
                        int(bounds.origin.x + bounds.size.width),
                        int(bounds.origin.y + bounds.size.height),
                    )
                )
            return tuple(rects)

        if coordinate_mode == "screen" and sys.platform == "win32" and ctypes is not None and wintypes is not None:
            rects: list[DisplayRect] = []

            callback_type = ctypes.WINFUNCTYPE(
                ctypes.c_int,
                wintypes.HMONITOR,
                wintypes.HDC,
                ctypes.POINTER(wintypes.RECT),
                wintypes.LPARAM,
            )

            @callback_type
            def enum_callback(monitor: object, hdc: object, rect_ptr: object, data: object) -> int:
                del monitor, hdc, data
                rect = rect_ptr.contents
                rects.append(DisplayRect(int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)))
                return 1

            ctypes.windll.user32.EnumDisplayMonitors(0, 0, enum_callback, 0)
            return tuple(rects)

        return ()

    def get_display_rects(self, coordinate_mode: str) -> tuple[DisplayRect, ...]:
        now = time.monotonic()
        with self.input_lock:
            if (
                self.display_rects_cache
                and self.display_rects_cache_mode == coordinate_mode
                and self.display_rects_cache_deadline > now
            ):
                return self.display_rects_cache

        rects = self.refresh_display_rects(coordinate_mode)
        with self.input_lock:
            self.display_rects_cache = rects
            self.display_rects_cache_mode = coordinate_mode
            self.display_rects_cache_deadline = now + AUX_DISPLAY_REFRESH_SECONDS
        return rects

    def point_matches_edge_segment(self, rect: DisplayRect, side: str, x: int, y: int) -> bool:
        if side in {AUTO_EDGE_SIDE_RIGHT, AUTO_EDGE_SIDE_LEFT}:
            return rect.min_y <= y < rect.max_y
        return rect.min_x <= x < rect.max_x

    def edge_boundary_for_rects(
        self,
        rects: tuple[DisplayRect, ...],
        side: str,
        x: int,
        y: int,
        coordinate_mode: str,
    ) -> int | None:
        matching_rects = [rect for rect in rects if self.point_matches_edge_segment(rect, side, x, y)]
        if not matching_rects:
            return None

        if side == AUTO_EDGE_SIDE_RIGHT:
            return max(rect.max_x for rect in matching_rects)
        if side == AUTO_EDGE_SIDE_LEFT:
            return min(rect.min_x for rect in matching_rects)
        if side == AUTO_EDGE_SIDE_TOP:
            if self.coordinate_mode_uses_bottom_left_origin(coordinate_mode):
                return max(rect.max_y for rect in matching_rects)
            return min(rect.min_y for rect in matching_rects)
        if self.coordinate_mode_uses_bottom_left_origin(coordinate_mode):
            return min(rect.min_y for rect in matching_rects)
        return max(rect.max_y for rect in matching_rects)

    def edge_boundary_for_point(self, side: str, x: int, y: int, coordinate_mode: str) -> int | None:
        rects = self.get_display_rects(coordinate_mode)
        return self.edge_boundary_for_rects(rects, side, x, y, coordinate_mode)

    def point_is_on_outer_edge_for_rects(
        self,
        rects: tuple[DisplayRect, ...],
        side: str,
        x: int,
        y: int,
        coordinate_mode: str,
    ) -> bool:
        boundary = self.edge_boundary_for_rects(rects, side, x, y, coordinate_mode)
        if boundary is None:
            return False

        if side == AUTO_EDGE_SIDE_RIGHT:
            return x >= boundary - 1
        if side == AUTO_EDGE_SIDE_LEFT:
            return x <= boundary + 1
        if side == AUTO_EDGE_SIDE_TOP:
            if self.coordinate_mode_uses_bottom_left_origin(coordinate_mode):
                return y >= boundary - 1
            return y <= boundary + 1
        if self.coordinate_mode_uses_bottom_left_origin(coordinate_mode):
            return y <= boundary + 1
        return y >= boundary - 1

    def point_is_on_outer_edge(self, side: str, x: int, y: int, coordinate_mode: str) -> bool:
        rects = self.get_display_rects(coordinate_mode)
        return self.point_is_on_outer_edge_for_rects(rects, side, x, y, coordinate_mode)

    def movement_toward_side(self, dx: int, dy: int, side: str, coordinate_mode: str) -> bool:
        if side == AUTO_EDGE_SIDE_RIGHT:
            return dx > 0
        if side == AUTO_EDGE_SIDE_LEFT:
            return dx < 0
        if side == AUTO_EDGE_SIDE_TOP:
            return dy > 0 if self.coordinate_mode_uses_bottom_left_origin(coordinate_mode) else dy < 0
        return dy < 0 if self.coordinate_mode_uses_bottom_left_origin(coordinate_mode) else dy > 0

    def movement_away_from_side(self, dx: int, dy: int, side: str, coordinate_mode: str) -> bool:
        if side == AUTO_EDGE_SIDE_RIGHT:
            return dx < 0
        if side == AUTO_EDGE_SIDE_LEFT:
            return dx > 0
        if side == AUTO_EDGE_SIDE_TOP:
            return dy < 0 if self.coordinate_mode_uses_bottom_left_origin(coordinate_mode) else dy > 0
        return dy > 0 if self.coordinate_mode_uses_bottom_left_origin(coordinate_mode) else dy < 0

    def point_is_inside_rects(self, rects: tuple[DisplayRect, ...], x: int, y: int) -> bool:
        return any(rect.min_x <= x < rect.max_x and rect.min_y <= y < rect.max_y for rect in rects)

    def constrain_point_to_rects(self, rects: tuple[DisplayRect, ...], x: int, y: int) -> tuple[int, int]:
        if not rects:
            return x, y

        if self.point_is_inside_rects(rects, x, y):
            return x, y

        best_candidate: tuple[int, int] | None = None
        best_distance: int | None = None
        for rect in rects:
            candidate_x = min(max(x, rect.min_x), rect.max_x - 1)
            candidate_y = min(max(y, rect.min_y), rect.max_y - 1)
            distance = (candidate_x - x) ** 2 + (candidate_y - y) ** 2
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_candidate = (candidate_x, candidate_y)

        if best_candidate is None:
            return x, y
        return best_candidate

    def translate_screen_delta_for_coordinate_mode(self, dx: int, dy: int, coordinate_mode: str) -> tuple[int, int]:
        if self.coordinate_mode_uses_bottom_left_origin(coordinate_mode):
            return dx, -dy
        return dx, dy

    def capture_local_pointer_context(self) -> tuple[str, int, int, tuple[DisplayRect, ...]] | None:
        if sys.platform == "darwin" and Quartz is not None:
            try:
                event = Quartz.CGEventCreate(None)
                point = Quartz.CGEventGetLocation(event)
                rects = self.get_display_rects("quartz")
            except Exception:
                return None

            if not rects:
                return None
            x, y = self.constrain_point_to_rects(rects, int(point.x), int(point.y))
            return "quartz", x, y, rects

        position = self.get_local_pointer_position()
        if position is None:
            return None

        rects = self.get_display_rects("screen")
        if not rects:
            return None
        x, y = self.constrain_point_to_rects(rects, position[0], position[1])
        return "screen", x, y, rects

    def advance_remote_pointer_locked(self, dx: int, dy: int) -> tuple[int, int, str] | None:
        if self.remote_pointer_position is None or not self.remote_display_rects or not self.remote_pointer_coordinate_mode:
            return None

        translated_dx, translated_dy = self.translate_screen_delta_for_coordinate_mode(
            dx, dy, self.remote_pointer_coordinate_mode
        )
        next_x = self.remote_pointer_position[0] + translated_dx
        next_y = self.remote_pointer_position[1] + translated_dy
        constrained = self.constrain_point_to_rects(self.remote_display_rects, next_x, next_y)
        self.remote_pointer_position = constrained
        return constrained[0], constrained[1], self.remote_pointer_coordinate_mode

    def get_idle_pointer_position(self) -> tuple[int, int, str] | None:
        if sys.platform == "darwin" and AppKit is not None:
            point = AppKit.NSEvent.mouseLocation()
            return int(point.x), int(point.y), "appkit"

        position = self.get_local_pointer_position()
        if position is None:
            return None
        return position[0], position[1], "screen"

    def handle_idle_edge_pointer_move(self, x: int, y: int, coordinate_mode: str) -> None:
        previous_pointer = self.last_idle_pointer_position
        self.last_idle_pointer_position = (x, y, coordinate_mode)

        with self.input_lock:
            if (
                not self.auto_edge_enabled
                or not self.serial_connected()
                or self.file_send_active
                or self.input_state != INPUT_STATE_IDLE
            ):
                self.reset_auto_edge_hold_locked()
                return

            side = self.current_peer_side()

        pointer = self.get_idle_pointer_position()
        if pointer is not None:
            x, y, coordinate_mode = pointer
            self.last_idle_pointer_position = pointer

        at_edge = self.point_is_on_outer_edge(side, x, y, coordinate_mode)
        if not at_edge:
            self.reset_auto_edge_hold()
            return

        with self.input_lock:
            mode = self.edge_hold_mode
            direction = self.edge_hold_direction
            started_at = self.edge_hold_started_at

        if mode == AUTO_EDGE_MODE_ENTER and direction == side and started_at:
            if previous_pointer is not None:
                prev_x, prev_y, prev_mode = previous_pointer
                if prev_mode == coordinate_mode and self.movement_away_from_side(x - prev_x, y - prev_y, side, coordinate_mode):
                    self.reset_auto_edge_hold()
                    return
            return

        if previous_pointer is None:
            return

        prev_x, prev_y, prev_mode = previous_pointer
        if prev_mode != coordinate_mode:
            return

        if not self.movement_toward_side(x - prev_x, y - prev_y, side, coordinate_mode):
            return

        self.start_auto_edge_hold(side, AUTO_EDGE_MODE_ENTER)

    def start_auto_edge_hold(self, direction: str, mode: str) -> None:
        now = time.monotonic()
        with self.input_lock:
            if self.edge_hold_mode == mode and self.edge_hold_direction == direction and self.edge_hold_started_at:
                self.edge_hold_last_progress_at = now
                return

            self.edge_hold_direction = direction
            self.edge_hold_mode = mode
            self.edge_hold_started_at = now
            self.edge_hold_last_progress_at = now

    def handle_auto_edge_exit_motion(self, dx: int, dy: int, remote_pointer: tuple[int, int, str] | None) -> None:
        with self.input_lock:
            if (
                not self.auto_edge_enabled
                or self.file_send_active
                or self.input_state != INPUT_STATE_CONTROLLING
            ):
                self.reset_auto_edge_hold_locked()
                return

            side = self.current_exit_side()
            coordinate_mode = self.remote_pointer_coordinate_mode
            rects = self.remote_display_rects

        if remote_pointer is None or not coordinate_mode or not rects:
            self.reset_auto_edge_hold()
            return

        translated_dx, translated_dy = self.translate_screen_delta_for_coordinate_mode(dx, dy, coordinate_mode)
        if not self.movement_toward_side(translated_dx, translated_dy, side, coordinate_mode):
            if dx != 0 or dy != 0:
                self.reset_auto_edge_hold()
            return

        if not self.point_is_on_outer_edge_for_rects(rects, side, remote_pointer[0], remote_pointer[1], coordinate_mode):
            self.reset_auto_edge_hold()
            return

        self.start_auto_edge_hold(side, AUTO_EDGE_MODE_EXIT)

    def start_input_backend(self) -> None:
        if not self.input_feature_available():
            self.refresh_input_ui()
            self.update_controls()
            return False

        try:
            if self.keyboard_controller is None:
                self.keyboard_controller = pynput_keyboard.Controller()
            if self.mouse_controller is None:
                self.mouse_controller = pynput_mouse.Controller()

            with self.input_lock:
                self.input_receive_ready = True
                self.input_permission_required = False

            self.ensure_keyboard_listener_started()
            self.ensure_mouse_listener_started()

            with self.input_lock:
                self.input_backend_ready = True
                self.input_receive_ready = True
                self.input_permission_required = False
                if sys.platform == "darwin":
                    self.input_support_message = "Global hooks ready. F8 toggles remote control while connected."
                else:
                    self.input_support_message = "Global hooks ready. Scroll Lock toggles remote control while connected."
        except Exception as exc:
            self.stop_input_capture_backend()
            with self.input_lock:
                self.input_backend_ready = False
                self.input_permission_required = sys.platform == "darwin"
                self.input_support_message = self.build_input_backend_error_message(exc)
            self.append_log(f"[input share] {self.input_support_message}")

        self.refresh_input_ui()
        self.update_controls()
        return self.input_backend_ready

    def stop_input_capture_backend(self) -> None:
        for listener_name in ("keyboard_listener", "mouse_listener"):
            listener = getattr(self, listener_name)
            if listener is None:
                continue
            try:
                listener.stop()
                listener.join(0.5)
            except Exception:
                pass
            setattr(self, listener_name, None)

        with self.input_lock:
            self.input_backend_ready = False
        self.update_windows_listener_suppression()

    def stop_input_backend(self) -> None:
        self.stop_input_capture_backend()
        self.stop_mac_hotkey_backend()
        self.keyboard_controller = None
        self.mouse_controller = None
        with self.input_lock:
            self.input_backend_ready = False
            self.input_receive_ready = False
            self.input_permission_required = False
            self.input_support_message = self.build_default_input_support_message()
        self.update_windows_listener_suppression()

    def raise_if_input_listener_failed(self, listener: object) -> None:
        if listener is None:
            raise RuntimeError("Input listener failed to initialize.")

        trusted = getattr(listener, "IS_TRUSTED", True)
        if sys.platform == "darwin" and not trusted:
            raise PermissionError("Input event monitoring is not trusted.")

        if listener.is_alive():
            return

        listener.join(0)
        raise RuntimeError("Input listener stopped during startup.")

    def stop_input_sessions_for_disconnect(self) -> None:
        local_session = ""
        remote_session = ""
        should_send_remote_stop = False

        with self.input_lock:
            if self.input_state in {INPUT_STATE_REQUESTING, INPUT_STATE_CONTROLLING} and self.local_input_session_id:
                local_session = self.local_input_session_id
                should_send_remote_stop = True

            if self.input_state == INPUT_STATE_CONTROLLED and self.remote_input_session_id:
                remote_session = self.remote_input_session_id

            self.local_input_session_id = ""
            self.remote_input_session_id = ""
            self.local_input_request_deadline = 0.0
            self.input_state = INPUT_STATE_IDLE
            self.clear_local_capture_state_locked()
            self.clear_input_notice_locked()
            self.clear_pending_auto_edge_anchor_locked()
            self.reset_auto_edge_hold_locked()

        if should_send_remote_stop and local_session:
            try:
                self.send_control_message("INPUT_RELEASE_ALL", local_session)
                self.send_control_message("INPUT_STOP", local_session)
            except Exception:
                pass

        if remote_session:
            self.release_remote_inputs()

        self.queue_refresh_input_ui()
        self.update_windows_listener_suppression()

    def clear_local_capture_state_locked(self) -> None:
        self.local_pressed_key_tokens.clear()
        self.local_pressed_mouse_tokens.clear()
        self.hotkey_pressed_key_tokens.clear()
        self.active_app_hotkey_key_tokens.clear()
        self.pending_mouse_dx = 0
        self.pending_mouse_dy = 0
        self.mouse_anchor = None
        self.mouse_warp_events_to_ignore = 0
        self.last_idle_pointer_position = None
        self.clear_remote_pointer_context_locked()
        self.reset_auto_edge_hold_locked()

    def update_windows_listener_suppression(self) -> None:
        if sys.platform != "win32":
            return

        with self.input_lock:
            suppress = self.input_state == INPUT_STATE_CONTROLLING

        for listener_name in ("keyboard_listener", "mouse_listener"):
            listener = getattr(self, listener_name)
            if listener is not None and hasattr(listener, "_suppress"):
                listener._suppress = suppress

    def stop_local_input_control(self, send_remote_stop: bool, log_message: str | None = None) -> None:
        session_id = ""
        with self.input_lock:
            if self.input_state not in {INPUT_STATE_REQUESTING, INPUT_STATE_CONTROLLING} or not self.local_input_session_id:
                return

            session_id = self.local_input_session_id
            self.local_input_session_id = ""
            self.local_input_request_deadline = 0.0
            self.input_state = INPUT_STATE_IDLE
            self.clear_local_capture_state_locked()
            self.clear_input_notice_locked()
            self.clear_pending_auto_edge_anchor_locked()

        if send_remote_stop and session_id:
            try:
                self.send_control_message("INPUT_RELEASE_ALL", session_id)
                self.send_control_message("INPUT_STOP", session_id)
            except Exception:
                pass

        if log_message:
            self.append_log(log_message)

        self.refresh_input_ui()
        self.update_controls()
        self.update_windows_listener_suppression()

    def finish_remote_input_control(self, expected_session_id: str, log_message: str | None = None) -> None:
        release_inputs = False

        with self.input_lock:
            if self.remote_input_session_id != expected_session_id:
                return

            self.remote_input_session_id = ""
            self.input_state = INPUT_STATE_IDLE
            self.clear_input_notice_locked()
            self.reset_auto_edge_hold_locked()
            release_inputs = True

        if release_inputs:
            self.release_remote_inputs()

        if log_message:
            self.ui_events.put(("log", log_message))
        self.queue_refresh_input_ui()
        self.update_windows_listener_suppression()

    def check_input_timers(self) -> None:
        refresh_required = False
        log_message = None
        auto_edge_action = ""

        with self.input_lock:
            now = time.monotonic()

            if self.input_notice_message and self.input_notice_deadline <= now:
                self.clear_input_notice_locked()
                refresh_required = True

            if self.input_state == INPUT_STATE_REQUESTING and self.local_input_request_deadline and self.local_input_request_deadline <= now:
                self.local_input_session_id = ""
                self.local_input_request_deadline = 0.0
                self.input_state = INPUT_STATE_IDLE
                self.clear_local_capture_state_locked()
                self.clear_pending_auto_edge_anchor_locked()
                log_message = "[input share] remote control request timed out."
                refresh_required = True

            if self.edge_hold_mode == AUTO_EDGE_MODE_EXIT and self.edge_hold_last_progress_at:
                if self.edge_hold_last_progress_at + AUTO_EDGE_EXIT_STALE_SECONDS <= now:
                    self.reset_auto_edge_hold_locked()

            if self.edge_hold_started_at and self.edge_hold_started_at + AUTO_EDGE_HOLD_SECONDS <= now:
                auto_edge_action = self.edge_hold_mode
                self.reset_auto_edge_hold_locked()

        if log_message:
            self.append_log(log_message)

        if auto_edge_action == AUTO_EDGE_MODE_ENTER:
            pointer = self.get_idle_pointer_position()
            if pointer is not None and self.point_is_on_outer_edge(self.current_peer_side(), pointer[0], pointer[1], pointer[2]):
                self.ui_events.put(("auto-edge-enter", None))
        elif auto_edge_action == AUTO_EDGE_MODE_EXIT:
            self.ui_events.put(("auto-edge-exit", None))

        if refresh_required:
            self.refresh_input_ui()
            self.update_controls()
            self.update_windows_listener_suppression()

    def toggle_remote_control(self) -> None:
        if not self.serial_connected():
            return

        if not self.input_feature_available():
            with self.input_lock:
                self.input_permission_required = False
                self.input_support_message = self.build_default_input_support_message()
            self.set_input_notice("Unavailable")
            self.refresh_input_ui()
            self.update_controls()
            return

        with self.input_lock:
            state = self.input_state
            input_backend_ready = self.input_backend_ready
            support_message = self.input_support_message
            permission_required = self.input_permission_required

        if state in {INPUT_STATE_REQUESTING, INPUT_STATE_CONTROLLING}:
            self.stop_local_input_control(send_remote_stop=True, log_message="[input share] remote control stopped.")
            return

        if state == INPUT_STATE_CONTROLLED:
            self.set_input_notice("Busy")
            self.append_log("[input share] peer is already controlling this machine.")
            return

        if self.file_send_active:
            self.set_input_notice("Busy")
            self.append_log("[input share] file transfer is active. Stop it before remote control.")
            return

        if not input_backend_ready:
            input_backend_ready = bool(self.start_input_backend())
            with self.input_lock:
                support_message = self.input_support_message
                permission_required = self.input_permission_required

        if not input_backend_ready:
            self.set_input_notice("Permission required" if permission_required else "Unavailable")
            self.append_log(f"[input share] {support_message}")
            self.refresh_input_ui()
            self.update_controls()
            return

        session_id = uuid.uuid4().hex
        with self.input_lock:
            self.input_state = INPUT_STATE_REQUESTING
            self.local_input_session_id = session_id
            self.local_input_request_deadline = time.monotonic() + INPUT_REQUEST_TIMEOUT_SECONDS
            self.clear_local_capture_state_locked()
            self.clear_input_notice_locked()
            if self.arm_auto_edge_anchor_on_next_request:
                self.pending_auto_edge_anchor_side = self.current_peer_side()
                self.pending_auto_edge_anchor_session_id = session_id
                self.arm_auto_edge_anchor_on_next_request = False
            else:
                self.pending_auto_edge_anchor_side = ""
                self.pending_auto_edge_anchor_session_id = ""

        self.append_log("[input share] requesting remote control...")
        self.refresh_input_ui()
        self.update_controls()

        try:
            self.send_control_message("INPUT_START", session_id)
        except serial.SerialException as exc:
            with self.input_lock:
                self.input_state = INPUT_STATE_IDLE
                self.local_input_session_id = ""
                self.local_input_request_deadline = 0.0
                self.clear_local_capture_state_locked()
                self.clear_pending_auto_edge_anchor_locked()
            self.refresh_input_ui()
            self.update_controls()
            messagebox.showerror("Remote Control Error", str(exc))
            self.disconnect()

    def read_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.serial_connected():
                break

            try:
                chunk = self.serial_port.read(self.serial_port.in_waiting or 1)
            except serial.SerialException as exc:
                self.ui_events.put(("status", f"Read error: {exc}"))
                break

            if not chunk:
                continue

            self.receive_buffer.extend(chunk)

            while True:
                try:
                    end_index = self.receive_buffer.index(MESSAGE_TERMINATOR[0])
                except ValueError:
                    break

                message_bytes = bytes(self.receive_buffer[:end_index])
                del self.receive_buffer[: end_index + 1]
                message = message_bytes.decode("utf-8", errors="replace")
                self.handle_received_message(message)

        self.close_receive_transfers(interrupted=True)
        self.ui_events.put(("disconnect", None))

    def handle_received_message(self, message: str) -> None:
        try:
            frame_type, payload = parse_serial_message(message)
        except ValueError as exc:
            self.ui_events.put(("log", f"[receive error] {exc}"))
            return

        if frame_type == FRAME_KIND_TEXT:
            self.ui_events.put(("text", str(payload)))
            return

        command, parts = payload

        if command in {"ACK_START", "ACK_CHUNK", "ACK_END"}:
            self.transfer_ack_queue.put((command, parts))
            return

        try:
            if command == "START":
                self.handle_file_start(parts)
            elif command == "CHUNK":
                self.handle_file_chunk(parts)
            elif command == "END":
                self.handle_file_end(parts)
            elif command == "ABORT":
                self.handle_file_abort(parts)
            elif command == "INPUT_START":
                self.handle_input_start(parts)
            elif command == "INPUT_ACK":
                self.handle_input_ack(parts)
            elif command == "INPUT_BUSY":
                self.handle_input_busy(parts)
            elif command == "INPUT_STOP":
                self.handle_input_stop(parts)
            elif command == "INPUT_RELEASE_ALL":
                self.handle_input_release_all(parts)
            elif command == "INPUT_KEY":
                self.handle_input_key(parts)
            elif command == "INPUT_MOUSE_MOVE":
                self.handle_input_mouse_move(parts)
            elif command == "INPUT_MOUSE_BUTTON":
                self.handle_input_mouse_button(parts)
            elif command == "INPUT_MOUSE_SCROLL":
                self.handle_input_mouse_scroll(parts)
            else:
                self.ui_events.put(("log", f"[receive error] unknown control command: {command}"))
        except Exception as exc:
            self.ui_events.put(("log", f"[receive error] {exc}"))

    def handle_file_start(self, parts: list[str]) -> None:
        if len(parts) != 3:
            raise ValueError("FILE START message is malformed.")

        transfer_id, name_b64, size_text = parts
        filename = decode_control_text(name_b64)
        expected_size = int(size_text)
        path = self.build_received_file_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handle = path.open("wb")

        self.receive_transfers[transfer_id] = ReceiveFileState(
            transfer_id=transfer_id,
            original_name=filename,
            path=path,
            expected_size=expected_size,
            file_handle=file_handle,
        )
        self.send_control_message("ACK_START", transfer_id)
        self.ui_events.put(("log", f"[receiving file] {filename} ({expected_size} bytes)"))

    def handle_file_chunk(self, parts: list[str]) -> None:
        if len(parts) != 3:
            raise ValueError("FILE CHUNK message is malformed.")

        transfer_id, chunk_index_text, chunk_b64 = parts
        state = self.receive_transfers.get(transfer_id)
        if state is None:
            raise ValueError("Received a file chunk without a matching transfer.")

        chunk_index = int(chunk_index_text)
        if chunk_index != state.next_chunk_index:
            raise ValueError(f"Unexpected file chunk index: expected {state.next_chunk_index}, got {chunk_index}.")

        try:
            chunk = base64.urlsafe_b64decode(chunk_b64.encode("ascii"))
        except binascii.Error as exc:
            raise ValueError("Received file chunk base64 is invalid.") from exc

        state.file_handle.write(chunk)
        state.received_size += len(chunk)
        state.next_chunk_index += 1
        self.send_control_message("ACK_CHUNK", transfer_id, str(chunk_index))

    def handle_file_end(self, parts: list[str]) -> None:
        if len(parts) != 1:
            raise ValueError("FILE END message is malformed.")

        transfer_id = parts[0]
        state = self.receive_transfers.pop(transfer_id, None)
        if state is None:
            raise ValueError("Received a file end marker without a matching transfer.")

        state.file_handle.close()

        if state.received_size != state.expected_size:
            self.ui_events.put(
                ("log", f"[received file incomplete] {state.original_name} ({state.received_size}/{state.expected_size} bytes) -> {state.path}")
            )
            return

        self.send_control_message("ACK_END", transfer_id)
        self.ui_events.put(("log", f"[received file] {state.original_name} -> {state.path}"))

    def handle_file_abort(self, parts: list[str]) -> None:
        if len(parts) != 2:
            raise ValueError("FILE ABORT message is malformed.")

        transfer_id, reason_b64 = parts
        reason = decode_control_text(reason_b64)
        state = self.receive_transfers.pop(transfer_id, None)
        if state is None:
            self.ui_events.put(("log", f"[receive aborted] {reason}"))
            return

        state.file_handle.close()
        self.ui_events.put(("log", f"[receive aborted] {state.original_name}: {reason}"))

    def handle_input_start(self, parts: list[str]) -> None:
        if len(parts) != 1:
            raise ValueError("INPUT START message is malformed.")

        requested_session_id = parts[0]
        accept_request = False
        reply_busy = False
        log_message = None
        ack_parts = [requested_session_id]

        with self.input_lock:
            state = self.input_state
            local_session_id = self.local_input_session_id
            remote_session_id = self.remote_input_session_id
            backend_ready = self.input_receive_ready

            if not backend_ready or self.file_send_active:
                reply_busy = True
                log_message = "[input share] rejected remote control request while busy."
            elif state == INPUT_STATE_IDLE:
                self.remote_input_session_id = requested_session_id
                self.input_state = INPUT_STATE_CONTROLLED
                self.clear_input_notice_locked()
                self.reset_auto_edge_hold_locked()
                accept_request = True
                log_message = "[input share] peer started controlling this machine."
            elif state == INPUT_STATE_REQUESTING:
                if local_session_id and requested_session_id < local_session_id:
                    self.local_input_session_id = ""
                    self.local_input_request_deadline = 0.0
                    self.clear_local_capture_state_locked()
                    self.remote_input_session_id = requested_session_id
                    self.input_state = INPUT_STATE_CONTROLLED
                    self.clear_input_notice_locked()
                    self.reset_auto_edge_hold_locked()
                    accept_request = True
                    log_message = "[input share] local request yielded to the peer."
                else:
                    reply_busy = True
                    log_message = "[input share] remote control request rejected because this side is already requesting control."
            elif state == INPUT_STATE_CONTROLLED and remote_session_id == requested_session_id:
                accept_request = True
            else:
                reply_busy = True
                log_message = "[input share] remote control request rejected because this side is already busy."

        if accept_request:
            pointer_context = self.capture_local_pointer_context()
            if pointer_context is not None:
                coordinate_mode, x, y, rects = pointer_context
                ack_parts.extend([coordinate_mode, str(x), str(y), encode_control_text(serialize_display_rects(rects))])

            self.send_control_message("INPUT_ACK", *ack_parts)
            if log_message:
                self.ui_events.put(("log", log_message))
            self.queue_refresh_input_ui()
            self.update_windows_listener_suppression()
            return

        if reply_busy:
            self.send_control_message("INPUT_BUSY", requested_session_id)
            if log_message:
                self.ui_events.put(("log", log_message))

    def handle_input_ack(self, parts: list[str]) -> None:
        if len(parts) not in {1, 5}:
            raise ValueError("INPUT ACK message is malformed.")

        session_id = parts[0]
        log_message = None
        remote_pointer_context: tuple[str, int, int, tuple[DisplayRect, ...]] | None = None

        if len(parts) == 5:
            coordinate_mode = parts[1]
            x = int(parts[2])
            y = int(parts[3])
            rects = deserialize_display_rects(decode_control_text(parts[4]))
            if not rects:
                raise ValueError("INPUT ACK display rect snapshot is empty.")
            x, y = self.constrain_point_to_rects(rects, x, y)
            remote_pointer_context = (coordinate_mode, x, y, rects)

        with self.input_lock:
            if self.input_state == INPUT_STATE_REQUESTING and self.local_input_session_id == session_id:
                self.input_state = INPUT_STATE_CONTROLLING
                self.local_input_request_deadline = 0.0
                self.clear_input_notice_locked()
                anchor_side = ""
                if self.pending_auto_edge_anchor_session_id == session_id:
                    anchor_side = self.pending_auto_edge_anchor_side
                self.clear_pending_auto_edge_anchor_locked()
                if remote_pointer_context is not None:
                    coordinate_mode, x, y, rects = remote_pointer_context
                    self.remote_pointer_coordinate_mode = coordinate_mode
                    self.remote_pointer_position = (x, y)
                    self.remote_display_rects = rects
                log_message = "[input share] remote control active."
            else:
                return

        self.reset_local_mouse_anchor(anchor_side or None)
        self.ui_events.put(("log", log_message))
        self.queue_refresh_input_ui()
        self.update_windows_listener_suppression()

    def handle_input_busy(self, parts: list[str]) -> None:
        if len(parts) != 1:
            raise ValueError("INPUT BUSY message is malformed.")

        session_id = parts[0]
        with self.input_lock:
            if self.input_state == INPUT_STATE_REQUESTING and self.local_input_session_id == session_id:
                self.input_state = INPUT_STATE_IDLE
                self.local_input_session_id = ""
                self.local_input_request_deadline = 0.0
                self.clear_local_capture_state_locked()
                self.clear_pending_auto_edge_anchor_locked()
                self.input_notice_message = "Busy"
                self.input_notice_deadline = time.monotonic() + INPUT_NOTICE_SECONDS
            else:
                return

        self.ui_events.put(("log", "[input share] peer is busy."))
        self.queue_refresh_input_ui()
        self.update_windows_listener_suppression()

    def handle_input_stop(self, parts: list[str]) -> None:
        if len(parts) != 1:
            raise ValueError("INPUT STOP message is malformed.")

        self.finish_remote_input_control(parts[0], log_message="[input share] peer stopped remote control.")

    def handle_input_release_all(self, parts: list[str]) -> None:
        if len(parts) != 1:
            raise ValueError("INPUT RELEASE ALL message is malformed.")

        session_id = parts[0]
        with self.input_lock:
            if self.input_state != INPUT_STATE_CONTROLLED or self.remote_input_session_id != session_id:
                return

        self.release_remote_inputs()

    def handle_input_key(self, parts: list[str]) -> None:
        if len(parts) != 3:
            raise ValueError("INPUT KEY message is malformed.")

        session_id, action, encoded_key = parts
        if action not in {"down", "up"}:
            raise ValueError("INPUT KEY action is malformed.")

        if not self.remote_session_matches(session_id):
            return

        key_token = decode_control_text(encoded_key)
        if action == "down":
            self.inject_remote_key_down(key_token)
        else:
            self.inject_remote_key_up(key_token)

    def handle_input_mouse_move(self, parts: list[str]) -> None:
        if len(parts) != 3:
            raise ValueError("INPUT MOUSE MOVE message is malformed.")

        session_id, dx_text, dy_text = parts
        if not self.remote_session_matches(session_id):
            return

        self.inject_remote_mouse_move(int(dx_text), int(dy_text))

    def handle_input_mouse_button(self, parts: list[str]) -> None:
        if len(parts) != 3:
            raise ValueError("INPUT MOUSE BUTTON message is malformed.")

        session_id, encoded_button, action = parts
        if action not in {"down", "up"}:
            raise ValueError("INPUT MOUSE BUTTON action is malformed.")

        if not self.remote_session_matches(session_id):
            return

        button_token = decode_control_text(encoded_button)
        if action == "down":
            self.inject_remote_mouse_button_down(button_token)
        else:
            self.inject_remote_mouse_button_up(button_token)

    def handle_input_mouse_scroll(self, parts: list[str]) -> None:
        if len(parts) != 3:
            raise ValueError("INPUT MOUSE SCROLL message is malformed.")

        session_id, dx_text, dy_text = parts
        if not self.remote_session_matches(session_id):
            return

        self.inject_remote_mouse_scroll(int(dx_text), int(dy_text))

    def remote_session_matches(self, session_id: str) -> bool:
        with self.input_lock:
            return self.input_state == INPUT_STATE_CONTROLLED and self.remote_input_session_id == session_id

    def close_receive_transfers(self, interrupted: bool) -> None:
        for transfer_id, state in list(self.receive_transfers.items()):
            try:
                state.file_handle.close()
            except OSError:
                pass

            if interrupted:
                self.ui_events.put(("log", f"[receive interrupted] {state.original_name} -> {state.path}"))

            del self.receive_transfers[transfer_id]

    def build_received_file_path(self, filename: str) -> Path:
        safe_name = Path(filename).name or "received_file.bin"
        candidate = RECEIVED_FILES_DIR / safe_name

        if not candidate.exists():
            return candidate

        stem = candidate.stem or "received_file"
        suffix = candidate.suffix
        index = 1
        while True:
            next_candidate = candidate.with_name(f"{stem}_{index}{suffix}")
            if not next_candidate.exists():
                return next_candidate
            index += 1

    def process_ui_events(self) -> None:
        try:
            while True:
                kind, payload = self.ui_events.get_nowait()

                if kind == "text":
                    message = str(payload)
                    self.last_received_raw = message
                    self.copy_button.configure(state="normal")
                    self.copy_decoded_button.configure(state="normal")
                    self.append_log(message)
                elif kind == "log":
                    self.append_log(str(payload))
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "disconnect":
                    self.disconnect()
                elif kind == "file-send-state":
                    self.file_send_active = bool(payload)
                    self.update_controls()
                elif kind == "refresh-status":
                    self.refresh_connection_status()
                elif kind == "refresh-input-ui":
                    self.refresh_input_ui()
                    self.update_controls()
                elif kind == "toggle-input":
                    self.toggle_remote_control()
                elif kind == "copy-last":
                    self.copy_last_received()
                elif kind == "copy-last-decoded":
                    self.copy_last_received_after_decode()
                elif kind == "send-plain-global":
                    self.send_message("plain")
                elif kind == "send-encoded-global":
                    self.send_message("encoded")
                elif kind == "auto-edge-enter":
                    self.handle_auto_edge_enter_action()
                elif kind == "auto-edge-exit":
                    self.handle_auto_edge_exit_action()
                elif kind == "force-stop-input":
                    self.stop_local_input_control(send_remote_stop=True, log_message="[input share] emergency stop.")
                elif kind == "force-close":
                    self.on_close()
                    return
        except queue.Empty:
            pass

        self.check_input_timers()
        self.root.after(100, self.process_ui_events)

    def send_plain_shortcut(self, event: tk.Event) -> str:
        del event
        self.send_message("plain")
        return "break"

    def send_encoded_shortcut(self, event: tk.Event) -> str:
        del event
        self.send_message("encoded")
        return "break"

    def handle_auto_edge_enter_action(self) -> None:
        if not self.auto_edge_enabled or not self.serial_connected():
            return

        if self.file_send_active:
            return

        with self.input_lock:
            if self.input_state != INPUT_STATE_IDLE:
                return
            self.arm_auto_edge_anchor_on_next_request = True

        self.append_log(f"[input share] auto edge entering toward {self.current_peer_side().lower()}.")
        self.toggle_remote_control()

    def handle_auto_edge_exit_action(self) -> None:
        if not self.auto_edge_enabled or not self.serial_connected():
            return

        with self.input_lock:
            if self.input_state != INPUT_STATE_CONTROLLING:
                return

        self.stop_local_input_control(send_remote_stop=True, log_message="[input share] remote control stopped by auto edge.")

    def send_message(self, mode: str) -> None:
        if mode not in {"plain", "encoded"}:
            raise ValueError(f"Unsupported message mode: {mode}")

        message = self.message_text.get("1.0", "end-1c")
        if not message.strip():
            return

        payload = message if mode == "plain" else obfuscate_text(message)

        try:
            self.send_text_message(payload)
        except serial.SerialException as exc:
            messagebox.showerror("Send Error", str(exc))
            self.disconnect()
            return

        self.append_log(payload)
        self.message_text.delete("1.0", "end")
        self.message_text.focus_set()

    def send_text_message(self, payload: str) -> None:
        self.send_serial_frame(build_text_message(payload))

    def send_serial_frame(self, frame_text: str) -> None:
        payload = frame_text.encode("utf-8") + MESSAGE_TERMINATOR

        with self.write_lock:
            port = self.serial_port
            if not port or not port.is_open:
                raise serial.SerialException("Connect to a serial port first.")

            port.write(payload)
            port.flush()

        if self.connected_baudrate >= HIGH_SPEED_BAUD_THRESHOLD:
            time.sleep(HIGH_SPEED_FRAME_DELAY_SECONDS)

    def select_files_for_send(self) -> None:
        file_paths = filedialog.askopenfilenames(parent=self.root, title="Select files to send")
        if not file_paths:
            return

        self.start_file_send([Path(path) for path in file_paths])

    def open_received_files_folder(self) -> None:
        RECEIVED_FILES_DIR.mkdir(parents=True, exist_ok=True)

        try:
            if sys.platform == "win32":
                os.startfile(str(RECEIVED_FILES_DIR))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(RECEIVED_FILES_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(RECEIVED_FILES_DIR)])
        except Exception as exc:
            messagebox.showerror("Open Folder Error", str(exc))

    def handle_drag_enter(self, event: tk.Event) -> str:
        del event
        input_state, _ = self.input_state_snapshot()
        if self.serial_connected() and not self.file_send_active and input_state == INPUT_STATE_IDLE:
            self.drop_zone.configure(bg="#dff3ff")
        return "break"

    def handle_drag_leave(self, event: tk.Event) -> str:
        del event
        self.update_drop_zone_state()
        return "break"

    def handle_file_drop(self, event: tk.Event) -> str:
        if not self.serial_connected():
            self.update_drop_zone_state()
            return "break"

        raw_paths = self.root.tk.splitlist(event.data)
        paths = [normalize_dropped_path(raw_path) for raw_path in raw_paths]
        self.update_drop_zone_state()
        self.start_file_send(paths)
        return "break"

    def start_file_send(self, paths: list[Path]) -> None:
        if not self.serial_connected():
            messagebox.showerror("File Send Error", "Connect to a serial port first.")
            return

        input_state, _ = self.input_state_snapshot()
        if input_state != INPUT_STATE_IDLE:
            messagebox.showerror("File Send Error", "Stop remote control before sending files.")
            return

        if self.file_send_active:
            messagebox.showerror("File Send Error", "A file transfer is already in progress.")
            return

        files = [path for path in paths if path.is_file()]
        if not files:
            messagebox.showerror("File Send Error", "No regular files were selected.")
            return

        self.reset_auto_edge_hold()
        self.file_send_active = True
        self.update_controls()
        self.file_send_thread = threading.Thread(target=self.send_files_worker, args=(files,), daemon=True)
        self.file_send_thread.start()

    def send_files_worker(self, files: list[Path]) -> None:
        self.ui_events.put(("status", f"Sending {len(files)} file(s)..."))
        self.ui_events.put(("file-send-state", True))

        try:
            for path in files:
                if self.stop_event.is_set():
                    raise RuntimeError("Connection closed.")

                self.send_single_file(path)
        except Exception as exc:
            self.ui_events.put(("log", f"[send error] {exc}"))
        finally:
            self.ui_events.put(("file-send-state", False))
            self.ui_events.put(("refresh-status", None))

    def send_single_file(self, path: Path) -> None:
        transfer_id = uuid.uuid4().hex
        file_size = path.stat().st_size
        self.ui_events.put(("log", f"[sending file] {path.name} ({file_size} bytes)"))
        self.ui_events.put(("status", f"Sending {path.name} ({file_size} bytes)"))

        try:
            self.send_control_message("START", transfer_id, encode_control_text(path.name), str(file_size))
            self.wait_for_file_ack("ACK_START", transfer_id)

            with path.open("rb") as file_handle:
                chunk_index = 0
                while True:
                    if self.stop_event.is_set():
                        raise RuntimeError("Connection closed.")

                    chunk = file_handle.read(FILE_CHUNK_SIZE)
                    if not chunk:
                        break

                    chunk_b64 = base64.urlsafe_b64encode(chunk).decode("ascii")
                    self.send_control_message("CHUNK", transfer_id, str(chunk_index), chunk_b64)
                    self.wait_for_file_ack("ACK_CHUNK", transfer_id, str(chunk_index))
                    chunk_index += 1

            self.send_control_message("END", transfer_id)
            self.wait_for_file_ack("ACK_END", transfer_id)
            self.ui_events.put(("log", f"[sent file] {path.name}"))
        except Exception as exc:
            try:
                self.send_control_message("ABORT", transfer_id, encode_control_text(str(exc)))
            except Exception:
                pass
            raise

    def send_control_message(self, command: str, *parts: str) -> None:
        self.send_serial_frame(build_control_message(command, *parts))

    def wait_for_file_ack(self, expected_command: str, transfer_id: str, expected_value: str | None = None) -> None:
        deadline = time.monotonic() + FILE_ACK_TIMEOUT_SECONDS

        while True:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError(f"Timed out waiting for {expected_command}.")

            try:
                command, parts = self.transfer_ack_queue.get(timeout=timeout)
            except queue.Empty as exc:
                raise TimeoutError(f"Timed out waiting for {expected_command}.") from exc

            if command != expected_command:
                continue

            if not parts or parts[0] != transfer_id:
                continue

            if expected_value is not None:
                if len(parts) < 2 or parts[1] != expected_value:
                    continue

            return

    def input_sender_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                item = self.input_outbound_queue.get(timeout=INPUT_SENDER_POLL_SECONDS)
            except queue.Empty:
                item = None

            self.flush_pending_mouse_move()

            if item is None:
                continue

            command, session_id, parts = item
            if command == "__shutdown__":
                break

            if not self.should_send_input_for_session(session_id):
                continue

            try:
                self.send_control_message(command, session_id, *parts)
            except serial.SerialException as exc:
                self.ui_events.put(("log", f"[input send error] {exc}"))
                self.ui_events.put(("disconnect", None))
                return
            except Exception as exc:
                self.ui_events.put(("log", f"[input send error] {exc}"))

    def should_send_input_for_session(self, session_id: str) -> bool:
        with self.input_lock:
            return (
                self.input_state == INPUT_STATE_CONTROLLING
                and self.local_input_session_id == session_id
                and self.serial_connected()
                and not self.stop_event.is_set()
            )

    def flush_pending_mouse_move(self) -> None:
        with self.input_lock:
            if self.input_state != INPUT_STATE_CONTROLLING or not self.local_input_session_id:
                self.pending_mouse_dx = 0
                self.pending_mouse_dy = 0
                return

            dx = self.pending_mouse_dx
            dy = self.pending_mouse_dy
            session_id = self.local_input_session_id
            self.pending_mouse_dx = 0
            self.pending_mouse_dy = 0

        if dx == 0 and dy == 0:
            return

        try:
            self.send_control_message("INPUT_MOUSE_MOVE", session_id, str(dx), str(dy))
        except serial.SerialException as exc:
            self.ui_events.put(("log", f"[input send error] {exc}"))
            self.ui_events.put(("disconnect", None))
        except Exception as exc:
            self.ui_events.put(("log", f"[input send error] {exc}"))

    def queue_input_command(self, command: str, session_id: str, *parts: str) -> None:
        self.input_outbound_queue.put((command, session_id, parts))

    def inset_anchor_from_side(self, anchor: tuple[int, int], side: str | None) -> tuple[int, int]:
        if side == AUTO_EDGE_SIDE_RIGHT:
            return anchor[0] - AUTO_EDGE_ENTRY_ANCHOR_INSET_PIXELS, anchor[1]
        if side == AUTO_EDGE_SIDE_LEFT:
            return anchor[0] + AUTO_EDGE_ENTRY_ANCHOR_INSET_PIXELS, anchor[1]
        if side == AUTO_EDGE_SIDE_TOP:
            return anchor[0], anchor[1] + AUTO_EDGE_ENTRY_ANCHOR_INSET_PIXELS
        if side == AUTO_EDGE_SIDE_BOTTOM:
            return anchor[0], anchor[1] - AUTO_EDGE_ENTRY_ANCHOR_INSET_PIXELS
        return anchor

    def reset_local_mouse_anchor(self, inset_side: str | None = None) -> None:
        anchor = self.get_local_pointer_position()
        should_warp = anchor is not None and inset_side in AUTO_EDGE_SIDE_OPTIONS
        if anchor is not None and should_warp:
            anchor = self.inset_anchor_from_side(anchor, inset_side)
        with self.input_lock:
            self.mouse_anchor = anchor
            self.mouse_warp_events_to_ignore = INPUT_WARP_IGNORE_EVENTS if should_warp else 0

        if should_warp:
            self.warp_local_pointer(anchor)

    def get_local_pointer_position(self) -> tuple[int, int] | None:
        controller = self.mouse_controller
        if controller is None:
            return None

        try:
            x, y = controller.position
        except Exception:
            return None

        return int(x), int(y)

    def is_emergency_stop_combo_active(self, trigger_token: str) -> bool:
        if trigger_token != EMERGENCY_STOP_KEY_TOKEN:
            return False

        with self.input_lock:
            return EMERGENCY_MODIFIER_TOKENS.issubset(self.hotkey_pressed_key_tokens)

    def is_emergency_exit_combo_active(self, trigger_token: str) -> bool:
        if trigger_token != EMERGENCY_EXIT_KEY_TOKEN:
            return False

        with self.input_lock:
            return EMERGENCY_MODIFIER_TOKENS.issubset(self.hotkey_pressed_key_tokens)

    def decode_key_token_char(self, key_token: str) -> str | None:
        if not key_token.startswith("char:"):
            return None

        try:
            return decode_control_text(key_token[5:]).lower()
        except Exception:
            return None

    def get_app_hotkey_action(self, key_token: str) -> str | None:
        char = self.decode_key_token_char(key_token)
        if char is None:
            return None

        with self.input_lock:
            if key_token in self.active_app_hotkey_key_tokens:
                return None
            active_tokens = set(self.hotkey_pressed_key_tokens)

        if not APP_HOTKEY_MODIFIER_TOKENS.issubset(active_tokens):
            return None

        if char in APP_COPY_TRIGGER_CHARS:
            return "copy-last-decoded" if "special:shift" in active_tokens else "copy-last"
        if char in APP_SEND_TRIGGER_CHARS:
            return "send-encoded-global" if "special:shift" in active_tokens else "send-plain-global"
        return None

    def key_token_requires_physical_hold(self, key_token: str) -> bool:
        return canonical_key_token(key_token) in {"special:ctrl", "special:alt", "special:shift", "special:cmd"}

    def key_token_supports_repeat(self, key_token: str) -> bool:
        if key_token.startswith("char:"):
            return True

        return canonical_key_token(key_token) in {
            "special:space",
            "special:tab",
            "special:enter",
            "special:backspace",
            "special:delete",
            "special:left",
            "special:right",
            "special:up",
            "special:down",
            "special:home",
            "special:end",
            "special:page_up",
            "special:page_down",
        }

    def handle_global_key_press(self, key: object, injected: bool = False) -> None:
        if injected:
            return

        with self.input_lock:
            state = self.input_state
            session_id = self.local_input_session_id

        key_token: str | None = None
        canonical_token: str | None = None
        try:
            key_token = encode_key_token(key)
            canonical_token = canonical_key_token(key_token)
        except Exception:
            pass

        if canonical_token is not None:
            with self.input_lock:
                self.hotkey_pressed_key_tokens.add(canonical_token)

            if self.is_emergency_exit_combo_active(canonical_token):
                self.ui_events.put(("force-close", None))
                return

            if self.is_emergency_stop_combo_active(canonical_token):
                self.ui_events.put(("force-stop-input", None))
                return

        if key_token is not None and sys.platform != "darwin":
            app_hotkey_action = self.get_app_hotkey_action(key_token)
            if app_hotkey_action is not None:
                with self.input_lock:
                    self.active_app_hotkey_key_tokens.add(key_token)
                self.ui_events.put((app_hotkey_action, None))
                return

        if self.is_toggle_key(key) and not (sys.platform == "darwin" and self.mac_hotkey_backend_ready):
            self.ui_events.put(("toggle-input", None))
            return

        if state != INPUT_STATE_CONTROLLING or not session_id or key_token is None:
            return

        with self.input_lock:
            if key_token in self.local_pressed_key_tokens:
                return
            self.local_pressed_key_tokens.add(key_token)

        self.queue_input_command("INPUT_KEY", session_id, "down", encode_control_text(key_token))

    def handle_global_key_release(self, key: object, injected: bool = False) -> None:
        if injected:
            return

        with self.input_lock:
            state = self.input_state
            session_id = self.local_input_session_id

        key_token: str | None = None
        canonical_token: str | None = None
        try:
            key_token = encode_key_token(key)
            canonical_token = canonical_key_token(key_token)
        except Exception:
            pass

        if canonical_token is not None:
            with self.input_lock:
                self.hotkey_pressed_key_tokens.discard(canonical_token)
                if key_token is not None:
                    self.active_app_hotkey_key_tokens.discard(key_token)

        if self.is_toggle_key(key) and not (sys.platform == "darwin" and self.mac_hotkey_backend_ready):
            return

        if state != INPUT_STATE_CONTROLLING or not session_id or key_token is None:
            return

        with self.input_lock:
            if key_token not in self.local_pressed_key_tokens:
                return
            self.local_pressed_key_tokens.discard(key_token)

        self.queue_input_command("INPUT_KEY", session_id, "up", encode_control_text(key_token))

    def handle_global_mouse_move(self, x: float, y: float, injected: bool = False) -> None:
        if injected:
            return

        with self.input_lock:
            state = self.input_state
            session_id = self.local_input_session_id

        if state != INPUT_STATE_CONTROLLING or not session_id:
            self.handle_idle_edge_pointer_move(int(x), int(y), coordinate_mode="screen")
            return

        dx = 0
        dy = 0
        anchor: tuple[int, int] | None = None
        remote_pointer: tuple[int, int, str] | None = None
        with self.input_lock:
            if self.mouse_anchor is None:
                self.mouse_anchor = (int(x), int(y))
                return

            anchor_x, anchor_y = self.mouse_anchor
            if self.mouse_warp_events_to_ignore > 0 and abs(int(x) - anchor_x) <= 1 and abs(int(y) - anchor_y) <= 1:
                self.mouse_warp_events_to_ignore -= 1
                return

            dx = int(round(x - anchor_x))
            dy = int(round(y - anchor_y))
            if dx == 0 and dy == 0:
                return

            self.pending_mouse_dx += dx
            self.pending_mouse_dy += dy
            self.mouse_warp_events_to_ignore = INPUT_WARP_IGNORE_EVENTS
            anchor = self.mouse_anchor
            remote_pointer = self.advance_remote_pointer_locked(dx, dy)

        self.handle_auto_edge_exit_motion(dx, dy, remote_pointer)
        self.warp_local_pointer(anchor)

    def handle_global_mouse_click(self, x: float, y: float, button: object, pressed: bool, injected: bool = False) -> None:
        del x, y
        if injected:
            return

        with self.input_lock:
            if self.input_state != INPUT_STATE_CONTROLLING or not self.local_input_session_id:
                return
            session_id = self.local_input_session_id

        try:
            button_token = encode_mouse_button_token(button)
        except Exception:
            return

        with self.input_lock:
            if pressed:
                if button_token in self.local_pressed_mouse_tokens:
                    return
                self.local_pressed_mouse_tokens.add(button_token)
                action = "down"
            else:
                if button_token not in self.local_pressed_mouse_tokens:
                    return
                self.local_pressed_mouse_tokens.discard(button_token)
                action = "up"

        self.queue_input_command("INPUT_MOUSE_BUTTON", session_id, encode_control_text(button_token), action)

    def handle_global_mouse_scroll(self, x: float, y: float, dx: float, dy: float, injected: bool = False) -> None:
        del x, y
        if injected:
            return

        with self.input_lock:
            if self.input_state != INPUT_STATE_CONTROLLING or not self.local_input_session_id:
                return
            session_id = self.local_input_session_id

        self.queue_input_command(
            "INPUT_MOUSE_SCROLL",
            session_id,
            str(int(dx * REMOTE_MOUSE_SCROLL_MULTIPLIER)),
            str(int(dy * REMOTE_MOUSE_SCROLL_MULTIPLIER)),
        )

    def warp_local_pointer(self, anchor: tuple[int, int] | None) -> None:
        controller = self.mouse_controller
        if controller is None or anchor is None:
            return

        try:
            controller.position = anchor
        except Exception:
            pass

    def is_toggle_key(self, key: object) -> bool:
        key_name = getattr(key, "name", "")
        key_vk = getattr(key, "vk", None)

        if sys.platform == "darwin":
            return key_name == "f8" or key_vk == DARWIN_F8_KEYCODE

        if sys.platform == "win32":
            return key_name == "scroll_lock" or key_vk == WINDOWS_SCROLL_LOCK_VK or str(key).lower() == "key.scroll_lock"

        return False

    def darwin_event_is_injected(self, event: object) -> bool:
        if Quartz is None:
            return False

        try:
            return bool(Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUnixProcessID))
        except Exception:
            return False

    def darwin_event_keycode(self, event: object) -> int | None:
        if Quartz is None:
            return None

        try:
            return int(Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode))
        except Exception:
            return None

    def darwin_keyboard_intercept(self, event_type: int, event: object) -> object | None:
        del event_type
        if self.darwin_event_is_injected(event):
            return event

        if self.darwin_event_keycode(event) == DARWIN_F8_KEYCODE:
            return None

        with self.input_lock:
            if self.input_state == INPUT_STATE_CONTROLLING:
                return None

        return event

    def darwin_mouse_intercept(self, event_type: int, event: object) -> object | None:
        del event_type
        if self.darwin_event_is_injected(event):
            return event

        with self.input_lock:
            if self.input_state == INPUT_STATE_CONTROLLING:
                return None

        return event

    def win32_keyboard_filter(self, msg: int, data: object) -> bool:
        del msg
        vk_code = getattr(data, "vkCode", None)
        flags = getattr(data, "flags", 0)

        if flags & WINDOWS_KEY_FLAG_INJECTED:
            return True

        if vk_code == WINDOWS_SCROLL_LOCK_VK:
            return True

        return True

    def win32_mouse_filter(self, msg: int, data: object) -> bool:
        del msg
        flags = getattr(data, "flags", 0)
        if flags & WINDOWS_MOUSE_FLAG_INJECTED:
            return True

        return True

    def inject_remote_key_down(self, key_token: str) -> None:
        controller = self.keyboard_controller
        if controller is None:
            return

        canonical_token = canonical_key_token(key_token)
        should_hold = self.key_token_requires_physical_hold(key_token)
        should_repeat = self.key_token_supports_repeat(key_token)
        with self.input_lock:
            if key_token in self.remote_pressed_key_tokens:
                return
            self.remote_pressed_key_tokens.add(key_token)

        try:
            decoded_key = decode_key_token(key_token)
            if canonical_token == "special:caps_lock":
                controller.tap(decoded_key)
            elif should_hold:
                controller.press(decoded_key)
                with self.input_lock:
                    self.remote_physical_key_tokens.add(key_token)
            else:
                controller.tap(decoded_key)
                if should_repeat:
                    with self.input_lock:
                        self.remote_repeat_deadlines[key_token] = time.monotonic() + REMOTE_KEY_REPEAT_INITIAL_DELAY_SECONDS
        except Exception as exc:
            with self.input_lock:
                self.remote_pressed_key_tokens.discard(key_token)
                self.remote_physical_key_tokens.discard(key_token)
                self.remote_repeat_deadlines.pop(key_token, None)
            self.ui_events.put(("log", f"[input share] key press failed: {exc}"))

    def inject_remote_key_up(self, key_token: str) -> None:
        controller = self.keyboard_controller
        if controller is None:
            return

        with self.input_lock:
            if key_token not in self.remote_pressed_key_tokens:
                return
            self.remote_pressed_key_tokens.discard(key_token)
            should_release = key_token in self.remote_physical_key_tokens
            self.remote_physical_key_tokens.discard(key_token)
            self.remote_repeat_deadlines.pop(key_token, None)

        if canonical_key_token(key_token) == "special:caps_lock":
            return

        if not should_release:
            return

        try:
            controller.release(decode_key_token(key_token))
        except Exception as exc:
            self.ui_events.put(("log", f"[input share] key release failed: {exc}"))

    def inject_remote_key_repeat(self, key_token: str) -> None:
        controller = self.keyboard_controller
        if controller is None:
            return

        if self.key_token_requires_physical_hold(key_token) or not self.key_token_supports_repeat(key_token):
            return

        try:
            controller.tap(decode_key_token(key_token))
        except Exception as exc:
            self.ui_events.put(("log", f"[input share] key repeat failed: {exc}"))

    def remote_key_repeat_loop(self) -> None:
        while not self.shutdown_event.is_set():
            repeat_tokens: list[str] = []
            now = time.monotonic()

            with self.input_lock:
                for key_token, deadline in list(self.remote_repeat_deadlines.items()):
                    if key_token not in self.remote_pressed_key_tokens:
                        self.remote_repeat_deadlines.pop(key_token, None)
                        continue

                    if deadline > now:
                        continue

                    repeat_tokens.append(key_token)
                    self.remote_repeat_deadlines[key_token] = now + REMOTE_KEY_REPEAT_INTERVAL_SECONDS

            for key_token in repeat_tokens:
                self.inject_remote_key_repeat(key_token)

            time.sleep(REMOTE_KEY_REPEAT_POLL_SECONDS)

    def inject_remote_mouse_move(self, dx: int, dy: int) -> None:
        controller = self.mouse_controller
        if controller is None:
            return

        try:
            controller.move(dx, dy)
        except Exception as exc:
            self.ui_events.put(("log", f"[input share] mouse move failed: {exc}"))

    def inject_remote_mouse_button_down(self, button_token: str) -> None:
        controller = self.mouse_controller
        if controller is None:
            return

        with self.input_lock:
            if button_token in self.remote_pressed_mouse_tokens:
                return
            self.remote_pressed_mouse_tokens.add(button_token)

        try:
            controller.press(decode_mouse_button_token(button_token))
        except Exception as exc:
            with self.input_lock:
                self.remote_pressed_mouse_tokens.discard(button_token)
            self.ui_events.put(("log", f"[input share] mouse button press failed: {exc}"))

    def inject_remote_mouse_button_up(self, button_token: str) -> None:
        controller = self.mouse_controller
        if controller is None:
            return

        with self.input_lock:
            if button_token not in self.remote_pressed_mouse_tokens:
                return
            self.remote_pressed_mouse_tokens.discard(button_token)

        try:
            controller.release(decode_mouse_button_token(button_token))
        except Exception as exc:
            self.ui_events.put(("log", f"[input share] mouse button release failed: {exc}"))

    def inject_remote_mouse_scroll(self, dx: int, dy: int) -> None:
        controller = self.mouse_controller
        if controller is None:
            return

        try:
            controller.scroll(dx, dy)
        except Exception as exc:
            self.ui_events.put(("log", f"[input share] mouse scroll failed: {exc}"))

    def release_remote_inputs(self) -> None:
        controller_keyboard = self.keyboard_controller
        controller_mouse = self.mouse_controller
        if controller_keyboard is None and controller_mouse is None:
            return

        with self.input_lock:
            key_tokens = list(self.remote_physical_key_tokens)
            button_tokens = list(self.remote_pressed_mouse_tokens)
            self.remote_pressed_key_tokens.clear()
            self.remote_physical_key_tokens.clear()
            self.remote_pressed_mouse_tokens.clear()
            self.remote_repeat_deadlines.clear()

        for button_token in button_tokens:
            if controller_mouse is None:
                break
            try:
                controller_mouse.release(decode_mouse_button_token(button_token))
            except Exception:
                pass

        for key_token in key_tokens:
            if controller_keyboard is None:
                break
            if canonical_key_token(key_token) == "special:caps_lock":
                continue
            try:
                controller_keyboard.release(decode_key_token(key_token))
            except Exception:
                pass

    def bind_editor_shortcuts(self) -> None:
        control_shortcuts = {
            "<Control-a>": self.select_all_message,
            "<Control-A>": self.select_all_message,
            "<Control-c>": self.copy_message_selection,
            "<Control-C>": self.copy_message_selection,
            "<Control-x>": self.cut_message_selection,
            "<Control-X>": self.cut_message_selection,
            "<Control-v>": self.paste_into_message,
            "<Control-V>": self.paste_into_message,
            "<Shift-Insert>": self.paste_into_message,
            "<Control-z>": self.undo_message_edit,
            "<Control-Z>": self.redo_message_edit,
            "<Control-y>": self.redo_message_edit,
            "<Control-Y>": self.redo_message_edit,
        }
        command_shortcuts = {
            "<Command-a>": self.select_all_message,
            "<Command-A>": self.select_all_message,
            "<Command-c>": self.copy_message_selection,
            "<Command-C>": self.copy_message_selection,
            "<Command-x>": self.cut_message_selection,
            "<Command-X>": self.cut_message_selection,
            "<Command-v>": self.paste_into_message,
            "<Command-V>": self.paste_into_message,
            "<Command-z>": self.undo_message_edit,
            "<Command-Z>": self.redo_message_edit,
            "<Command-y>": self.redo_message_edit,
            "<Command-Y>": self.redo_message_edit,
        }

        for sequence, handler in control_shortcuts.items():
            self.message_text.bind(sequence, handler)

        if sys.platform == "darwin":
            self.message_text.bind("<FocusOut>", self.clear_editor_modifier_state, add="+")
            self.message_text.bind("<KeyPress>", self.handle_darwin_editor_control_char, add="+")
            self.message_text.bind("<KeyPress>", self.handle_darwin_editor_modifier_tracking, add="+")
            self.message_text.bind("<KeyRelease>", self.handle_darwin_editor_modifier_tracking, add="+")
            self.message_text.bind("<Control-KeyPress>", self.handle_darwin_control_editor_shortcut, add="+")
            self.message_text.bind("<Command-KeyPress>", self.handle_darwin_command_editor_shortcut, add="+")
            self.root.bind_all("<KeyPress>", self.handle_darwin_editor_control_char, add="+")
            self.root.bind_all("<KeyPress>", self.handle_darwin_editor_modifier_tracking, add="+")
            self.root.bind_all("<KeyRelease>", self.handle_darwin_editor_modifier_tracking, add="+")
            self.root.bind_all("<Control-KeyPress>", self.handle_darwin_control_editor_shortcut, add="+")

        if sys.platform == "darwin":
            for sequence, handler in command_shortcuts.items():
                self.message_text.bind(sequence, handler)
                self.root.bind_all(sequence, handler, add="+")
            self.root.bind_all("<Command-KeyPress>", self.handle_darwin_command_editor_shortcut, add="+")

    def handle_darwin_control_editor_shortcut(self, event: tk.Event) -> str | None:
        return self.handle_darwin_editor_shortcut(event)

    def handle_darwin_command_editor_shortcut(self, event: tk.Event) -> str | None:
        return self.handle_darwin_editor_shortcut(event)

    def handle_darwin_editor_control_char(self, event: tk.Event) -> str | None:
        if not self.message_editor_matches_event(event):
            return None

        action = DARWIN_CONTROL_CHAR_SHORTCUTS.get(getattr(event, "char", ""))
        if action is None:
            return None

        return self.dispatch_darwin_editor_shortcut(action, event)

    def handle_darwin_editor_modifier_tracking(self, event: tk.Event) -> str | None:
        keysym = str(getattr(event, "keysym", "")).lower()
        modifier = EDITOR_SHORTCUT_MODIFIER_KEYSYMS.get(keysym)
        if modifier is not None:
            if getattr(event, "type", None) == tk.EventType.KeyRelease:
                self.editor_modifier_state.discard(modifier)
            else:
                self.editor_modifier_state.add(modifier)
            return None

        if not self.message_editor_matches_event(event):
            return None

        if not (self.editor_modifier_state & {"control", "command"}):
            return None

        action = self.resolve_darwin_editor_shortcut_action(event)
        if action is None:
            return None

        if action == "undo" and "shift" in self.editor_modifier_state:
            action = "redo"
        return self.dispatch_darwin_editor_shortcut(action, event)

    def handle_darwin_editor_shortcut(self, event: tk.Event) -> str | None:
        if not self.message_editor_matches_event(event):
            return None

        action = self.resolve_darwin_editor_shortcut_action(event)
        if action is None:
            return None

        return self.dispatch_darwin_editor_shortcut(action, event)

    def resolve_darwin_editor_shortcut_action(self, event: tk.Event) -> str | None:
        action = DARWIN_SHORTCUT_KEYCODES.get(getattr(event, "keycode", None))
        if action is not None:
            return action

        char = str(getattr(event, "char", "")).lower()
        if not char:
            return None
        return DARWIN_EDITOR_SHORTCUT_CHARS.get(char)

    def dispatch_darwin_editor_shortcut(self, action: str, event: tk.Event) -> str | None:
        if action == "select_all":
            return self.select_all_message(event)
        if action == "copy":
            return self.copy_message_selection(event)
        if action == "cut":
            return self.cut_message_selection(event)
        if action == "paste":
            return self.paste_into_message(event)
        if action == "undo":
            if getattr(event, "state", 0) & 0x1:
                return self.redo_message_edit(event)
            return self.undo_message_edit(event)
        if action == "redo":
            return self.redo_message_edit(event)

        return None
    def message_editor_has_focus(self) -> bool:
        return self.root.focus_get() is self.message_text

    def message_editor_matches_event(self, event: tk.Event) -> bool:
        return getattr(event, "widget", None) is self.message_text or self.message_editor_has_focus()

    def clear_editor_modifier_state(self, event: tk.Event | None = None) -> None:
        self.editor_modifier_state.clear()

    def select_all_message(self, event: tk.Event) -> str | None:
        if not self.message_editor_matches_event(event):
            return None

        self.message_text.tag_add("sel", "1.0", "end-1c")
        self.message_text.mark_set("insert", "1.0")
        self.message_text.see("insert")
        return "break"

    def copy_message_selection(self, event: tk.Event) -> str | None:
        if not self.message_editor_matches_event(event):
            return None

        try:
            text = self.message_text.get("sel.first", "sel.last")
        except tk.TclError:
            return "break"

        self.copy_text(text)
        return "break"

    def cut_message_selection(self, event: tk.Event) -> str | None:
        if not self.message_editor_matches_event(event):
            return None

        try:
            text = self.message_text.get("sel.first", "sel.last")
            self.message_text.delete("sel.first", "sel.last")
        except tk.TclError:
            return "break"

        self.copy_text(text)
        return "break"

    def paste_into_message(self, event: tk.Event) -> str | None:
        if not self.message_editor_matches_event(event):
            return None

        try:
            clipboard_text = self.get_clipboard_text()
        except tk.TclError:
            return "break"

        try:
            self.message_text.delete("sel.first", "sel.last")
        except tk.TclError:
            pass

        self.message_text.edit_separator()
        self.message_text.insert("insert", clipboard_text)
        self.message_text.see("insert")
        self.message_text.edit_separator()
        return "break"

    def undo_message_edit(self, event: tk.Event) -> str | None:
        if not self.message_editor_matches_event(event):
            return None

        try:
            self.message_text.edit_undo()
        except tk.TclError:
            pass
        return "break"

    def redo_message_edit(self, event: tk.Event) -> str | None:
        if not self.message_editor_matches_event(event):
            return None

        try:
            self.message_text.edit_redo()
        except tk.TclError:
            pass
        return "break"

    def get_clipboard_text(self) -> str:
        try:
            return self.root.clipboard_get(type="UTF8_STRING")
        except (tk.TclError, TypeError):
            pass

        try:
            return self.root.clipboard_get()
        except tk.TclError:
            return self.root.selection_get(selection="CLIPBOARD")

    def copy_last_received(self) -> None:
        if not self.last_received_raw:
            messagebox.showerror("Copy Error", "No received data to copy.")
            return

        self.copy_text(self.last_received_raw)

    def copy_last_received_after_decode(self) -> None:
        if not self.last_received_raw:
            messagebox.showerror("Decode Error", "No received data to decode.")
            return

        try:
            decoded = deobfuscate_text(self.last_received_raw)
        except ValueError as exc:
            messagebox.showerror("Decode Error", str(exc))
            return

        self.copy_text(decoded)

    def copy_text(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

    def append_log(self, message: str) -> None:
        line = f"{message}\n\n"

        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def on_close(self) -> None:
        self.shutdown_event.set()
        self.input_outbound_queue.put(("__shutdown__", "", ()))
        self.disconnect()
        self.root.destroy()


def main() -> None:
    if TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    SerialChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
