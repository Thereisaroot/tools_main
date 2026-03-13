from __future__ import annotations

import base64
import binascii
import queue
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

MESSAGE_TERMINATOR = b"\0"
CONTROL_PREFIX = "\x1eSTCFILE"
CONTROL_SEPARATOR = "\x1f"
OBFUSCATION_MARKERS = "ABC"
FILE_CHUNK_SIZE = 1024
HIGH_SPEED_FRAME_DELAY_SECONDS = 0.002
HIGH_SPEED_BAUD_THRESHOLD = 460800
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
    return CONTROL_PREFIX + CONTROL_SEPARATOR + CONTROL_SEPARATOR.join((command, *parts))


def parse_control_message(message: str) -> tuple[str, list[str]] | None:
    prefix = CONTROL_PREFIX + CONTROL_SEPARATOR
    if not message.startswith(prefix):
        return None

    parts = message[len(prefix) :].split(CONTROL_SEPARATOR)
    if not parts or not parts[0]:
        raise ValueError("Control message is malformed.")

    return parts[0], parts[1:]


def normalize_dropped_path(raw_path: str) -> Path:
    if raw_path.startswith("file://"):
        parsed = urlparse(raw_path)
        path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            path = f"//{parsed.netloc}{path}"
        return Path(path)

    return Path(raw_path)


@dataclass
class ReceiveFileState:
    transfer_id: str
    original_name: str
    path: Path
    expected_size: int
    file_handle: BinaryIO
    received_size: int = 0


class SerialChatApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Serial Text Chat v1")
        self.root.geometry("840x680")

        self.serial_port: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.file_send_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.ui_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.receive_buffer = bytearray()
        self.receive_transfers: dict[str, ReceiveFileState] = {}
        self.last_received_raw = ""
        self.connected_port_name = ""
        self.connected_baudrate = 0
        self.file_send_active = False
        self.write_lock = threading.Lock()

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="115200")
        self.status_var = tk.StringVar(value="Disconnected")

        self._build_ui()
        self.refresh_ports()
        self.update_controls()
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

    def refresh_ports(self) -> None:
        ports = [port.device for port in list_ports.comports()]
        self.port_combo["values"] = ports

        if ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])

    def toggle_connection(self) -> None:
        if self.serial_port and self.serial_port.is_open:
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
        self.reader_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.reader_thread.start()
        self.refresh_connection_status()
        self.update_controls()
        self.message_text.focus_set()

    def disconnect(self) -> None:
        self.stop_event.set()

        if self.serial_port:
            try:
                if self.serial_port.is_open:
                    self.serial_port.close()
            except serial.SerialException:
                pass

            self.serial_port = None

        self.connected_port_name = ""
        self.connected_baudrate = 0
        self.file_send_active = False
        self.receive_buffer.clear()
        self.refresh_connection_status()
        self.update_controls()

    def refresh_connection_status(self) -> None:
        if self.serial_port and self.serial_port.is_open and self.connected_port_name:
            self.status_var.set(f"Connected to {self.connected_port_name} @ {self.connected_baudrate}")
        else:
            self.status_var.set("Disconnected")

    def update_controls(self) -> None:
        connected = bool(self.serial_port and self.serial_port.is_open)
        self.connect_button.configure(text="Disconnect" if connected else "Connect")
        self.send_plain_button.configure(state="normal" if connected else "disabled")
        self.send_encoded_button.configure(state="normal" if connected else "disabled")
        self.select_file_button.configure(state="normal" if connected and not self.file_send_active else "disabled")
        self.update_drop_zone_state()

    def update_drop_zone_state(self) -> None:
        connected = bool(self.serial_port and self.serial_port.is_open)

        if DND_FILES is None:
            self.drop_zone.configure(text="Drag and drop unavailable. Install tkinterdnd2 or use Select Files.")
            return

        if not connected:
            self.drop_zone.configure(text="Connect first, then drop files here to send", bg="#f7f7f7")
        elif self.file_send_active:
            self.drop_zone.configure(text="Sending files...", bg="#e7f0ff")
        else:
            self.drop_zone.configure(text="Drop files here to send", bg="#f7f7f7")

    def read_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.serial_port or not self.serial_port.is_open:
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
            control_message = parse_control_message(message)
        except ValueError as exc:
            self.ui_events.put(("log", f"[receive error] {exc}"))
            return

        if control_message is None:
            self.ui_events.put(("text", message))
            return

        command, parts = control_message

        try:
            if command == "START":
                self.handle_file_start(parts)
            elif command == "CHUNK":
                self.handle_file_chunk(parts)
            elif command == "END":
                self.handle_file_end(parts)
            elif command == "ABORT":
                self.handle_file_abort(parts)
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
        self.ui_events.put(("log", f"[receiving file] {filename} ({expected_size} bytes)"))

    def handle_file_chunk(self, parts: list[str]) -> None:
        if len(parts) != 2:
            raise ValueError("FILE CHUNK message is malformed.")

        transfer_id, chunk_b64 = parts
        state = self.receive_transfers.get(transfer_id)
        if state is None:
            raise ValueError("Received a file chunk without a matching transfer.")

        try:
            chunk = base64.urlsafe_b64decode(chunk_b64.encode("ascii"))
        except binascii.Error as exc:
            raise ValueError("Received file chunk base64 is invalid.") from exc

        state.file_handle.write(chunk)
        state.received_size += len(chunk)

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
        except queue.Empty:
            pass

        self.root.after(100, self.process_ui_events)

    def send_plain_shortcut(self, event: tk.Event) -> str:
        del event
        self.send_message("plain")
        return "break"

    def send_encoded_shortcut(self, event: tk.Event) -> str:
        del event
        self.send_message("encoded")
        return "break"

    def send_message(self, mode: str) -> None:
        if mode not in {"plain", "encoded"}:
            raise ValueError(f"Unsupported message mode: {mode}")

        message = self.message_text.get("1.0", "end-1c")
        if not message.strip():
            return

        payload = message if mode == "plain" else obfuscate_text(message)

        try:
            self.send_serial_text(payload)
        except serial.SerialException as exc:
            messagebox.showerror("Send Error", str(exc))
            self.disconnect()
            return

        self.append_log(payload)
        self.message_text.delete("1.0", "end")
        self.message_text.focus_set()

    def send_serial_text(self, text: str) -> None:
        payload = text.encode("utf-8") + MESSAGE_TERMINATOR

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

    def handle_drag_enter(self, event: tk.Event) -> str:
        del event
        if self.serial_port and self.serial_port.is_open and not self.file_send_active:
            self.drop_zone.configure(bg="#dff3ff")
        return "break"

    def handle_drag_leave(self, event: tk.Event) -> str:
        del event
        self.update_drop_zone_state()
        return "break"

    def handle_file_drop(self, event: tk.Event) -> str:
        if not (self.serial_port and self.serial_port.is_open):
            self.update_drop_zone_state()
            return "break"

        raw_paths = self.root.tk.splitlist(event.data)
        paths = [normalize_dropped_path(raw_path) for raw_path in raw_paths]
        self.update_drop_zone_state()
        self.start_file_send(paths)
        return "break"

    def start_file_send(self, paths: list[Path]) -> None:
        if not (self.serial_port and self.serial_port.is_open):
            messagebox.showerror("File Send Error", "Connect to a serial port first.")
            return

        if self.file_send_active:
            messagebox.showerror("File Send Error", "A file transfer is already in progress.")
            return

        files = [path for path in paths if path.is_file()]
        if not files:
            messagebox.showerror("File Send Error", "No regular files were selected.")
            return

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

            with path.open("rb") as file_handle:
                while True:
                    if self.stop_event.is_set():
                        raise RuntimeError("Connection closed.")

                    chunk = file_handle.read(FILE_CHUNK_SIZE)
                    if not chunk:
                        break

                    chunk_b64 = base64.urlsafe_b64encode(chunk).decode("ascii")
                    self.send_control_message("CHUNK", transfer_id, chunk_b64)

            self.send_control_message("END", transfer_id)
            self.ui_events.put(("log", f"[sent file] {path.name}"))
        except Exception as exc:
            try:
                self.send_control_message("ABORT", transfer_id, encode_control_text(str(exc)))
            except Exception:
                pass
            raise

    def send_control_message(self, command: str, *parts: str) -> None:
        self.send_serial_text(build_control_message(command, *parts))

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
            for sequence, handler in command_shortcuts.items():
                self.message_text.bind(sequence, handler)
                self.root.bind_all(sequence, handler, add="+")

    def message_editor_has_focus(self) -> bool:
        return self.root.focus_get() is self.message_text

    def select_all_message(self, event: tk.Event) -> str | None:
        del event

        if not self.message_editor_has_focus():
            return None

        self.message_text.tag_add("sel", "1.0", "end-1c")
        self.message_text.mark_set("insert", "1.0")
        self.message_text.see("insert")
        return "break"

    def copy_message_selection(self, event: tk.Event) -> str | None:
        del event

        if not self.message_editor_has_focus():
            return None

        try:
            text = self.message_text.get("sel.first", "sel.last")
        except tk.TclError:
            return "break"

        self.copy_text(text)
        return "break"

    def cut_message_selection(self, event: tk.Event) -> str | None:
        del event

        if not self.message_editor_has_focus():
            return None

        try:
            text = self.message_text.get("sel.first", "sel.last")
            self.message_text.delete("sel.first", "sel.last")
        except tk.TclError:
            return "break"

        self.copy_text(text)
        return "break"

    def paste_into_message(self, event: tk.Event) -> str | None:
        del event

        if not self.message_editor_has_focus():
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
        del event

        if not self.message_editor_has_focus():
            return None

        try:
            self.message_text.edit_undo()
        except tk.TclError:
            pass
        return "break"

    def redo_message_edit(self, event: tk.Event) -> str | None:
        del event

        if not self.message_editor_has_focus():
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
        self.disconnect()
        self.root.destroy()


def main() -> None:
    if TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = SerialChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
