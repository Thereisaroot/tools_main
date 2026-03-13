from __future__ import annotations

import base64
import binascii
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import serial
from serial.tools import list_ports

MESSAGE_TERMINATOR = b"\0"
OBFUSCATION_MARKERS = "ABC"


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

class SerialChatApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Serial Text Chat v1")
        self.root.geometry("760x520")

        self.serial_port: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.incoming_messages: queue.Queue[tuple[str, str]] = queue.Queue()
        self.receive_buffer = bytearray()
        self.last_received_raw = ""

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="9600")
        self.status_var = tk.StringVar(value="Disconnected")

        self._build_ui()
        self.refresh_ports()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.process_incoming)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(fill="x")

        ttk.Label(controls, text="Port").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(controls, textvariable=self.port_var, width=16, state="normal")
        self.port_combo.grid(row=1, column=0, padx=(0, 8), sticky="ew")

        ttk.Label(controls, text="Baud").grid(row=0, column=1, sticky="w")
        baud_combo = ttk.Combobox(
            controls,
            textvariable=self.baud_var,
            values=["9600", "19200", "38400", "57600", "115200"],
            width=12,
            state="readonly",
        )
        baud_combo.grid(row=1, column=1, padx=(0, 8), sticky="ew")

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
        self.reader_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.reader_thread.start()

        self.connect_button.configure(text="Disconnect")
        self.send_plain_button.configure(state="normal")
        self.send_encoded_button.configure(state="normal")
        self.status_var.set(f"Connected to {port_name} @ {baudrate}")
        self.message_text.focus_set()

    def disconnect(self) -> None:
        self.stop_event.set()

        if self.serial_port:
            try:
                port_name = self.serial_port.port
            except serial.SerialException:
                port_name = "serial port"

            try:
                if self.serial_port.is_open:
                    self.serial_port.close()
            except serial.SerialException:
                pass

            self.serial_port = None
            self.receive_buffer.clear()

        self.connect_button.configure(text="Connect")
        self.send_plain_button.configure(state="disabled")
        self.send_encoded_button.configure(state="disabled")
        self.status_var.set("Disconnected")

    def read_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.serial_port or not self.serial_port.is_open:
                break

            try:
                chunk = self.serial_port.read(self.serial_port.in_waiting or 1)
            except serial.SerialException as exc:
                self.incoming_messages.put(("System", f"Read error: {exc}"))
                self.stop_event.set()
                break

            if not chunk:
                continue

            self.receive_buffer.extend(chunk)

            try:
                while True:
                    end_index = self.receive_buffer.index(MESSAGE_TERMINATOR[0])
                    message_bytes = bytes(self.receive_buffer[:end_index])
                    del self.receive_buffer[: end_index + 1]
                    self.incoming_messages.put(("Data", message_bytes.decode("utf-8", errors="replace")))
            except ValueError:
                pass

        self.incoming_messages.put(("System", "Connection closed"))

    def process_incoming(self) -> None:
        try:
            while True:
                kind, message = self.incoming_messages.get_nowait()

                if kind == "Data":
                    self.last_received_raw = message
                    self.copy_button.configure(state="normal")
                    self.copy_decoded_button.configure(state="normal")
                    self.append_log(message)
                    continue

                if kind == "System" and message in {"Connection closed"}:
                    self.disconnect()
                elif kind == "System":
                    self.status_var.set(message)
        except queue.Empty:
            pass

        self.root.after(100, self.process_incoming)

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

        if not self.serial_port or not self.serial_port.is_open:
            messagebox.showerror("Send Error", "Connect to a serial port first.")
            return

        message = self.message_text.get("1.0", "end-1c")
        if not message.strip():
            return

        payload = message if mode == "plain" else obfuscate_text(message)

        try:
            self.serial_port.write(payload.encode("utf-8") + MESSAGE_TERMINATOR)
            self.serial_port.flush()
        except serial.SerialException as exc:
            messagebox.showerror("Send Error", str(exc))
            self.disconnect()
            return

        self.append_log(payload)
        self.message_text.delete("1.0", "end")
        self.message_text.focus_set()

    def bind_editor_shortcuts(self) -> None:
        shortcuts = {
            "<Command-a>": self.select_all_message,
            "<Command-A>": self.select_all_message,
            "<Control-a>": self.select_all_message,
            "<Control-A>": self.select_all_message,
            "<Command-c>": self.copy_message_selection,
            "<Command-C>": self.copy_message_selection,
            "<Control-c>": self.copy_message_selection,
            "<Control-C>": self.copy_message_selection,
            "<Command-x>": self.cut_message_selection,
            "<Command-X>": self.cut_message_selection,
            "<Control-x>": self.cut_message_selection,
            "<Control-X>": self.cut_message_selection,
            "<Command-v>": self.paste_into_message,
            "<Command-V>": self.paste_into_message,
            "<Control-v>": self.paste_into_message,
            "<Control-V>": self.paste_into_message,
            "<Shift-Insert>": self.paste_into_message,
            "<Command-z>": self.undo_message_edit,
            "<Command-Z>": self.redo_message_edit,
            "<Control-z>": self.undo_message_edit,
            "<Control-Z>": self.redo_message_edit,
            "<Command-y>": self.redo_message_edit,
            "<Command-Y>": self.redo_message_edit,
            "<Control-y>": self.redo_message_edit,
            "<Control-Y>": self.redo_message_edit,
        }

        for sequence, handler in shortcuts.items():
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
    root = tk.Tk()
    app = SerialChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
