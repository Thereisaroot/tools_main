from __future__ import annotations

import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import serial
from serial.tools import list_ports


class SerialChatApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Serial Text Chat v1")
        self.root.geometry("760x520")

        self.serial_port: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.incoming_messages: queue.Queue[str] = queue.Queue()

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="9600")
        self.name_var = tk.StringVar(value="PC")
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

        ttk.Label(controls, text="Name").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.name_var, width=16).grid(row=1, column=2, padx=(0, 8), sticky="ew")

        ttk.Button(controls, text="Refresh Ports", command=self.refresh_ports).grid(row=1, column=3, padx=(0, 8))

        self.connect_button = ttk.Button(controls, text="Connect", command=self.toggle_connection)
        self.connect_button.grid(row=1, column=4)

        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=0)
        controls.columnconfigure(2, weight=0)

        log_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        send_frame = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        send_frame.pack(fill="x")

        ttk.Label(send_frame, text="Message").pack(anchor="w")

        self.message_entry = ttk.Entry(send_frame)
        self.message_entry.pack(fill="x", side="left", expand=True, padx=(0, 8))
        self.message_entry.bind("<Return>", self.send_message)

        self.send_button = ttk.Button(send_frame, text="Send", command=self.send_message, state="disabled")
        self.send_button.pack(side="left")

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
        self.reader_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.reader_thread.start()

        self.connect_button.configure(text="Disconnect")
        self.send_button.configure(state="normal")
        self.status_var.set(f"Connected to {port_name} @ {baudrate}")
        self.append_log("System", f"Connected to {port_name} at {baudrate} bps")
        self.message_entry.focus_set()

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
            self.append_log("System", f"Disconnected from {port_name}")

        self.connect_button.configure(text="Connect")
        self.send_button.configure(state="disabled")
        self.status_var.set("Disconnected")

    def read_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.serial_port or not self.serial_port.is_open:
                break

            try:
                raw = self.serial_port.readline()
            except serial.SerialException as exc:
                self.incoming_messages.put(f"System|Read error: {exc}")
                self.stop_event.set()
                break

            if not raw:
                continue

            text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if text:
                self.incoming_messages.put(f"Peer|{text}")

        self.incoming_messages.put("System|Connection closed")

    def process_incoming(self) -> None:
        try:
            while True:
                packet = self.incoming_messages.get_nowait()
                sender, message = packet.split("|", 1)
                self.append_log(sender, message)

                if sender == "System" and message in {"Connection closed"}:
                    self.disconnect()
        except queue.Empty:
            pass

        self.root.after(100, self.process_incoming)

    def send_message(self, event: tk.Event | None = None) -> None:
        del event

        if not self.serial_port or not self.serial_port.is_open:
            messagebox.showerror("Send Error", "Connect to a serial port first.")
            return

        message = self.message_entry.get().strip()
        if not message:
            return

        display_name = self.name_var.get().strip() or "PC"
        payload = f"{display_name}: {message}\r\n"

        try:
            self.serial_port.write(payload.encode("utf-8"))
            self.serial_port.flush()
        except serial.SerialException as exc:
            messagebox.showerror("Send Error", str(exc))
            self.disconnect()
            return

        self.append_log("Me", payload.rstrip("\r\n"))
        self.message_entry.delete(0, "end")

    def append_log(self, sender: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {sender}: {message}\n"

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
