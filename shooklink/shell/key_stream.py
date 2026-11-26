"""Volatile, offset-acknowledged keyboard input for one authenticated PTY."""

import threading
import time
from collections.abc import Callable

FEATURE = "shell_keys_v1"
MAX_BLOCK = 4096
MAX_PENDING = 1 << 20
MAX_OFFSET = (1 << 53) - 1
MIN_STALL_SECONDS = 8.0


class KeyReader:
    def __init__(self):
        self.next_offset = 0
        self._failed = False
        self._lock = threading.Lock()

    def receive(self, offset: int, data: bytes, write: Callable[[bytes], None]) -> int:
        if (type(offset) is not int or not 0 <= offset <= MAX_OFFSET - len(data)
                or not 0 < len(data) <= MAX_BLOCK):
            raise ValueError("invalid shell input range")
        with self._lock:
            if self._failed:
                raise OSError("previous PTY write was not confirmed")
            if offset <= self.next_offset < offset + len(data):
                try:
                    write(data[self.next_offset - offset:])
                except Exception as exc:
                    # A PTY may have consumed part of a failed write. Never replay it.
                    self._failed = True
                    raise OSError("uncertain PTY write") from exc
                self.next_offset = offset + len(data)
            return self.next_offset


class KeyWriter:
    def __init__(self, send, failed, baud=115200):
        self._send, self._failed = send, failed
        self._baud = max(300, baud)
        self._block = min(MAX_BLOCK, max(1, self._baud // 200))
        self._cv = threading.Condition()
        self._pending = bytearray()
        self._base = self._sent = 0
        self._last = self._progress = 0.0
        self._retries = 0
        self._closed = False
        self._thread = None

    def write(self, data: bytes):
        if not data:
            return
        with self._cv:
            if self._closed:
                raise RuntimeError("shell input is closed")
            if len(self._pending) + len(data) > MAX_PENDING or self._base + len(self._pending) + len(data) > MAX_OFFSET:
                raise BufferError("shell input queue full")
            if not self._pending:
                self._progress = time.monotonic()
            self._pending.extend(data)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="shooklink-shell-keys", daemon=True)
                self._thread.start()
            self._cv.notify()

    def acknowledge(self, end: int):
        with self._cv:
            if type(end) is not int or not 0 <= end <= self._sent:
                raise ValueError("invalid shell input acknowledgement")
            if end > self._base:
                del self._pending[:end - self._base]
                self._base = end
                self._retries = 0
                self._progress = time.monotonic()
                self._cv.notify()

    def close(self):
        with self._cv:
            self._closed = True
            self._pending.clear()
            self._cv.notify()

    def _run(self):
        try:
            while True:
                with self._cv:
                    self._cv.wait_for(lambda: self._closed or self._pending)
                    if self._closed:
                        return
                    now = time.monotonic()
                    count = min(self._block, len(self._pending))
                    end = self._base + count
                    rto = max(.16, .08 + 10 * (256 + count) / self._baud)
                    deadline = self._progress + max(MIN_STALL_SECONDS, 5 * rto)
                    if now >= deadline:
                        raise TimeoutError("shell input not confirmed")
                    due = self._last + rto * 2 ** min(3, self._retries)
                    if self._base == self._sent:
                        due = now  # First key never waits for an ACK or a batching timer.
                    elif end > self._sent and count <= 96:
                        # Carry the outstanding prefix with the new keys. A short burst
                        # coalesces, and one lost keystroke heals on the next packet.
                        due = min(due, self._last + .004)
                    if now < due:
                        self._cv.wait(min(due, deadline) - now)
                        continue
                    offset, payload = self._base, bytes(self._pending[:count])
                    self._retries = self._retries + 1 if end == self._sent else 0
                    self._last, self._sent = now, end
                self._send(offset, payload)
        except Exception:
            self.close()
            self._failed()  # Do not log, persist, or transfer outstanding keystrokes to another session.
