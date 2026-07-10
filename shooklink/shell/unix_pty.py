"""Unix interactive terminal process backed by a real PTY."""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import pty
import select
import signal
import struct
import sys
import termios
import threading
import time
from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)


class UnixPtyProcess:
    def __init__(
        self,
        on_output: Callable[[bytes], None],
        on_exit: Callable[[int | None], None],
        *,
        command: Sequence[str] | None = None,
    ) -> None:
        shell = os.environ.get("SHELL", "/bin/sh")
        self.command = list(command) if command is not None else [shell, "-l"]
        if not self.command:
            raise ValueError("terminal command cannot be empty")
        self._on_output = on_output
        self._on_exit = on_exit
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._pid: int | None = None
        self._master_fd: int | None = None
        self._running = False
        self._exit_notified = False
        self._reader_thread: threading.Thread | None = None
        self._waiter_thread: threading.Thread | None = None

    def start(self, columns: int, rows: int) -> None:
        _validate_size(columns, rows)
        with self._lock:
            if self._pid is not None or self._running:
                raise RuntimeError("terminal process can only be started once")

        master_fd, slave_fd = pty.openpty()
        _set_window_size(slave_fd, columns, rows)
        environment = os.environ.copy()
        environment.setdefault("TERM", "xterm-256color")
        environment.setdefault("COLORTERM", "truecolor")
        file_actions = [
            (os.POSIX_SPAWN_DUP2, slave_fd, 0),
            (os.POSIX_SPAWN_DUP2, slave_fd, 1),
            (os.POSIX_SPAWN_DUP2, slave_fd, 2),
            (os.POSIX_SPAWN_CLOSE, master_fd),
        ]
        if slave_fd > 2:
            file_actions.append((os.POSIX_SPAWN_CLOSE, slave_fd))
        try:
            pid = os.posix_spawn(
                sys.executable,
                [
                    sys.executable,
                    "-m",
                    "shooklink.shell.pty_child",
                    *self.command,
                ],
                environment,
                file_actions=file_actions,
                setsid=True,
            )
        except BaseException:
            os.close(master_fd)
            os.close(slave_fd)
            raise

        os.close(slave_fd)
        os.set_blocking(master_fd, False)
        with self._lock:
            self._pid = pid
            self._master_fd = master_fd
            self._running = True
            self._reader_thread = threading.Thread(
                target=self._read_loop,
                name="shooklink-pty-reader",
                daemon=True,
            )
            self._waiter_thread = threading.Thread(
                target=self._wait_loop,
                name="shooklink-pty-waiter",
                daemon=True,
            )
            reader = self._reader_thread
            waiter = self._waiter_thread
        reader.start()
        waiter.start()

    def write(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("terminal input must be bytes")
        with self._write_lock:
            offset = 0
            while offset < len(data):
                with self._lock:
                    if not self._running or self._master_fd is None:
                        raise RuntimeError("terminal process is not running")
                    master_fd = self._master_fd
                try:
                    written = os.write(master_fd, data[offset:])
                except BlockingIOError:
                    select.select([], [master_fd], [], 0.1)
                    continue
                if written <= 0:
                    raise OSError("PTY input made no write progress")
                offset += written

    def resize(self, columns: int, rows: int) -> None:
        _validate_size(columns, rows)
        with self._lock:
            if not self._running or self._master_fd is None:
                raise RuntimeError("terminal process is not running")
            master_fd = self._master_fd
            pid = self._pid
        _set_window_size(master_fd, columns, rows)
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGWINCH)
            except ProcessLookupError:
                pass

    def terminate(self) -> None:
        with self._lock:
            if not self._running or self._pid is None:
                return
            pid = self._pid
        self._signal_process_group(pid, signal.SIGTERM)
        threading.Thread(
            target=self._force_kill_after_grace,
            args=(pid,),
            name="shooklink-pty-killer",
            daemon=True,
        ).start()

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def _read_loop(self) -> None:
        while True:
            with self._lock:
                master_fd = self._master_fd
            if master_fd is None:
                return
            try:
                readable, _writable, _exceptional = select.select(
                    [master_fd], [], [], 0.1
                )
                if not readable:
                    continue
                data = os.read(master_fd, 16 * 1024)
            except OSError as error:
                if error.errno in (errno.EBADF, errno.EIO):
                    return
                logger.exception("PTY read failed")
                return
            if not data:
                return
            try:
                self._on_output(data)
            except BaseException:
                logger.exception("PTY output callback failed")

    def _wait_loop(self) -> None:
        with self._lock:
            pid = self._pid
            reader = self._reader_thread
        if pid is None:
            return
        exit_code: int | None = None
        try:
            _waited_pid, status = os.waitpid(pid, 0)
            exit_code = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            pass
        finally:
            with self._lock:
                self._running = False
                self._pid = None
            if reader is not None and reader is not threading.current_thread():
                reader.join(1.0)
            self._close_master()
            self._notify_exit(exit_code)

    def _force_kill_after_grace(self, pid: int) -> None:
        time.sleep(1.0)
        with self._lock:
            still_running = self._running and self._pid == pid
        if still_running:
            self._signal_process_group(pid, signal.SIGKILL)

    @staticmethod
    def _signal_process_group(pid: int, signal_number: int) -> None:
        try:
            os.killpg(pid, signal_number)
        except ProcessLookupError:
            return
        except OSError:
            try:
                os.kill(pid, signal_number)
            except ProcessLookupError:
                pass

    def _close_master(self) -> None:
        with self._lock:
            master_fd = self._master_fd
            self._master_fd = None
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass

    def _notify_exit(self, exit_code: int | None) -> None:
        with self._lock:
            if self._exit_notified:
                return
            self._exit_notified = True
        try:
            self._on_exit(exit_code)
        except BaseException:
            logger.exception("PTY exit callback failed")


def _validate_size(columns: int, rows: int) -> None:
    if type(columns) is not int or not 2 <= columns <= 500:
        raise ValueError("terminal columns must be from 2 to 500")
    if type(rows) is not int or not 2 <= rows <= 500:
        raise ValueError("terminal rows must be from 2 to 500")


def _set_window_size(descriptor: int, columns: int, rows: int) -> None:
    packed = struct.pack("HHHH", rows, columns, 0, 0)
    fcntl.ioctl(descriptor, termios.TIOCSWINSZ, packed)


__all__ = ["UnixPtyProcess"]
