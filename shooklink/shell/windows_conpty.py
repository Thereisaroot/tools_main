"""Windows ConPTY backend with lazy pywinpty loading."""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def _load_pty_process_type():
    from winpty import PtyProcess

    return PtyProcess


def _default_windows_shell() -> str:
    return (
        shutil.which("pwsh.exe")
        or shutil.which("powershell.exe")
        or shutil.which("cmd.exe")
        or "cmd.exe"
    )


class WindowsConPtyProcess:
    def __init__(
        self,
        on_output: Callable[[bytes], None],
        on_exit: Callable[[int | None], None],
        *,
        command: str | None = None,
        pty_process_type: Any | None = None,
    ) -> None:
        self.command = command or _default_windows_shell()
        self._on_output = on_output
        self._on_exit = on_exit
        self._pty_process_type = pty_process_type
        self._lock = threading.RLock()
        self._process = None
        self._reader_thread: threading.Thread | None = None
        self._exit_notified = False

    def start(self, columns: int, rows: int) -> None:
        _validate_size(columns, rows)
        with self._lock:
            if self._process is not None:
                raise RuntimeError("terminal process can only be started once")
            process_type = self._pty_process_type or _load_pty_process_type()
            self._process = process_type.spawn(
                self.command,
                dimensions=(rows, columns),
            )
            self._reader_thread = threading.Thread(
                target=self._read_loop,
                name="shooklink-conpty-reader",
                daemon=True,
            )
            reader = self._reader_thread
        reader.start()

    def write(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("terminal input must be bytes")
        with self._lock:
            process = self._process
        if process is None:
            raise RuntimeError("terminal process is not running")
        process.write(data.decode("utf-8"))

    def resize(self, columns: int, rows: int) -> None:
        _validate_size(columns, rows)
        with self._lock:
            process = self._process
        if process is None:
            raise RuntimeError("terminal process is not running")
        process.setwinsize(rows, columns)

    def terminate(self) -> None:
        with self._lock:
            process = self._process
        if process is None or not process.isalive():
            return
        process.terminate(force=True)

    def is_running(self) -> bool:
        with self._lock:
            process = self._process
        return bool(process is not None and process.isalive())

    def _read_loop(self) -> None:
        exit_code: int | None = None
        try:
            while True:
                with self._lock:
                    process = self._process
                if process is None:
                    break
                try:
                    text = process.read(16 * 1024)
                except EOFError:
                    break
                if not text:
                    if not process.isalive():
                        break
                    continue
                try:
                    self._on_output(text.encode("utf-8"))
                except BaseException:
                    logger.exception("ConPTY output callback failed")
            status_getter = getattr(process, "get_exitstatus", None)
            if callable(status_getter):
                exit_code = status_getter()
        except BaseException:
            logger.exception("ConPTY reader failed")
        finally:
            self._notify_exit(exit_code)

    def _notify_exit(self, exit_code: int | None) -> None:
        with self._lock:
            if self._exit_notified:
                return
            self._exit_notified = True
        try:
            self._on_exit(exit_code)
        except BaseException:
            logger.exception("ConPTY exit callback failed")


def _validate_size(columns: int, rows: int) -> None:
    if type(columns) is not int or not 2 <= columns <= 500:
        raise ValueError("terminal columns must be from 2 to 500")
    if type(rows) is not int or not 2 <= rows <= 500:
        raise ValueError("terminal rows must be from 2 to 500")


__all__ = ["WindowsConPtyProcess"]
