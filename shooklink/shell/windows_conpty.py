"""Windows ConPTY backend with lazy pywinpty loading."""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class _WindowsJob:
    """Kill-on-close job object that owns the spawned shell process tree."""

    def __init__(self, process_id: int) -> None:
        import ctypes
        from ctypes import wintypes

        if not hasattr(ctypes, "WinDLL"):
            raise OSError("Windows job objects are unavailable")

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        self._kernel32 = kernel32
        self._handle = job
        try:
            information = ExtendedLimitInformation()
            information.BasicLimitInformation.LimitFlags = 0x00002000
            if not kernel32.SetInformationJobObject(
                job,
                9,
                ctypes.byref(information),
                ctypes.sizeof(information),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            process = kernel32.OpenProcess(0x0001 | 0x0100, False, process_id)
            if not process:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not kernel32.AssignProcessToJobObject(job, process):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel32.CloseHandle(process)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        handle = self._handle
        if handle:
            self._handle = None
            self._kernel32.CloseHandle(handle)


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
        term: str = "xterm-256color",
        process_tree_guard_factory: Any | None = None,
    ) -> None:
        self.command = command or _default_windows_shell()
        self.term = term
        self._on_output = on_output
        self._on_exit = on_exit
        self._pty_process_type = pty_process_type
        self._process_tree_guard_factory = process_tree_guard_factory
        self._lock = threading.RLock()
        self._process = None
        self._reader_thread: threading.Thread | None = None
        self._exit_notified = False
        self._tree_guard = None
        self._cleanup_done = False

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
            guard_factory = self._process_tree_guard_factory
            if guard_factory is None:
                guard_factory = _WindowsJob
            process_id = getattr(self._process, "pid", None)
            if process_id is not None:
                try:
                    self._tree_guard = guard_factory(process_id)
                except BaseException:
                    logger.exception("could not attach ConPTY process to a Windows job")
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
        if process is None:
            return
        if process.isalive():
            process.terminate(force=True)
        self._cleanup_process()

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
            self._cleanup_process()
            self._notify_exit(exit_code)

    def _cleanup_process(self) -> None:
        with self._lock:
            if self._cleanup_done:
                return
            self._cleanup_done = True
            process = self._process
            guard = self._tree_guard
        if guard is not None:
            try:
                guard.close()
            except BaseException:
                logger.exception("could not close Windows process-tree job")
        if process is not None:
            close = getattr(process, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException:
                    logger.exception("could not close ConPTY handles")

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
