"""Shared terminal-process contract for PTY and ConPTY backends."""

from __future__ import annotations

from typing import Protocol


class TerminalProcess(Protocol):
    def start(self, columns: int, rows: int) -> None: ...

    def write(self, data: bytes) -> None: ...

    def resize(self, columns: int, rows: int) -> None: ...

    def terminate(self) -> None: ...

    def is_running(self) -> bool: ...


__all__ = ["TerminalProcess"]
