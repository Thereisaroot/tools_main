"""Authorization and protocol state for interactive remote shells."""

from __future__ import annotations

import re
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from shooklink.protocol.messages import Message, MessageType
from shooklink.shell.process import TerminalProcess
from shooklink.transport.multiplexer import Priority

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHELL_TYPES = frozenset(
    {
        MessageType.SHELL_OPEN,
        MessageType.SHELL_ACCEPT,
        MessageType.SHELL_DENY,
        MessageType.SHELL_INPUT,
        MessageType.SHELL_OUTPUT,
        MessageType.SHELL_RESIZE,
        MessageType.SHELL_EXIT,
    }
)


class ShellProtocolError(RuntimeError):
    """Raised when a remote shell message violates the session contract."""


class ShellUnavailable(RuntimeError):
    """Raised when a new outgoing shell cannot be opened."""


class ShellBus(Protocol):
    trusted: bool

    def send(
        self,
        message: Message,
        *,
        secure: bool = True,
        priority: Priority = Priority.NORMAL,
    ) -> None: ...

    def decrypt_secure(self, message: Message) -> bytes: ...


class ProcessFactory(Protocol):
    def __call__(
        self,
        on_output: Callable[[bytes], None],
        on_exit: Callable[[int | None], None],
    ) -> TerminalProcess: ...


@dataclass(frozen=True, slots=True)
class ShellOutput:
    session_id: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ShellState:
    session_id: str
    direction: str
    state: str
    reason: str | None = None
    exit_code: int | None = None


@dataclass(slots=True)
class _ShellSession:
    session_id: str
    direction: str
    state: str
    process: TerminalProcess | None = None


class ShellService:
    def __init__(
        self,
        bus: ShellBus,
        process_factory: ProcessFactory | None = None,
    ) -> None:
        self._bus = bus
        self._process_factory = process_factory or _default_process_factory
        self._lock = threading.RLock()
        self._allow_remote_shell = False
        self._session: _ShellSession | None = None
        self._output_listeners: list[Callable[[ShellOutput], None]] = []
        self._state_listeners: list[Callable[[ShellState], None]] = []

    @property
    def allow_remote_shell(self) -> bool:
        with self._lock:
            return self._allow_remote_shell

    @property
    def active_session_id(self) -> str | None:
        with self._lock:
            return None if self._session is None else self._session.session_id

    def add_output_listener(self, listener: Callable[[ShellOutput], None]) -> None:
        with self._lock:
            if listener not in self._output_listeners:
                self._output_listeners.append(listener)

    def remove_output_listener(self, listener: Callable[[ShellOutput], None]) -> None:
        with self._lock:
            try:
                self._output_listeners.remove(listener)
            except ValueError:
                pass

    def add_state_listener(self, listener: Callable[[ShellState], None]) -> None:
        with self._lock:
            if listener not in self._state_listeners:
                self._state_listeners.append(listener)

    def remove_state_listener(self, listener: Callable[[ShellState], None]) -> None:
        with self._lock:
            try:
                self._state_listeners.remove(listener)
            except ValueError:
                pass

    def set_allow_remote_shell(self, allowed: bool) -> None:
        if not isinstance(allowed, bool):
            raise TypeError("remote shell permission must be a boolean")
        process = None
        session_id = None
        with self._lock:
            self._allow_remote_shell = allowed
            if (
                not allowed
                and self._session is not None
                and self._session.direction == "incoming"
            ):
                process = self._session.process
                session_id = self._session.session_id
                self._session = None
        if process is not None:
            process.terminate()
            self._send(
                Message(
                    MessageType.SHELL_EXIT,
                    {"session_id": session_id, "reason": "permission"},
                ),
                Priority.INTERACTIVE,
            )
            self._notify_state(
                ShellState(session_id, "incoming", "exited", reason="permission")
            )

    def open_remote(
        self,
        *,
        columns: int,
        rows: int,
        term: str = "xterm-256color",
    ) -> str:
        _validate_terminal_size(columns, rows)
        _validate_term(term)
        if not self._bus.trusted:
            raise ShellUnavailable("trust the connected peer before opening a shell")
        session_id = uuid.uuid4().hex
        with self._lock:
            if self._session is not None:
                raise ShellUnavailable("a shell session is already active")
            self._session = _ShellSession(session_id, "outgoing", "requesting")
        try:
            self._send(
                Message(
                    MessageType.SHELL_OPEN,
                    {
                        "session_id": session_id,
                        "columns": columns,
                        "rows": rows,
                        "term": term,
                    },
                ),
                Priority.INTERACTIVE,
            )
        except BaseException:
            with self._lock:
                if self._session is not None and self._session.session_id == session_id:
                    self._session = None
            raise
        self._notify_state(ShellState(session_id, "outgoing", "requesting"))
        return session_id

    def handle_message(self, message: Message) -> bool:
        if message.message_type not in _SHELL_TYPES:
            return False
        if not self._bus.trusted:
            return False
        try:
            body = self._bus.decrypt_secure(message)
        except Exception:
            return False
        if not isinstance(body, bytes):
            return False
        authenticated = Message(message.message_type, message.metadata, body)
        handlers = {
            MessageType.SHELL_OPEN: self._handle_open,
            MessageType.SHELL_ACCEPT: self._handle_accept,
            MessageType.SHELL_DENY: self._handle_deny,
            MessageType.SHELL_INPUT: self._handle_input,
            MessageType.SHELL_OUTPUT: self._handle_output,
            MessageType.SHELL_RESIZE: self._handle_resize,
            MessageType.SHELL_EXIT: self._handle_exit,
        }
        try:
            handlers[authenticated.message_type](authenticated)
        except (ShellProtocolError, TypeError, ValueError):
            return False
        return True

    def send_input(self, session_id: str, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("shell input must be bytes")
        self._require_outgoing_session(session_id)
        self._send(
            Message(MessageType.SHELL_INPUT, {"session_id": session_id}, data),
            Priority.INTERACTIVE,
        )

    def resize(self, session_id: str, columns: int, rows: int) -> None:
        _validate_terminal_size(columns, rows)
        self._require_outgoing_session(session_id)
        self._send(
            Message(
                MessageType.SHELL_RESIZE,
                {"session_id": session_id, "columns": columns, "rows": rows},
            ),
            Priority.INTERACTIVE,
        )

    def close_session(self, session_id: str) -> None:
        process = None
        with self._lock:
            if self._session is None or self._session.session_id != session_id:
                return
            process = self._session.process
            self._session = None
        if process is not None:
            process.terminate()
        self._send(
            Message(
                MessageType.SHELL_EXIT,
                {"session_id": session_id, "reason": "closed"},
            ),
            Priority.INTERACTIVE,
        )
        self._notify_state(ShellState(session_id, "outgoing", "exited", reason="closed"))

    def close(self) -> None:
        process = None
        with self._lock:
            if self._session is not None:
                process = self._session.process
            self._session = None
            self._allow_remote_shell = False
        if process is not None:
            process.terminate()

    def _handle_open(self, message: Message) -> None:
        session_id = _session_id(message.metadata)
        columns, rows = _terminal_size(message.metadata)
        _validate_term(message.metadata.get("term"))
        if message.body:
            raise ShellProtocolError("shell open cannot contain a body")
        with self._lock:
            if not self._allow_remote_shell:
                reason = "permission"
            elif self._session is not None:
                reason = "busy"
            else:
                reason = None
        if reason is not None:
            self._send(
                Message(
                    MessageType.SHELL_DENY,
                    {"session_id": session_id, "reason": reason},
                ),
                Priority.INTERACTIVE,
            )
            return

        process = self._process_factory(
            lambda data: self._process_output(session_id, data),
            lambda code: self._process_exit(session_id, code),
        )
        with self._lock:
            if self._session is not None:
                self._send(
                    Message(
                        MessageType.SHELL_DENY,
                        {"session_id": session_id, "reason": "busy"},
                    ),
                    Priority.INTERACTIVE,
                )
                return
            self._session = _ShellSession(
                session_id,
                "incoming",
                "active",
                process,
            )
        try:
            process.start(columns, rows)
        except BaseException:
            with self._lock:
                if self._session is not None and self._session.session_id == session_id:
                    self._session = None
            process.terminate()
            self._send(
                Message(
                    MessageType.SHELL_DENY,
                    {"session_id": session_id, "reason": "start_failed"},
                ),
                Priority.INTERACTIVE,
            )
            return
        self._send(
            Message(MessageType.SHELL_ACCEPT, {"session_id": session_id}),
            Priority.INTERACTIVE,
        )
        with self._lock:
            active = (
                self._session is not None
                and self._session.session_id == session_id
                and self._session.direction == "incoming"
            )
        if active:
            self._notify_state(ShellState(session_id, "incoming", "active"))

    def _handle_accept(self, message: Message) -> None:
        session_id = _session_id(message.metadata)
        if message.body:
            raise ShellProtocolError("shell accept cannot contain a body")
        with self._lock:
            if (
                self._session is None
                or self._session.session_id != session_id
                or self._session.direction != "outgoing"
                or self._session.state != "requesting"
            ):
                raise ShellProtocolError("stale shell accept")
            self._session.state = "active"
        self._notify_state(ShellState(session_id, "outgoing", "active"))

    def _handle_deny(self, message: Message) -> None:
        session_id = _session_id(message.metadata)
        reason = message.metadata.get("reason")
        denied = False
        with self._lock:
            if (
                self._session is not None
                and self._session.session_id == session_id
                and self._session.direction == "outgoing"
            ):
                self._session = None
                denied = True
        if denied:
            self._notify_state(
                ShellState(session_id, "outgoing", "denied", reason=str(reason))
            )

    def _handle_input(self, message: Message) -> None:
        session_id = _session_id(message.metadata)
        process = self._require_incoming_process(session_id)
        process.write(message.body)

    def _handle_output(self, message: Message) -> None:
        session_id = _session_id(message.metadata)
        self._require_outgoing_session(session_id)
        output = ShellOutput(session_id, message.body)
        with self._lock:
            listeners = tuple(self._output_listeners)
        for listener in listeners:
            listener(output)

    def _handle_resize(self, message: Message) -> None:
        session_id = _session_id(message.metadata)
        columns, rows = _terminal_size(message.metadata)
        process = self._require_incoming_process(session_id)
        process.resize(columns, rows)

    def _handle_exit(self, message: Message) -> None:
        session_id = _session_id(message.metadata)
        process = None
        direction = "outgoing"
        with self._lock:
            if self._session is None or self._session.session_id != session_id:
                return
            process = self._session.process
            direction = self._session.direction
            self._session = None
        if process is not None:
            process.terminate()
        exit_code = message.metadata.get("exit_code")
        if type(exit_code) is not int:
            exit_code = None
        self._notify_state(
            ShellState(session_id, direction, "exited", exit_code=exit_code)
        )

    def _process_output(self, session_id: str, data: bytes) -> None:
        if not isinstance(data, bytes):
            return
        with self._lock:
            if (
                self._session is None
                or self._session.session_id != session_id
                or self._session.direction != "incoming"
            ):
                return
        self._send(
            Message(MessageType.SHELL_OUTPUT, {"session_id": session_id}, data),
            Priority.NORMAL,
        )

    def _process_exit(self, session_id: str, exit_code: int | None) -> None:
        with self._lock:
            if (
                self._session is None
                or self._session.session_id != session_id
                or self._session.direction != "incoming"
            ):
                return
            self._session = None
        self._send(
            Message(
                MessageType.SHELL_EXIT,
                {"session_id": session_id, "exit_code": exit_code},
            ),
            Priority.INTERACTIVE,
        )
        self._notify_state(
            ShellState(session_id, "incoming", "exited", exit_code=exit_code)
        )

    def _require_outgoing_session(self, session_id: str) -> _ShellSession:
        _validate_session_id(session_id)
        with self._lock:
            if (
                self._session is None
                or self._session.session_id != session_id
                or self._session.direction != "outgoing"
                or self._session.state != "active"
            ):
                raise ShellUnavailable("remote shell is not active")
            return self._session

    def _require_incoming_process(self, session_id: str) -> TerminalProcess:
        _validate_session_id(session_id)
        with self._lock:
            if (
                self._session is None
                or self._session.session_id != session_id
                or self._session.direction != "incoming"
                or self._session.process is None
            ):
                raise ShellProtocolError("stale executing shell message")
            return self._session.process

    def _send(self, message: Message, priority: Priority) -> None:
        self._bus.send(message, secure=True, priority=priority)

    def _notify_state(self, state: ShellState) -> None:
        with self._lock:
            listeners = tuple(self._state_listeners)
        for listener in listeners:
            listener(state)


def _default_process_factory(on_output, on_exit) -> TerminalProcess:
    if sys.platform == "win32":
        from shooklink.shell.windows_conpty import WindowsConPtyProcess

        return WindowsConPtyProcess(on_output, on_exit)
    from shooklink.shell.unix_pty import UnixPtyProcess

    return UnixPtyProcess(on_output, on_exit)


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise ShellProtocolError("shell session_id is invalid")


def _session_id(metadata: dict[str, Any]) -> str:
    try:
        session_id = metadata["session_id"]
    except (KeyError, TypeError) as error:
        raise ShellProtocolError("shell session_id is missing") from error
    _validate_session_id(session_id)
    return session_id


def _validate_terminal_size(columns: int, rows: int) -> None:
    if type(columns) is not int or not 2 <= columns <= 500:
        raise ValueError("terminal columns must be from 2 to 500")
    if type(rows) is not int or not 2 <= rows <= 500:
        raise ValueError("terminal rows must be from 2 to 500")


def _terminal_size(metadata: dict[str, Any]) -> tuple[int, int]:
    try:
        columns = metadata["columns"]
        rows = metadata["rows"]
    except (KeyError, TypeError) as error:
        raise ShellProtocolError("terminal size is missing") from error
    _validate_terminal_size(columns, rows)
    return columns, rows


def _validate_term(term: Any) -> None:
    if not isinstance(term, str) or not term or len(term) > 64:
        raise ShellProtocolError("terminal type is invalid")


__all__ = [
    "ShellOutput",
    "ShellProtocolError",
    "ShellService",
    "ShellState",
    "ShellUnavailable",
]
