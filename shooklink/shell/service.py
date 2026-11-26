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
from shooklink.shell.codec import MAX_SHELL_IO_BYTES, decode_output, output_message
from shooklink.shell.key_stream import KeyReader, KeyWriter
from shooklink.shell.process import TerminalProcess
from shooklink.transport.multiplexer import Priority

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHELL_TYPES = frozenset(
    {
        MessageType.SHELL_OPEN,
        MessageType.SHELL_ACCEPT,
        MessageType.SHELL_DENY,
        MessageType.SHELL_INPUT,
        MessageType.SHELL_KEYS,
        MessageType.SHELL_KEYS_ACK,
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
        term: str,
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
    pending_size: tuple[int, int] | None = None
    confirm_on_output: bool = True
    writer: KeyWriter | None = None
    reader: KeyReader | None = None
    link_id: int | None = None


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
        self._link_epoch = 0
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

    def _clear_session_locked(self) -> None:
        if self._session is not None and self._session.writer is not None:
            self._session.writer.close()
        self._session = None

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
                self._clear_session_locked()
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
        reliable = getattr(self._bus, "reliable_shell_input", False)
        link_id = getattr(self._bus, "chat_connection_id", None)
        baud = getattr(self._bus, "shell_baud", 115200)
        with self._lock:
            if self._session is not None:
                raise ShellUnavailable("a shell session is already active")
            self._session = _ShellSession(session_id, "outgoing", "requesting")
            if reliable:
                self._session.link_id = link_id
                self._session.writer = KeyWriter(
                    lambda offset, data: self._send_key_block(session_id, link_id, offset, data),
                    lambda: self._input_failed(session_id), baud,
                )
        try:
            self._send(
                Message(
                    MessageType.SHELL_OPEN,
                    {
                        "session_id": session_id,
                        "columns": columns,
                        "rows": rows,
                        "term": term,
                        "utf8": True,
                    },
                ),
                Priority.INTERACTIVE,
            )
        except BaseException:
            with self._lock:
                if self._session is not None and self._session.session_id == session_id:
                    self._clear_session_locked()
            raise
        self._notify_state(ShellState(session_id, "outgoing", "requesting"))
        return session_id

    def handle_message(self, message: Message) -> bool:
        if message.message_type not in _SHELL_TYPES:
            return False
        with self._lock:
            link_epoch = self._link_epoch
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
            MessageType.SHELL_OPEN: lambda request: self._handle_open(request, link_epoch),
            MessageType.SHELL_ACCEPT: self._handle_accept,
            MessageType.SHELL_DENY: self._handle_deny,
            MessageType.SHELL_INPUT: self._handle_input,
            MessageType.SHELL_KEYS: self._handle_keys,
            MessageType.SHELL_KEYS_ACK: self._handle_keys_ack,
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
        session = self._require_outgoing_session(session_id)
        if session.writer is not None:
            try:
                session.writer.write(data)
            except (BufferError, RuntimeError):
                self._input_failed(session_id)
            return
        for offset in range(0, len(data), MAX_SHELL_IO_BYTES):
            self._send(
                Message(
                    MessageType.SHELL_INPUT,
                    {"session_id": session_id},
                    data[offset : offset + MAX_SHELL_IO_BYTES],
                ),
                Priority.INTERACTIVE,
            )

    def resize(self, session_id: str, columns: int, rows: int) -> None:
        _validate_terminal_size(columns, rows)
        _validate_session_id(session_id)
        with self._lock:
            if (
                self._session is None
                or self._session.session_id != session_id
                or self._session.direction != "outgoing"
            ):
                raise ShellUnavailable("remote shell is not active")
            if self._session.state == "requesting":
                self._session.pending_size = (columns, rows)
                return
            if self._session.state != "active":
                raise ShellUnavailable("remote shell is not active")
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
            self._clear_session_locked()
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

    def cancel_pending_open(self, session_id: str) -> bool:
        with self._lock:
            if (self._session is None or self._session.session_id != session_id
                    or self._session.direction != "outgoing" or self._session.state != "requesting"):
                return False
            self._clear_session_locked()
        try:
            self._send(Message(MessageType.SHELL_EXIT, {"session_id": session_id, "reason": "timeout"}),
                       Priority.INTERACTIVE)
        except (RuntimeError, OSError):
            pass  # A disconnected link already terminates the peer's shell.
        self._notify_state(ShellState(session_id, "outgoing", "denied", reason="timeout"))
        return True

    def close(self, *, revoke_permission: bool = True) -> None:
        process = None
        with self._lock:
            self._link_epoch += 1
            if self._session is not None:
                process = self._session.process
            self._clear_session_locked()
            if revoke_permission:
                self._allow_remote_shell = False
        if process is not None:
            process.terminate()

    def _handle_open(self, message: Message, link_epoch: int) -> None:
        session_id = _session_id(message.metadata)
        columns, rows = _terminal_size(message.metadata)
        if set(message.metadata) != {
            "session_id",
            "columns",
            "rows",
            "term",
            "utf8",
        }:
            raise ShellProtocolError("shell open fields are invalid")
        term = message.metadata.get("term")
        _validate_term(term)
        if message.metadata.get("utf8") is not True:
            raise ShellProtocolError("UTF-8 terminal support is required")
        if message.body:
            raise ShellProtocolError("shell open cannot contain a body")
        with self._lock:
            if link_epoch != self._link_epoch:
                return
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
            term,
        )
        denied_reason = None
        reliable = getattr(self._bus, "reliable_shell_input", False)
        link_id = getattr(self._bus, "chat_connection_id", None)
        with self._lock:
            if link_epoch != self._link_epoch:
                denied_reason = "disconnected"
            elif not self._allow_remote_shell:
                denied_reason = "permission"
            elif self._session is not None:
                denied_reason = "busy"
            else:
                self._session = _ShellSession(
                    session_id,
                    "incoming",
                    "starting",
                    process,
                    reader=KeyReader() if reliable else None,
                    link_id=link_id,
                )
        if denied_reason is not None:
            process.terminate()
            if denied_reason == "disconnected":
                return  # No response belongs on a replacement link.
            self._send(
                Message(
                    MessageType.SHELL_DENY,
                    {"session_id": session_id, "reason": denied_reason},
                ),
                Priority.INTERACTIVE,
            )
            return
        try:
            self._send(
                Message(MessageType.SHELL_ACCEPT, {"session_id": session_id}),
                Priority.INTERACTIVE,
            )
        except BaseException:
            with self._lock:
                if self._session is not None and self._session.session_id == session_id:
                    self._session = None
            process.terminate()
            raise
        try:
            with self._lock:
                can_start = (
                    self._allow_remote_shell
                    and self._session is not None
                    and self._session.session_id == session_id
                    and self._session.process is process
                )
            if not can_start:
                process.terminate()
                return
            process.start(columns, rows)
        except BaseException:
            with self._lock:
                active = (
                    self._session is not None
                    and self._session.session_id == session_id
                )
                if active:
                    self._session = None
            process.terminate()
            if active:
                self._send(
                    Message(
                        MessageType.SHELL_EXIT,
                        {"session_id": session_id, "reason": "start_failed"},
                    ),
                    Priority.INTERACTIVE,
                )
            return
        with self._lock:
            active = (
                self._allow_remote_shell
                and self._session is not None
                and self._session.session_id == session_id
                and self._session.direction == "incoming"
                and self._session.state == "starting"
                and self._session.process is process
            )
            if active:
                self._session.state = "active"
                self._notify_state(ShellState(session_id, "incoming", "active"))
        if not active:
            process.terminate()

    def _handle_accept(self, message: Message) -> None:
        session_id = _session_id(message.metadata)
        if message.body:
            raise ShellProtocolError("shell accept cannot contain a body")
        self._confirm_outgoing(session_id)

    def _confirm_outgoing(self, session_id: str) -> None:
        pending_size = None
        with self._lock:
            if (
                self._session is None
                or self._session.session_id != session_id
                or self._session.direction != "outgoing"
            ):
                raise ShellProtocolError("stale shell accept")
            if self._session.state == "active":
                return
            self._session.state = "active"
            pending_size = self._session.pending_size
            self._session.pending_size = None
        self._notify_state(ShellState(session_id, "outgoing", "active"))
        if pending_size is not None:
            self.resize(session_id, *pending_size)

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
                self._clear_session_locked()
                denied = True
        if denied:
            self._notify_state(
                ShellState(session_id, "outgoing", "denied", reason=str(reason))
            )

    def _handle_input(self, message: Message) -> None:
        session_id = _session_id(message.metadata)
        if len(message.body) > MAX_SHELL_IO_BYTES:
            raise ShellProtocolError("shell input exceeds the frame budget")
        with self._lock:
            process = self._require_incoming_process(session_id)
            if self._session.reader is not None:
                raise ShellProtocolError("unsequenced keyboard input in reliable session")
        process.write(message.body)

    def _key_send(self, message, link_id):
        options = {} if link_id is None else {"expected_connection_id": link_id}
        self._bus.send(message, secure=True, priority=Priority.INTERACTIVE, **options)

    def _send_key_block(self, session_id, link_id, offset, data):
        self._require_outgoing_session(session_id)
        self._key_send(Message(MessageType.SHELL_KEYS, {"session_id": session_id, "offset": offset}, data), link_id)

    def _handle_keys(self, message: Message):
        sid = _session_id(message.metadata)
        if set(message.metadata) != {"session_id", "offset"}:
            raise ShellProtocolError("invalid shell key fields")
        with self._lock:
            process = self._require_incoming_process(sid)
            session = self._session
            if not self._allow_remote_shell or session.reader is None:
                raise ShellProtocolError("shell input not negotiated")
        try:
            end = session.reader.receive(message.metadata["offset"], message.body, process.write)
        except OSError:
            self._input_failed(sid)
            return
        self._key_send(Message(MessageType.SHELL_KEYS_ACK, {"session_id": sid, "end": end}), session.link_id)

    def _handle_keys_ack(self, message: Message):
        sid = _session_id(message.metadata)
        if message.body or set(message.metadata) != {"session_id", "end"}:
            raise ShellProtocolError("invalid shell input acknowledgement")
        with self._lock:
            session = self._session
            if (session is None or session.session_id != sid or session.writer is None
                    or session.direction != "outgoing" or session.state != "active"):
                raise ShellProtocolError("stale shell input acknowledgement")
            session.writer.acknowledge(message.metadata["end"])

    def _input_failed(self, sid):
        with self._lock:
            session = self._session
            if session is None or session.session_id != sid:
                return
            self._clear_session_locked()
        try:
            if session.process is not None:
                session.process.terminate()
            self._key_send(Message(MessageType.SHELL_EXIT, {"session_id": sid, "reason": "input_unconfirmed"}),
                           session.link_id)
        except (RuntimeError, OSError):
            pass
        finally:
            self._notify_state(ShellState(sid, session.direction, "exited", reason="input_unconfirmed"))

    def _handle_output(self, message: Message) -> None:
        session_id = _session_id(message.metadata)
        data = decode_output(message, getattr(self._bus, "shell_output_compression", False))
        # An authenticated frame for our pending session also proves the peer accepted it.
        self._confirm_outgoing(session_id)
        with self._lock:
            if self._session is None or self._session.session_id != session_id:
                return
            listeners = tuple(self._output_listeners)
        output = ShellOutput(session_id, data)
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
            self._clear_session_locked()
        if process is not None:
            process.terminate()
        exit_code = message.metadata.get("exit_code")
        if type(exit_code) is not int:
            exit_code = None
        self._notify_state(
            ShellState(session_id, direction, "exited", exit_code=exit_code,
                       reason="input_unconfirmed" if message.metadata.get("reason") == "input_unconfirmed" else None)
        )

    def _process_output(self, session_id: str, data: bytes) -> None:
        if not isinstance(data, bytes) or not data:
            return
        capable = getattr(self._bus, "shell_output_compression", False)
        with self._lock:
            if (
                self._session is None
                or self._session.session_id != session_id
                or self._session.direction != "incoming"
            ):
                return
            repeat_accept = self._session.confirm_on_output and not capable
            self._session.confirm_on_output = False
        if repeat_accept:
            # Legacy controllers otherwise drop the prompt if the first accept was lost.
            self._send(Message(MessageType.SHELL_ACCEPT, {"session_id": session_id}), Priority.INTERACTIVE)
        for offset in range(0, len(data), MAX_SHELL_IO_BYTES):
            self._send(
                output_message(session_id, data[offset : offset + MAX_SHELL_IO_BYTES], capable),
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
                or self._session.state != "active"
                or self._session.process is None
            ):
                raise ShellProtocolError("stale executing shell message")
            self._session.confirm_on_output = False
            return self._session.process

    def _send(self, message: Message, priority: Priority) -> None:
        self._bus.send(message, secure=True, priority=priority)

    def _notify_state(self, state: ShellState) -> None:
        with self._lock:
            listeners = tuple(self._state_listeners)
        for listener in listeners:
            listener(state)


def _default_process_factory(on_output, on_exit, term: str) -> TerminalProcess:
    if sys.platform == "win32":
        from shooklink.shell.windows_conpty import WindowsConPtyProcess

        return WindowsConPtyProcess(on_output, on_exit, term=term)
    from shooklink.shell.unix_pty import UnixPtyProcess

    return UnixPtyProcess(on_output, on_exit, term=term)


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
    "MAX_SHELL_IO_BYTES",
    "ShellProtocolError",
    "ShellService",
    "ShellState",
    "ShellUnavailable",
]
