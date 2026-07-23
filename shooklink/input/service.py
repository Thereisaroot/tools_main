"""Authenticated bidirectional input-sharing session state."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from shooklink.input.backend import BaseInputBackend, PermissionStatus
from shooklink.input.events import (
    InputEvent,
    KeyAction,
    KeyEvent,
    KeyLocation,
    Modifiers,
    MouseButton,
    MouseButtonEvent,
    PointerMotionEvent,
    PointerPositionEvent,
    WheelEvent,
)
from shooklink.input.pointer import LogicalPointer, TransitionKind
from shooklink.input.topology import Monitor, Rect, Side, Topology
from shooklink.protocol.messages import Message, MessageType
from shooklink.transport.multiplexer import Priority

EDGE_HOLD_SECONDS = 0.5
MOTION_INTERVAL_SECONDS = 1 / 120
MAX_MONITORS = 32
MAX_COORDINATE = 10_000_000
MAX_PEER_ID_BYTES = 256
MAX_MOTION_SEQUENCE = (1 << 32) - 1
_MODIFIER_MASK = int(
    Modifiers.SHIFT
    | Modifiers.CONTROL
    | Modifiers.ALT
    | Modifiers.META
    | Modifiers.CAPS_LOCK
    | Modifiers.NUM_LOCK
)
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_INPUT_TYPES = frozenset(
    {
        MessageType.INPUT_REQUEST,
        MessageType.INPUT_ACCEPT,
        MessageType.INPUT_BUSY,
        MessageType.INPUT_ENTER,
        MessageType.INPUT_LEAVE,
        MessageType.INPUT_KEY,
        MessageType.INPUT_BUTTON,
        MessageType.INPUT_MOVE,
        MessageType.INPUT_WHEEL,
        MessageType.INPUT_POINTER_STATE,
        MessageType.INPUT_STOP,
        MessageType.INPUT_RELEASE_ALL,
    }
)


class InputProtocolError(RuntimeError):
    """Raised when an input-control message violates the wire contract."""


class InputUnavailable(RuntimeError):
    """Raised when local input sharing cannot start."""


class InputSessionState(str, Enum):
    IDLE = "idle"
    REQUESTING = "requesting_remote_control"
    CONTROLLING = "controlling_remote"
    BEING_CONTROLLED = "being_controlled"


@dataclass(frozen=True, slots=True)
class InputStateChange:
    state: InputSessionState
    session_id: str | None = None
    reason: str | None = None


class InputBus(Protocol):
    trusted: bool

    def send(
        self,
        message: Message,
        *,
        secure: bool = True,
        priority: Priority = Priority.NORMAL,
    ) -> None: ...

    def decrypt_secure(self, message: Message) -> bytes: ...


class InputService:
    def __init__(
        self,
        bus: InputBus,
        backend: BaseInputBackend,
        *,
        local_peer_id: str,
        peer_id: str,
        peer_side: Side = Side.RIGHT,
        auto_edge_enabled: bool = False,
        connected: bool = True,
        clock: Callable[[], float] = time.monotonic,
        session_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        _validate_peer_id(local_peer_id)
        _validate_peer_id(peer_id)
        if local_peer_id == peer_id:
            raise ValueError("local and remote peer IDs must differ")
        if not isinstance(peer_side, Side):
            raise TypeError("peer_side must be a Side")
        if not isinstance(auto_edge_enabled, bool):
            raise TypeError("auto_edge_enabled must be a boolean")
        if not isinstance(connected, bool):
            raise TypeError("connected must be a boolean")
        if not callable(clock) or not callable(session_factory):
            raise TypeError("clock and session_factory must be callable")
        self._bus = bus
        self._backend = backend
        self.local_peer_id = local_peer_id
        self.peer_id = peer_id
        self._peer_side = peer_side
        self._auto_edge_enabled = auto_edge_enabled
        self._clock = clock
        self._session_factory = session_factory
        self._lock = threading.RLock()
        self._capture_transition_lock = threading.RLock()
        self._state = InputSessionState.IDLE
        self._session_id: str | None = None
        self._allow_remote_input = False
        self._connected = connected
        self._closed = False
        self._pointer: LogicalPointer | None = None
        self._remote_topology: Topology | None = None
        self._local_session_topology: Topology | None = None
        self._session_peer_side = peer_side
        self._enter_from_edge = False
        self._entry_fraction = 0.5
        self._local_return_position: tuple[int, int] | None = None
        self._pending_pointer: tuple[int, int] | None = None
        self._last_motion_sent_at = self._clock()
        self._next_motion_sequence = 1
        self._last_motion_sequence = 0
        self._last_received_motion_sequence = 0
        self._edge_timer: threading.Timer | None = None
        self._motion_timer: threading.Timer | None = None
        self._state_listeners: list[Callable[[InputStateChange], None]] = []
        self._emergency_listeners: list[Callable[[str], None]] = []
        if auto_edge_enabled:
            self._refresh_idle_capture()

    @property
    def state(self) -> InputSessionState:
        with self._lock:
            return self._state

    @property
    def active_session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    @property
    def allow_remote_input(self) -> bool:
        with self._lock:
            return self._allow_remote_input

    @property
    def peer_side(self) -> Side:
        with self._lock:
            return self._peer_side

    @property
    def auto_edge_enabled(self) -> bool:
        with self._lock:
            return self._auto_edge_enabled

    def permission_status(self) -> PermissionStatus:
        return self._backend.permission_status()

    def add_state_listener(self, listener: Callable[[InputStateChange], None]) -> None:
        with self._lock:
            if listener not in self._state_listeners:
                self._state_listeners.append(listener)

    def remove_state_listener(self, listener: Callable[[InputStateChange], None]) -> None:
        with self._lock:
            try:
                self._state_listeners.remove(listener)
            except ValueError:
                pass

    def add_emergency_listener(self, listener: Callable[[str], None]) -> None:
        if not callable(listener):
            raise TypeError("emergency listener must be callable")
        with self._lock:
            if listener not in self._emergency_listeners:
                self._emergency_listeners.append(listener)

    def remove_emergency_listener(self, listener: Callable[[str], None]) -> None:
        with self._lock:
            try:
                self._emergency_listeners.remove(listener)
            except ValueError:
                pass

    def set_allow_remote_input(self, allowed: bool) -> None:
        if not isinstance(allowed, bool):
            raise TypeError("remote input permission must be a boolean")
        with self._lock:
            self._allow_remote_input = allowed
            terminate = not allowed and self._state is InputSessionState.BEING_CONTROLLED
        if terminate:
            self._finish_session(reason="permission", send_remote=True)

    def set_peer_side(self, side: Side | str) -> None:
        try:
            normalized = side if isinstance(side, Side) else Side(side)
        except (TypeError, ValueError) as error:
            raise ValueError("peer side must be left, right, top, or bottom") from error
        with self._lock:
            if self._state is not InputSessionState.IDLE:
                raise InputUnavailable("peer side cannot change during an input session")
            self._peer_side = normalized
            self._session_peer_side = normalized
            self._reset_edge_hold_locked()

    def set_peer_id(self, peer_id: str) -> None:
        _validate_peer_id(peer_id)
        if peer_id == self.local_peer_id:
            raise ValueError("local and remote peer IDs must differ")
        with self._lock:
            if self._closed:
                raise InputUnavailable("input service is closed")
            if (
                self._connected
                or self._state is not InputSessionState.IDLE
                or self._session_id is not None
            ):
                raise InputUnavailable(
                    "peer identity can change only while disconnected and idle"
                )
            self.peer_id = peer_id

    def set_auto_edge_enabled(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("auto edge setting must be a boolean")
        with self._lock:
            previous = self._auto_edge_enabled
            self._auto_edge_enabled = enabled
            self._reset_edge_hold_locked()
        try:
            self._refresh_idle_capture()
        except BaseException:
            with self._lock:
                self._auto_edge_enabled = previous
                self._reset_edge_hold_locked()
            raise

    def request_control(self) -> str:
        return self._request_control(enter_from_edge=False)

    def _request_control(self, *, enter_from_edge: bool) -> str:
        with self._lock:
            if self._closed:
                raise InputUnavailable("input service is closed")
            if not self._connected or not self._bus.trusted:
                raise InputUnavailable("trust the connected peer before sharing input")
            if self._state is not InputSessionState.IDLE:
                raise InputUnavailable("an input session is already active")
            if enter_from_edge and not self._auto_edge_enabled:
                raise InputUnavailable("automatic edge switching is disabled")
            status = self._backend.permission_status()
            if not status.capture_allowed:
                raise InputUnavailable(status.detail)
            local_topology = self._local_topology()
            position = self._backend.cursor_position()
            side = self._peer_side
            if enter_from_edge and not local_topology.is_on_outer_edge(
                side,
                *position,
            ):
                raise InputUnavailable("pointer is no longer on the configured edge")
            fraction = _entry_fraction(local_topology, side, position)
            return_position = _return_position(local_topology, side, position)
            session_id = self._session_factory()
            _validate_session_id(session_id)
            self._state = InputSessionState.REQUESTING
            self._session_id = session_id
            self._session_peer_side = side
            self._enter_from_edge = enter_from_edge
            self._entry_fraction = fraction
            self._local_return_position = return_position
            self._local_session_topology = local_topology
            self._reset_edge_hold_locked()
        try:
            self._stop_capture_if_running()
        except BaseException:
            self._clear_session(session_id)
            raise
        try:
            self._send(
                Message(
                    MessageType.INPUT_REQUEST,
                    {
                        "session_id": session_id,
                        "controller_id": self.local_peer_id,
                        "side": side.value,
                        "monitors": _encode_topology(local_topology),
                    },
                ),
                Priority.INTERACTIVE,
            )
        except BaseException:
            self._clear_session(session_id)
            self._refresh_idle_capture_if_idle()
            raise
        with self._lock:
            active = (
                not self._closed
                and self._connected
                and self._state is InputSessionState.REQUESTING
                and self._session_id == session_id
            )
            if active:
                self._notify_state(
                    InputStateChange(InputSessionState.REQUESTING, session_id)
                )
        if not active:
            self._safe_send(
                Message(
                    MessageType.INPUT_STOP,
                    {"session_id": session_id, "reason": "cancelled"},
                ),
                Priority.INTERACTIVE,
            )
            self._refresh_idle_capture_if_idle()
            raise InputUnavailable("input request was cancelled")
        return session_id

    def stop_control(self, *, reason: str = "manual") -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("stop reason must be a non-empty string")
        with self._lock:
            if self._state is InputSessionState.IDLE:
                return
        self._finish_session(reason=reason, send_remote=True)

    def poll(self, *, now: float | None = None) -> None:
        timestamp = self._clock() if now is None else now
        with self._lock:
            if (
                self._state is not InputSessionState.CONTROLLING
                or self._pending_pointer is None
                or timestamp - self._last_motion_sent_at < MOTION_INTERVAL_SECONDS
            ):
                return
            self._flush_pointer_locked(timestamp)

    def connection_changed(self, connected: bool) -> None:
        if not isinstance(connected, bool):
            raise TypeError("connected must be a boolean")
        if not connected:
            self.disconnect()
            return
        with self._lock:
            self._connected = True
        self._refresh_idle_capture()

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False
            self._reset_edge_hold_locked()
        self._finish_session(reason="disconnected", send_remote=False)
        self._stop_capture_if_running()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connected = False
            self._auto_edge_enabled = False
            self._allow_remote_input = False
            self._reset_edge_hold_locked()
        try:
            self._finish_session(reason="closed", send_remote=False)
        except Exception:
            pass
        try:
            self._stop_capture_if_running()
        except Exception:
            pass
        try:
            self._backend.release_all()
        except Exception:
            pass
        with self._lock:
            self._cancel_motion_timer_locked()

    def handle_message(self, message: Message) -> bool:
        if message.message_type not in _INPUT_TYPES:
            return False
        with self._lock:
            if self._closed or not self._connected:
                return False
        if not self._bus.trusted:
            return False
        try:
            body = self._bus.decrypt_secure(message)
        except Exception:
            return False
        if body != b"":
            return False
        authenticated = Message(message.message_type, message.metadata, body)
        handlers = {
            MessageType.INPUT_REQUEST: self._handle_request,
            MessageType.INPUT_ACCEPT: self._handle_accept,
            MessageType.INPUT_BUSY: self._handle_busy,
            MessageType.INPUT_ENTER: self._handle_enter,
            MessageType.INPUT_LEAVE: self._handle_leave,
            MessageType.INPUT_KEY: self._handle_key,
            MessageType.INPUT_BUTTON: self._handle_button,
            MessageType.INPUT_MOVE: self._handle_move,
            MessageType.INPUT_WHEEL: self._handle_wheel,
            MessageType.INPUT_POINTER_STATE: self._handle_pointer_state,
            MessageType.INPUT_STOP: self._handle_stop,
            MessageType.INPUT_RELEASE_ALL: self._handle_release_all,
        }
        try:
            handlers[authenticated.message_type](authenticated)
        except InputProtocolError:
            return False
        except Exception:
            try:
                self._finish_session(reason="handler_failed", send_remote=True)
            except Exception:
                pass
            return False
        return True

    def _handle_request(self, message: Message) -> None:
        expected = {"session_id", "controller_id", "side", "monitors"}
        _require_fields(message.metadata, expected)
        session_id = _metadata_session_id(message.metadata)
        controller_id = message.metadata["controller_id"]
        _validate_peer_id(controller_id)
        if controller_id != self.peer_id:
            raise InputProtocolError("input controller identity does not match the peer")
        try:
            side = Side(message.metadata["side"])
        except (TypeError, ValueError) as error:
            raise InputProtocolError("input peer side is invalid") from error
        remote_topology = _decode_topology(message.metadata["monitors"])
        status = self._backend.permission_status()
        with self._lock:
            reason = self._incoming_busy_reason_locked(
                controller_id,
                session_id,
                status.inject_allowed,
            )
        if reason is not None:
            if reason == "disconnected":
                return
            self._send_busy(session_id, reason)
            return
        try:
            local_topology = self._local_topology()
            cursor_x, cursor_y = local_topology.nearest_point(
                *self._backend.cursor_position()
            )
        except (InputUnavailable, OSError, TypeError, ValueError):
            self._send_busy(session_id, "unavailable")
            return
        inject_allowed = self._backend.permission_status().inject_allowed
        with self._lock:
            reason = self._incoming_busy_reason_locked(
                controller_id,
                session_id,
                inject_allowed,
            )
            if reason is None:
                self._state = InputSessionState.BEING_CONTROLLED
                self._session_id = session_id
                self._session_peer_side = side
                self._remote_topology = remote_topology
                self._local_session_topology = local_topology
                self._pointer = None
                self._pending_pointer = None
                self._reset_edge_hold_locked()
        if reason is not None:
            if reason == "disconnected":
                return
            self._send_busy(session_id, reason)
            return
        try:
            self._stop_capture_if_running()
            self._backend.release_all()
        except Exception:
            with self._lock:
                if self._session_id == session_id:
                    self._clear_session_locked()
            self._send_busy(session_id, "unavailable")
            self._notify_state(
                InputStateChange(InputSessionState.IDLE, reason="unavailable")
            )
            return
        with self._lock:
            active = (
                not self._closed
                and self._connected
                and self._state is InputSessionState.BEING_CONTROLLED
                and self._session_id == session_id
            )
        if not active:
            return
        try:
            self._send(
                Message(
                    MessageType.INPUT_ACCEPT,
                    {
                        "session_id": session_id,
                        "peer_id": self.local_peer_id,
                        "monitors": _encode_topology(local_topology),
                        "x": cursor_x,
                        "y": cursor_y,
                    },
                ),
                Priority.INTERACTIVE,
            )
        except BaseException:
            self._finish_session(reason="accept_failed", send_remote=False)
            raise
        with self._lock:
            active = (
                not self._closed
                and self._connected
                and self._state is InputSessionState.BEING_CONTROLLED
                and self._session_id == session_id
            )
            if active:
                self._notify_state(
                    InputStateChange(InputSessionState.BEING_CONTROLLED, session_id)
                )
        if not active:
            self._safe_send(
                Message(
                    MessageType.INPUT_STOP,
                    {"session_id": session_id, "reason": "disconnected"},
                ),
                Priority.INTERACTIVE,
            )

    def _handle_accept(self, message: Message) -> None:
        _require_fields(
            message.metadata,
            {"session_id", "peer_id", "monitors", "x", "y"},
        )
        session_id = _metadata_session_id(message.metadata)
        peer_id = message.metadata["peer_id"]
        _validate_peer_id(peer_id)
        if peer_id != self.peer_id:
            raise InputProtocolError("input accept identity does not match the peer")
        topology = _decode_topology(message.metadata["monitors"])
        cursor_x = _bounded_int(message.metadata["x"], "pointer x")
        cursor_y = _bounded_int(message.metadata["y"], "pointer y")
        if not topology.contains(cursor_x, cursor_y):
            raise InputProtocolError("input accept pointer is outside peer monitors")
        with self._lock:
            if (
                self._state is not InputSessionState.REQUESTING
                or self._session_id != session_id
            ):
                raise InputProtocolError("stale input accept")
            pointer = LogicalPointer(
                topology,
                return_side=self._session_peer_side.opposite,
            )
            if self._enter_from_edge:
                transition = pointer.enter(
                    self._session_peer_side.opposite,
                    self._entry_fraction,
                )
                initial_x, initial_y = transition.position
            else:
                pointer.set_position(cursor_x, cursor_y)
                initial_x, initial_y = cursor_x, cursor_y
            self._pointer = pointer
            self._remote_topology = topology
            self._pending_pointer = None
            self._last_motion_sent_at = self._clock()
            self._next_motion_sequence = 1
            self._last_motion_sequence = 0
            self._last_received_motion_sequence = 0
            return_position = self._local_return_position
        try:
            self._start_capture(
                self._captured_event,
                self._emergency_stop,
                suppress=True,
            )
        except BaseException:
            with self._lock:
                if self._session_id == session_id:
                    self._clear_session_locked()
            self._safe_send(
                Message(
                    MessageType.INPUT_STOP,
                    {"session_id": session_id, "reason": "capture_failed"},
                ),
                Priority.INTERACTIVE,
            )
            self._restore_local_cursor(return_position)
            self._refresh_idle_capture()
            raise
        send_error = None
        failed_return_position = return_position
        with self._lock:
            cancelled = (
                self._state is not InputSessionState.REQUESTING
                or self._session_id != session_id
            )
            if not cancelled:
                self._state = InputSessionState.CONTROLLING
                try:
                    self._send(
                        Message(
                            MessageType.INPUT_ENTER,
                            {
                                "session_id": session_id,
                                "x": initial_x,
                                "y": initial_y,
                            },
                        ),
                        Priority.INTERACTIVE,
                    )
                except BaseException as error:
                    send_error = error
                    failed_return_position = self._local_return_position
                    self._clear_session_locked()
        if cancelled or send_error is not None:
            if send_error is not None:
                self._safe_send(
                    Message(
                        MessageType.INPUT_STOP,
                        {"session_id": session_id, "reason": "enter_failed"},
                    ),
                    Priority.INTERACTIVE,
                )
            try:
                self._stop_capture_if_running()
            finally:
                self._restore_local_cursor(failed_return_position)
            if send_error is not None:
                self._refresh_idle_capture()
                raise send_error
            self._refresh_idle_capture()
            return
        self._notify_state(InputStateChange(InputSessionState.CONTROLLING, session_id))

    def _handle_busy(self, message: Message) -> None:
        if not set(message.metadata).issubset({"session_id", "reason"}):
            raise InputProtocolError("input busy fields are invalid")
        session_id = _metadata_session_id(message.metadata)
        reason = message.metadata.get("reason", "busy")
        if not isinstance(reason, str):
            raise InputProtocolError("input busy reason is invalid")
        with self._lock:
            if (
                self._state is not InputSessionState.REQUESTING
                or self._session_id != session_id
            ):
                raise InputProtocolError("stale input busy")
            self._clear_session_locked()
        self._notify_state(InputStateChange(InputSessionState.IDLE, reason=reason))
        self._refresh_idle_capture()

    def _handle_enter(self, message: Message) -> None:
        session_id, x, y = self._position_message(message)
        self._require_being_controlled(session_id)
        self._inject(PointerPositionEvent(x, y), session_id)
        # Native injection can apply asynchronously; acknowledge the logical target.
        self._send_pointer_state(session_id, 0, x, y)

    def _handle_leave(self, message: Message) -> None:
        session_id, _x, _y = self._position_message(message)
        self._require_being_controlled(session_id)
        self._finish_session(reason="edge", send_remote=False)

    def _handle_key(self, message: Message) -> None:
        expected = {
            "session_id",
            "action",
            "usage",
            "scan_code",
            "virtual_key",
            "text",
            "modifiers",
            "location",
            "repeat",
            "extended",
        }
        _require_fields(message.metadata, expected)
        session_id = _metadata_session_id(message.metadata)
        self._require_being_controlled(session_id)
        modifier_bits = message.metadata["modifiers"]
        if (
            type(modifier_bits) is not int
            or modifier_bits < 0
            or modifier_bits & ~_MODIFIER_MASK
        ):
            raise InputProtocolError("input key modifiers are invalid")
        try:
            event = KeyEvent(
                KeyAction(message.metadata["action"]),
                usage=message.metadata["usage"],
                scan_code=message.metadata["scan_code"],
                virtual_key=message.metadata["virtual_key"],
                text=message.metadata["text"],
                modifiers=Modifiers(modifier_bits),
                location=KeyLocation(message.metadata["location"]),
                repeat=message.metadata["repeat"],
                extended=message.metadata["extended"],
            )
        except (TypeError, ValueError) as error:
            raise InputProtocolError("input key payload is invalid") from error
        self._inject(event, session_id)

    def _handle_button(self, message: Message) -> None:
        _require_fields(message.metadata, {"session_id", "button", "action"})
        session_id = _metadata_session_id(message.metadata)
        self._require_being_controlled(session_id)
        try:
            event = MouseButtonEvent(
                MouseButton(message.metadata["button"]),
                KeyAction(message.metadata["action"]),
            )
        except (TypeError, ValueError) as error:
            raise InputProtocolError("input button payload is invalid") from error
        self._inject(event, session_id)

    def _handle_move(self, message: Message) -> None:
        _require_fields(
            message.metadata,
            {"session_id", "motion_sequence", "x", "y"},
        )
        session_id = _metadata_session_id(message.metadata)
        motion_sequence = _motion_sequence(message.metadata["motion_sequence"])
        x = _bounded_int(message.metadata["x"], "pointer x")
        y = _bounded_int(message.metadata["y"], "pointer y")
        self._require_being_controlled(session_id)
        with self._lock:
            topology = self._local_session_topology
        if topology is None or not topology.contains(x, y):
            raise InputProtocolError("pointer position is outside connected monitors")
        with self._lock:
            if motion_sequence <= self._last_received_motion_sequence:
                return
        self._inject(PointerPositionEvent(x, y), session_id)
        with self._lock:
            if (
                self._state is InputSessionState.BEING_CONTROLLED
                and self._session_id == session_id
            ):
                self._last_received_motion_sequence = motion_sequence
        self._send_pointer_state(
            session_id,
            motion_sequence,
            x,
            y,
        )

    def _handle_wheel(self, message: Message) -> None:
        _require_fields(message.metadata, {"session_id", "dx", "dy"})
        session_id = _metadata_session_id(message.metadata)
        self._require_being_controlled(session_id)
        event = WheelEvent(
            _bounded_int(message.metadata["dx"], "wheel dx"),
            _bounded_int(message.metadata["dy"], "wheel dy"),
        )
        self._inject(event, session_id)

    def _handle_pointer_state(self, message: Message) -> None:
        _require_fields(
            message.metadata,
            {"session_id", "motion_sequence", "x", "y"},
        )
        session_id = _metadata_session_id(message.metadata)
        motion_sequence = _motion_sequence(
            message.metadata["motion_sequence"],
            allow_zero=True,
        )
        x = _bounded_int(message.metadata["x"], "pointer x")
        y = _bounded_int(message.metadata["y"], "pointer y")
        with self._lock:
            if (
                self._state is not InputSessionState.CONTROLLING
                or self._session_id != session_id
                or self._pointer is None
                or self._remote_topology is None
            ):
                raise InputProtocolError("stale input pointer state")
            if not self._remote_topology.contains(x, y):
                raise InputProtocolError("remote pointer state is outside its topology")
            if (
                motion_sequence != self._last_motion_sequence
                or self._pending_pointer is not None
            ):
                return
            self._pointer.set_position(x, y)

    def _handle_stop(self, message: Message) -> None:
        if not set(message.metadata).issubset({"session_id", "reason"}):
            raise InputProtocolError("input stop fields are invalid")
        session_id = _metadata_session_id(message.metadata)
        with self._lock:
            if self._session_id != session_id or self._state is InputSessionState.IDLE:
                raise InputProtocolError("stale input stop")
        reason = message.metadata.get("reason", "remote")
        if not isinstance(reason, str):
            raise InputProtocolError("input stop reason is invalid")
        self._finish_session(reason=reason, send_remote=False)

    def _handle_release_all(self, message: Message) -> None:
        _require_fields(message.metadata, {"session_id"})
        session_id = _metadata_session_id(message.metadata)
        self._require_being_controlled(session_id)
        self._backend.release_all()

    def _position_message(self, message: Message) -> tuple[str, int, int]:
        _require_fields(message.metadata, {"session_id", "x", "y"})
        session_id = _metadata_session_id(message.metadata)
        x = _bounded_int(message.metadata["x"], "pointer x")
        y = _bounded_int(message.metadata["y"], "pointer y")
        with self._lock:
            topology = self._local_session_topology
        if topology is None:
            raise InputProtocolError("input session topology is unavailable")
        if not topology.contains(x, y):
            raise InputProtocolError("pointer position is outside connected monitors")
        return session_id, x, y

    def _captured_event(self, event: InputEvent) -> None:
        try:
            with self._lock:
                state = self._state
            if state is InputSessionState.IDLE:
                self._captured_idle_event(event)
            elif state is InputSessionState.CONTROLLING:
                self._captured_controlling_event(event)
        except Exception:
            try:
                self._finish_session(reason="send_failed", send_remote=True)
            except Exception:
                pass

    def _captured_idle_event(self, event: InputEvent) -> None:
        if not isinstance(event, PointerMotionEvent):
            return
        with self._lock:
            if not self._auto_edge_enabled or self._state is not InputSessionState.IDLE:
                return
            side = self._peer_side
        topology = self._local_topology()
        position = self._backend.cursor_position()
        outward = _moves_outward(side, event.dx, event.dy)
        at_edge = topology.is_on_outer_edge(side, *position)
        with self._lock:
            if not at_edge:
                self._reset_edge_hold_locked()
                return
            if outward:
                self._start_edge_timer_locked(side)
            elif _moves_inward(side, event.dx, event.dy):
                self._reset_edge_hold_locked()

    def _captured_controlling_event(self, event: InputEvent) -> None:
        with self._lock:
            if (
                self._state is not InputSessionState.CONTROLLING
                or self._session_id is None
            ):
                return
            session_id = self._session_id
            if isinstance(event, PointerMotionEvent):
                if self._pointer is None:
                    return
                transition = self._pointer.move(event.dx, event.dy)
                if transition.kind is TransitionKind.LEAVE:
                    self._flush_pointer_locked(
                        self._clock(),
                        priority=Priority.INTERACTIVE,
                    )
                    self._send(
                        Message(
                            MessageType.INPUT_RELEASE_ALL,
                            {"session_id": session_id},
                        ),
                        Priority.INTERACTIVE,
                    )
                    self._send(
                        Message(
                            MessageType.INPUT_LEAVE,
                            {
                                "session_id": session_id,
                                "x": transition.x,
                                "y": transition.y,
                            },
                        ),
                        Priority.INTERACTIVE,
                    )
                    leave = True
                else:
                    self._pending_pointer = transition.position
                    now = self._clock()
                    if now - self._last_motion_sent_at >= MOTION_INTERVAL_SECONDS:
                        self._flush_pointer_locked(now)
                    else:
                        self._schedule_motion_flush_locked(now)
                    leave = False
            elif isinstance(event, KeyEvent):
                self._flush_pointer_locked(
                    self._clock(),
                    priority=Priority.INTERACTIVE,
                )
                self._send(Message(MessageType.INPUT_KEY, _encode_key(session_id, event)), Priority.INTERACTIVE)
                leave = False
            elif isinstance(event, MouseButtonEvent):
                self._flush_pointer_locked(
                    self._clock(),
                    priority=Priority.INTERACTIVE,
                )
                self._send(
                    Message(
                        MessageType.INPUT_BUTTON,
                        {
                            "session_id": session_id,
                            "button": event.button.value,
                            "action": event.action.value,
                        },
                    ),
                    Priority.INTERACTIVE,
                )
                leave = False
            elif isinstance(event, WheelEvent):
                self._flush_pointer_locked(
                    self._clock(),
                    priority=Priority.INTERACTIVE,
                )
                self._send(
                    Message(
                        MessageType.INPUT_WHEEL,
                        {"session_id": session_id, "dx": event.dx, "dy": event.dy},
                    ),
                    Priority.INTERACTIVE,
                )
                leave = False
            else:
                return
        if leave:
            self._finish_session(reason="edge", send_remote=False)

    def _flush_pointer_locked(
        self,
        now: float,
        *,
        priority: Priority = Priority.INTERACTIVE,
    ) -> None:
        if self._pending_pointer is None or self._session_id is None:
            return
        self._cancel_motion_timer_locked()
        x, y = self._pending_pointer
        self._pending_pointer = None
        self._last_motion_sent_at = now
        if self._next_motion_sequence > MAX_MOTION_SEQUENCE:
            raise InputUnavailable("input motion sequence is exhausted")
        motion_sequence = self._next_motion_sequence
        self._next_motion_sequence += 1
        self._last_motion_sequence = motion_sequence
        self._send(
            Message(
                MessageType.INPUT_MOVE,
                {
                    "session_id": self._session_id,
                    "motion_sequence": motion_sequence,
                    "x": x,
                    "y": y,
                },
            ),
            priority,
        )

    def _schedule_motion_flush_locked(self, now: float) -> None:
        if self._motion_timer is not None:
            return
        delay = max(
            0.0,
            MOTION_INTERVAL_SECONDS - (now - self._last_motion_sent_at),
        )
        timer = threading.Timer(delay, self._scheduled_motion_flush)
        timer.daemon = True
        self._motion_timer = timer
        timer.start()

    def _scheduled_motion_flush(self) -> None:
        failure = None
        with self._lock:
            self._motion_timer = None
            if (
                self._state is not InputSessionState.CONTROLLING
                or self._pending_pointer is None
            ):
                return
            timestamp = max(
                self._clock(),
                self._last_motion_sent_at + MOTION_INTERVAL_SECONDS,
            )
            try:
                self._flush_pointer_locked(timestamp)
            except Exception as error:
                failure = error
        if failure is not None:
            try:
                self._finish_session(reason="send_failed", send_remote=True)
            except Exception:
                pass

    def _cancel_motion_timer_locked(self) -> None:
        timer = self._motion_timer
        self._motion_timer = None
        if timer is not None:
            timer.cancel()

    def _emergency_stop(self, action: str) -> None:
        try:
            self._finish_session(
                reason="emergency_exit" if action == "exit" else "emergency",
                send_remote=True,
            )
        except Exception:
            pass
        if action != "exit":
            return
        with self._lock:
            listeners = tuple(self._emergency_listeners)
        for listener in listeners:
            try:
                listener(action)
            except Exception:
                pass

    def _inject(self, event: InputEvent, session_id: str) -> None:
        try:
            with self._lock:
                if (
                    self._state is not InputSessionState.BEING_CONTROLLED
                    or self._session_id != session_id
                ):
                    raise InputProtocolError("input session ended before injection")
                self._backend.inject(event)
        except BaseException:
            try:
                self._finish_session(reason="inject_failed", send_remote=True)
            except Exception:
                pass
            raise

    def _send_pointer_state(
        self,
        session_id: str,
        motion_sequence: int,
        x: int,
        y: int,
    ) -> None:
        self._send(
            Message(
                MessageType.INPUT_POINTER_STATE,
                {
                    "session_id": session_id,
                    "motion_sequence": motion_sequence,
                    "x": x,
                    "y": y,
                },
            ),
            Priority.MOTION,
        )

    def _finish_session(self, *, reason: str, send_remote: bool) -> None:
        cleanup_error: BaseException | None = None
        with self._lock:
            state = self._state
            session_id = self._session_id
            if state is InputSessionState.IDLE and session_id is None:
                return
            if state is InputSessionState.CONTROLLING:
                try:
                    self._flush_pointer_locked(
                        self._clock(),
                        priority=Priority.INTERACTIVE,
                    )
                except BaseException:
                    self._pending_pointer = None
            return_position = self._local_return_position
            self._clear_session_locked()
        if state is InputSessionState.CONTROLLING:
            try:
                self._stop_capture_if_running()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if state is InputSessionState.BEING_CONTROLLED:
            try:
                self._backend.release_all()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if send_remote and session_id is not None:
            if state is InputSessionState.CONTROLLING:
                self._safe_send(
                    Message(MessageType.INPUT_RELEASE_ALL, {"session_id": session_id}),
                    Priority.INTERACTIVE,
                )
            self._safe_send(
                Message(
                    MessageType.INPUT_STOP,
                    {"session_id": session_id, "reason": reason},
                ),
                Priority.INTERACTIVE,
            )
        if state is InputSessionState.CONTROLLING and return_position is not None:
            self._restore_local_cursor(return_position)
        self._notify_state(InputStateChange(InputSessionState.IDLE, reason=reason))
        try:
            self._refresh_idle_capture()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error

    def _restore_local_cursor(self, position: tuple[int, int] | None) -> None:
        if position is None:
            return
        try:
            self._backend.warp_cursor(*position)
        except OSError:
            pass

    def _clear_session(self, session_id: str) -> None:
        with self._lock:
            if self._session_id == session_id:
                self._clear_session_locked()

    def _clear_session_locked(self) -> None:
        self._cancel_motion_timer_locked()
        self._state = InputSessionState.IDLE
        self._session_id = None
        self._pointer = None
        self._remote_topology = None
        self._local_session_topology = None
        self._pending_pointer = None
        self._next_motion_sequence = 1
        self._last_motion_sequence = 0
        self._last_received_motion_sequence = 0
        self._enter_from_edge = False
        self._entry_fraction = 0.5
        self._local_return_position = None
        self._reset_edge_hold_locked()

    def _refresh_idle_capture(self) -> None:
        with self._capture_transition_lock:
            with self._lock:
                if (
                    self._state is not InputSessionState.IDLE
                    or self._session_id is not None
                ):
                    return
                should_capture = (
                    not self._closed
                    and self._connected
                    and self._bus.trusted
                    and self._auto_edge_enabled
                    and self._backend.permission_status().capture_allowed
                )
                running = self._backend.capture_running
            if should_capture and not running:
                self._backend.start_capture(
                    self._captured_event,
                    self._emergency_stop,
                    suppress=False,
                )
            elif not should_capture and running:
                self._backend.stop_capture()

    def _refresh_idle_capture_if_idle(self) -> None:
        self._refresh_idle_capture()

    def _start_capture(self, on_event, on_emergency, *, suppress: bool) -> None:
        with self._capture_transition_lock:
            self._backend.start_capture(
                on_event,
                on_emergency,
                suppress=suppress,
            )

    def _stop_capture_if_running(self) -> None:
        with self._capture_transition_lock:
            if self._backend.capture_running:
                self._backend.stop_capture()

    def _require_being_controlled(self, session_id: str) -> None:
        with self._lock:
            if (
                self._state is not InputSessionState.BEING_CONTROLLED
                or self._session_id != session_id
            ):
                raise InputProtocolError("stale remotely controlled input event")

    def _incoming_busy_reason_locked(
        self,
        controller_id: str,
        session_id: str,
        inject_allowed: bool,
    ) -> str | None:
        if self._closed or not self._connected:
            return "disconnected"
        if not self._allow_remote_input or not inject_allowed:
            return "permission"
        if self._state is InputSessionState.IDLE:
            return None
        if self._state is InputSessionState.REQUESTING:
            local_contender = (self.local_peer_id, self._session_id or "")
            remote_contender = (controller_id, session_id)
            return None if remote_contender < local_contender else "busy"
        return "busy"

    def _local_topology(self) -> Topology:
        monitors = tuple(self._backend.monitors())
        if not monitors:
            raise InputUnavailable("no connected display topology is available")
        try:
            topology = Topology(monitors)
            return _decode_topology(_encode_topology(topology))
        except (InputProtocolError, TypeError, ValueError) as error:
            raise InputUnavailable(f"local monitor topology is invalid: {error}") from error

    def _send(self, message: Message, priority: Priority) -> None:
        self._bus.send(message, secure=True, priority=priority)

    def _safe_send(self, message: Message, priority: Priority) -> None:
        try:
            self._send(message, priority)
        except Exception:
            pass

    def _send_busy(self, session_id: str, reason: str) -> None:
        self._send(
            Message(
                MessageType.INPUT_BUSY,
                {"session_id": session_id, "reason": reason},
            ),
            Priority.INTERACTIVE,
        )

    def _reset_edge_hold_locked(self) -> None:
        timer = self._edge_timer
        self._edge_timer = None
        if timer is not None:
            timer.cancel()

    def _start_edge_timer_locked(self, side: Side) -> None:
        if self._edge_timer is not None:
            return
        timer = threading.Timer(
            EDGE_HOLD_SECONDS,
            lambda: self._edge_timer_elapsed(timer, side),
        )
        timer.daemon = True
        self._edge_timer = timer
        timer.start()

    def _edge_timer_elapsed(self, timer: threading.Timer, side: Side) -> None:
        with self._lock:
            if self._edge_timer is not timer:
                return
            self._edge_timer = None
            eligible = (
                not self._closed
                and self._connected
                and self._bus.trusted
                and self._auto_edge_enabled
                and self._state is InputSessionState.IDLE
                and self._session_id is None
                and self._peer_side is side
            )
        if not eligible:
            return
        try:
            topology = self._local_topology()
            position = self._backend.cursor_position()
        except (InputUnavailable, OSError, TypeError, ValueError):
            return
        if not topology.is_on_outer_edge(side, *position):
            return
        try:
            self._request_control(enter_from_edge=True)
        except InputUnavailable:
            pass

    def _notify_state(self, change: InputStateChange) -> None:
        with self._lock:
            listeners = tuple(self._state_listeners)
        for listener in listeners:
            listener(change)


def _validate_peer_id(peer_id: Any) -> None:
    if not isinstance(peer_id, str) or not peer_id:
        raise ValueError("peer ID must be a non-empty string")
    try:
        encoded = peer_id.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("peer ID must be valid UTF-8") from error
    if len(encoded) > MAX_PEER_ID_BYTES or any(ord(character) < 32 for character in peer_id):
        raise ValueError("peer ID is invalid")


def _validate_session_id(session_id: Any) -> None:
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise InputProtocolError("input session ID is invalid")


def _metadata_session_id(metadata: Mapping[str, Any]) -> str:
    try:
        session_id = metadata["session_id"]
    except (KeyError, TypeError) as error:
        raise InputProtocolError("input session ID is missing") from error
    _validate_session_id(session_id)
    return session_id


def _require_fields(metadata: Mapping[str, Any], expected: set[str]) -> None:
    if not isinstance(metadata, dict) or set(metadata) != expected:
        raise InputProtocolError("input message fields are invalid")


def _encode_topology(topology: Topology) -> list[dict[str, int | str]]:
    return [
        {
            "id": monitor.monitor_id,
            "x": monitor.rect.x,
            "y": monitor.rect.y,
            "width": monitor.rect.width,
            "height": monitor.rect.height,
        }
        for monitor in topology.monitors
    ]


def _decode_topology(value: Any) -> Topology:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_MONITORS:
        raise InputProtocolError("input topology must contain connected monitors")
    monitors = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "x", "y", "width", "height"}:
            raise InputProtocolError("input monitor fields are invalid")
        monitor_id = item["id"]
        if not isinstance(monitor_id, str) or not monitor_id or len(monitor_id) > 128:
            raise InputProtocolError("input monitor ID is invalid")
        x = _bounded_int(item["x"], "monitor x")
        y = _bounded_int(item["y"], "monitor y")
        width = _positive_bounded_int(item["width"], "monitor width")
        height = _positive_bounded_int(item["height"], "monitor height")
        if (
            x + width - 1 > MAX_COORDINATE
            or y + height - 1 > MAX_COORDINATE
        ):
            raise InputProtocolError("input monitor exceeds coordinate bounds")
        monitors.append(Monitor(monitor_id, Rect(x, y, width, height)))
    try:
        return Topology(monitors)
    except (TypeError, ValueError) as error:
        raise InputProtocolError("input topology is invalid") from error


def _bounded_int(value: Any, name: str) -> int:
    if type(value) is not int or not -MAX_COORDINATE <= value <= MAX_COORDINATE:
        raise InputProtocolError(f"{name} is invalid")
    return value


def _positive_bounded_int(value: Any, name: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_COORDINATE:
        raise InputProtocolError(f"{name} is invalid")
    return value


def _motion_sequence(value: Any, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or not minimum <= value <= MAX_MOTION_SEQUENCE:
        raise InputProtocolError("input motion sequence is invalid")
    return value


def _entry_fraction(topology: Topology, side: Side, position: tuple[int, int]) -> float:
    if topology.is_on_outer_edge(side, *position):
        return topology.edge_fraction(side, *position)
    return 0.5


def _return_position(
    topology: Topology,
    side: Side,
    position: tuple[int, int],
) -> tuple[int, int]:
    x, y = position
    if topology.is_on_outer_edge(side, x, y):
        if side is Side.LEFT:
            x += 1
        elif side is Side.RIGHT:
            x -= 1
        elif side is Side.TOP:
            y += 1
        else:
            y -= 1
    return topology.nearest_point(x, y)


def _moves_outward(side: Side, dx: int, dy: int) -> bool:
    return {
        Side.LEFT: dx < 0,
        Side.RIGHT: dx > 0,
        Side.TOP: dy < 0,
        Side.BOTTOM: dy > 0,
    }[side]


def _moves_inward(side: Side, dx: int, dy: int) -> bool:
    return _moves_outward(side.opposite, dx, dy)


def _encode_key(session_id: str, event: KeyEvent) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "action": event.action.value,
        "usage": event.usage,
        "scan_code": event.scan_code,
        "virtual_key": event.virtual_key,
        "text": event.text,
        "modifiers": int(event.modifiers),
        "location": event.location.value,
        "repeat": event.repeat,
        "extended": event.extended,
    }


__all__ = [
    "EDGE_HOLD_SECONDS",
    "InputProtocolError",
    "InputService",
    "InputSessionState",
    "InputStateChange",
    "InputUnavailable",
    "MOTION_INTERVAL_SECONDS",
]
