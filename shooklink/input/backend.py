"""Lifecycle and pressed-input safety shared by native input backends."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

from shooklink.input.events import (
    InputEvent,
    KeyAction,
    KeyEvent,
    Modifiers,
    MouseButton,
    MouseButtonEvent,
)


@dataclass(frozen=True, slots=True)
class PermissionStatus:
    capture_allowed: bool
    inject_allowed: bool
    detail: str


class BaseInputBackend:
    def __init__(self) -> None:
        self._lifecycle_lock = threading.RLock()
        self._backend_lock = threading.RLock()
        self._capture_callback: Callable[[InputEvent], None] | None = None
        self._emergency_callback: Callable[[str], None] | None = None
        self._capture_suppress = False
        self._capture_running = False
        self._pressed_keys: dict[int, KeyEvent] = {}
        self._pressed_buttons: dict[MouseButton, MouseButtonEvent] = {}

    @property
    def capture_running(self) -> bool:
        with self._backend_lock:
            return self._capture_running

    @property
    def pressed_keys(self) -> frozenset[int]:
        with self._backend_lock:
            return frozenset(self._pressed_keys)

    @property
    def pressed_buttons(self) -> frozenset[MouseButton]:
        with self._backend_lock:
            return frozenset(self._pressed_buttons)

    def start_capture(
        self,
        on_event: Callable[[InputEvent], None],
        on_emergency: Callable[[str], None],
        *,
        suppress: bool = False,
    ) -> None:
        if not callable(on_event) or not callable(on_emergency):
            raise TypeError("capture callbacks must be callable")
        with self._lifecycle_lock:
            with self._backend_lock:
                if self._capture_running:
                    raise RuntimeError("input capture is already running")
                self._capture_callback = on_event
                self._emergency_callback = on_emergency
                self._capture_suppress = bool(suppress)
                self._capture_running = True
            try:
                self._start_native_capture(bool(suppress))
            except BaseException as start_error:
                try:
                    self._stop_native_capture()
                except BaseException as stop_error:
                    failure = RuntimeError(
                        "native input capture failed to start and cleanup failed; "
                        "state retained for retry"
                    )
                    failure.add_note(f"start failure: {start_error!r}")
                    raise failure from stop_error
                with self._backend_lock:
                    self._clear_capture_state()
                raise

    def stop_capture(self) -> None:
        with self._lifecycle_lock:
            with self._backend_lock:
                if not self._capture_running:
                    return
            try:
                self._stop_native_capture()
            except Exception as error:
                raise RuntimeError(
                    "native input capture failed to stop; state retained for retry"
                ) from error
            with self._backend_lock:
                self._clear_capture_state()

    def emit_captured(self, event: InputEvent) -> bool:
        if not isinstance(
            event,
            (KeyEvent, MouseButtonEvent),
        ) and not hasattr(event, "injected"):
            raise TypeError("event must be a normalized input event")
        if event.self_injected:
            return True
        with self._backend_lock:
            if not self._capture_running:
                return False
            callback = self._capture_callback
            emergency = self._emergency_callback
        emergency_action = _emergency_action(event)
        if emergency_action is not None:
            if emergency is not None:
                emergency(emergency_action)
            return True
        if callback is not None:
            callback(event)
        return False

    def inject(self, event: InputEvent) -> None:
        with self._backend_lock:
            self._inject_native(event)
            if isinstance(event, KeyEvent):
                if event.action is KeyAction.DOWN:
                    self._pressed_keys[event.usage] = event
                else:
                    self._pressed_keys.pop(event.usage, None)
            elif isinstance(event, MouseButtonEvent):
                if event.action is KeyAction.DOWN:
                    self._pressed_buttons[event.button] = event
                else:
                    self._pressed_buttons.pop(event.button, None)

    def release_all(self) -> None:
        with self._backend_lock:
            keys = tuple(self._pressed_keys.values())
            buttons = tuple(self._pressed_buttons.values())
            for event in keys:
                self._inject_native(replace(event, action=KeyAction.UP, repeat=False))
                self._pressed_keys.pop(event.usage, None)
            for event in buttons:
                self._inject_native(replace(event, action=KeyAction.UP))
                self._pressed_buttons.pop(event.button, None)

    def _clear_capture_state(self) -> None:
        self._capture_running = False
        self._capture_callback = None
        self._emergency_callback = None
        self._capture_suppress = False

    def permission_status(self) -> PermissionStatus:
        raise NotImplementedError

    def monitors(self):
        raise NotImplementedError

    def cursor_position(self) -> tuple[int, int]:
        raise NotImplementedError

    def warp_cursor(self, x: int, y: int) -> None:
        raise NotImplementedError

    def _start_native_capture(self, suppress: bool) -> None:
        raise NotImplementedError

    def _stop_native_capture(self) -> None:
        raise NotImplementedError

    def _inject_native(self, event: InputEvent) -> None:
        raise NotImplementedError


def _emergency_action(event: InputEvent) -> str | None:
    required = Modifiers.CONTROL | Modifiers.ALT | Modifiers.SHIFT
    if not (
        isinstance(event, KeyEvent)
        and event.action is KeyAction.DOWN
        and event.modifiers & required == required
    ):
        return None
    return {0x2A: "stop", 0x29: "exit"}.get(event.usage)


__all__ = ["BaseInputBackend", "PermissionStatus"]
