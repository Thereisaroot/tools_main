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
        with self._backend_lock:
            if self._capture_running:
                raise RuntimeError("input capture is already running")
            self._capture_callback = on_event
            self._emergency_callback = on_emergency
            self._capture_suppress = bool(suppress)
            self._capture_running = True
        try:
            self._start_native_capture(bool(suppress))
        except BaseException:
            with self._backend_lock:
                self._capture_running = False
                self._capture_callback = None
                self._emergency_callback = None
            raise

    def stop_capture(self) -> None:
        with self._backend_lock:
            if not self._capture_running:
                return
            self._capture_running = False
        try:
            self._stop_native_capture()
        finally:
            with self._backend_lock:
                self._capture_callback = None
                self._emergency_callback = None
                self._capture_suppress = False

    def emit_captured(self, event: InputEvent) -> bool:
        if not isinstance(
            event,
            (KeyEvent, MouseButtonEvent),
        ) and not hasattr(event, "injected"):
            raise TypeError("event must be a normalized input event")
        if event.injected:
            return True
        with self._backend_lock:
            if not self._capture_running:
                return False
            callback = self._capture_callback
            emergency = self._emergency_callback
        if _is_emergency_stop(event):
            if emergency is not None:
                emergency("stop")
            return True
        if callback is not None:
            callback(event)
        return False

    def inject(self, event: InputEvent) -> None:
        self._inject_native(event)
        with self._backend_lock:
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
            self._pressed_keys.clear()
            self._pressed_buttons.clear()
        for event in keys:
            self._inject_native(replace(event, action=KeyAction.UP, repeat=False))
        for event in buttons:
            self._inject_native(replace(event, action=KeyAction.UP))

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


def _is_emergency_stop(event: InputEvent) -> bool:
    required = Modifiers.CONTROL | Modifiers.ALT | Modifiers.SHIFT
    return (
        isinstance(event, KeyEvent)
        and event.action is KeyAction.DOWN
        and event.usage == 0x29
        and event.modifiers & required == required
    )


__all__ = ["BaseInputBackend", "PermissionStatus"]
