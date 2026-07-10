import sys

import pytest

from shooklink.input.backend import BaseInputBackend, PermissionStatus
from shooklink.input.events import (
    KeyAction,
    KeyEvent,
    Modifiers,
    MouseButton,
    MouseButtonEvent,
)
from shooklink.input.topology import Monitor, Rect


class MemoryBackend(BaseInputBackend):
    def __init__(self):
        super().__init__()
        self.native_started = 0
        self.native_stopped = 0
        self.injected = []
        self.position = (10, 20)

    def permission_status(self):
        return PermissionStatus(True, True, "ready")

    def monitors(self):
        return (Monitor("display", Rect(-100, 0, 1920, 1080)),)

    def cursor_position(self):
        return self.position

    def warp_cursor(self, x, y):
        self.position = (x, y)

    def _start_native_capture(self, suppress):
        self.native_started += 1

    def _stop_native_capture(self):
        self.native_stopped += 1

    def _inject_native(self, event):
        self.injected.append(event)


def key(action, *, injected=False, modifiers=Modifiers.NONE, usage=4):
    return KeyEvent(
        action,
        usage=usage,
        scan_code=0,
        virtual_key=0,
        text="a",
        modifiers=modifiers,
        injected=injected,
    )


def test_backend_lifecycle_filters_injected_and_consumes_emergency_shortcut():
    backend = MemoryBackend()
    captured = []
    emergency = []
    backend.start_capture(captured.append, emergency.append, suppress=True)

    backend.emit_captured(key(KeyAction.DOWN))
    backend.emit_captured(key(KeyAction.DOWN, injected=True))
    backend.emit_captured(
        key(
            KeyAction.DOWN,
            usage=0x29,
            modifiers=Modifiers.CONTROL | Modifiers.ALT | Modifiers.SHIFT,
        )
    )

    assert captured == [key(KeyAction.DOWN)]
    assert emergency == ["stop"]
    assert backend.native_started == 1
    with pytest.raises(RuntimeError, match="already"):
        backend.start_capture(captured.append, emergency.append)

    backend.stop_capture()
    backend.stop_capture()
    assert backend.native_stopped == 1


def test_backend_injection_tracks_and_releases_pressed_inputs():
    backend = MemoryBackend()
    key_down = key(KeyAction.DOWN)
    button_down = MouseButtonEvent(MouseButton.LEFT, KeyAction.DOWN)
    backend.inject(key_down)
    backend.inject(button_down)

    backend.release_all()

    assert backend.injected[-2].action is KeyAction.UP
    assert backend.injected[-2].usage == key_down.usage
    assert backend.injected[-1] == MouseButtonEvent(MouseButton.LEFT, KeyAction.UP)
    assert backend.pressed_keys == frozenset()
    assert backend.pressed_buttons == frozenset()


def test_backend_reports_monitors_permissions_and_cursor_warp():
    backend = MemoryBackend()

    assert backend.permission_status().capture_allowed
    assert backend.monitors()[0].rect.x == -100
    assert backend.cursor_position() == (10, 20)
    backend.warp_cursor(-50, 100)
    assert backend.cursor_position() == (-50, 100)


def test_windows_backend_import_is_side_effect_free_off_windows():
    import shooklink.input.windows_backend as windows_backend

    assert windows_backend.WindowsInputBackend
    if sys.platform != "win32":
        assert "user32" not in vars(windows_backend)
