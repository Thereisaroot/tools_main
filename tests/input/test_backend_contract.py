import sys
import threading

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


def test_backend_lifecycle_filters_injected_events_and_stops_idempotently():
    backend = MemoryBackend()
    captured = []
    emergency = []
    backend.start_capture(captured.append, emergency.append, suppress=True)

    backend.emit_captured(key(KeyAction.DOWN))
    backend.emit_captured(key(KeyAction.DOWN, injected=True))

    assert captured == [key(KeyAction.DOWN)]
    assert emergency == []
    assert backend.native_started == 1
    with pytest.raises(RuntimeError, match="already"):
        backend.start_capture(captured.append, emergency.append)

    backend.stop_capture()
    backend.stop_capture()
    assert backend.native_stopped == 1


@pytest.mark.parametrize(
    ("usage", "action"),
    [(0x2A, "stop"), (0x29, "exit")],
)
def test_backend_consumes_emergency_shortcuts_before_forwarding(usage, action):
    backend = MemoryBackend()
    captured = []
    emergency = []
    backend.start_capture(captured.append, emergency.append)

    consumed = backend.emit_captured(
        key(
            KeyAction.DOWN,
            usage=usage,
            modifiers=Modifiers.CONTROL | Modifiers.ALT | Modifiers.SHIFT,
        )
    )

    assert consumed is True
    assert emergency == [action]
    assert captured == []


def test_failed_native_stop_retains_capture_state_for_retry():
    class RetryStopBackend(MemoryBackend):
        def __init__(self):
            super().__init__()
            self.fail_stop = True

        def _stop_native_capture(self):
            self.native_stopped += 1
            if self.fail_stop:
                raise OSError("quit failed")

    backend = RetryStopBackend()
    captured = []
    backend.start_capture(captured.append, lambda _action: None)

    with pytest.raises(RuntimeError, match="state retained for retry"):
        backend.stop_capture()

    assert backend.capture_running is True
    backend.emit_captured(key(KeyAction.DOWN))
    assert captured == [key(KeyAction.DOWN)]

    backend.fail_stop = False
    backend.stop_capture()
    backend.stop_capture()
    assert backend.capture_running is False
    assert backend.native_stopped == 2


def test_concurrent_stop_waits_for_native_start_then_stops_it():
    class BlockingStartBackend(MemoryBackend):
        def __init__(self):
            super().__init__()
            self.start_entered = threading.Event()
            self.allow_start = threading.Event()
            self.native_stop_entered = threading.Event()
            self.native_active = False

        def _start_native_capture(self, suppress):
            self.native_started += 1
            self.start_entered.set()
            assert self.allow_start.wait(2)
            self.native_active = True

        def _stop_native_capture(self):
            self.native_stopped += 1
            self.native_stop_entered.set()
            self.native_active = False

    backend = BlockingStartBackend()
    start_thread = threading.Thread(
        target=backend.start_capture,
        args=(lambda _event: None, lambda _action: None),
    )
    start_thread.start()
    assert backend.start_entered.wait(2)

    stop_thread = threading.Thread(target=backend.stop_capture)
    stop_thread.start()
    stopped_before_start_completed = backend.native_stop_entered.wait(0.2)
    backend.allow_start.set()
    start_thread.join(2)
    stop_thread.join(2)

    assert not start_thread.is_alive()
    assert not stop_thread.is_alive()
    assert stopped_before_start_completed is False
    assert backend.native_active is False
    assert backend.capture_running is False
    assert backend.native_started == backend.native_stopped == 1


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


def test_release_all_waits_for_in_flight_injection_and_releases_it():
    class BlockingInjectionBackend(MemoryBackend):
        def __init__(self):
            super().__init__()
            self.down_entered = threading.Event()
            self.allow_down = threading.Event()

        def _inject_native(self, event):
            self.injected.append(event)
            if isinstance(event, KeyEvent) and event.action is KeyAction.DOWN:
                self.down_entered.set()
                assert self.allow_down.wait(2)

    backend = BlockingInjectionBackend()
    inject_thread = threading.Thread(target=backend.inject, args=(key(KeyAction.DOWN),))
    inject_thread.start()
    assert backend.down_entered.wait(2)

    release_finished = threading.Event()

    def release_all():
        backend.release_all()
        release_finished.set()

    release_thread = threading.Thread(target=release_all)
    release_thread.start()
    released_before_injection_completed = release_finished.wait(0.2)
    backend.allow_down.set()
    inject_thread.join(2)
    release_thread.join(2)

    assert released_before_injection_completed is False
    assert backend.injected == [key(KeyAction.DOWN), key(KeyAction.UP)]
    assert backend.pressed_keys == frozenset()


def test_failed_release_retains_only_unreleased_inputs_for_retry():
    class RetryReleaseBackend(MemoryBackend):
        def __init__(self):
            super().__init__()
            self.failed_once = False

        def _inject_native(self, event):
            if (
                isinstance(event, KeyEvent)
                and event.action is KeyAction.UP
                and not self.failed_once
            ):
                self.failed_once = True
                raise OSError("release failed")
            self.injected.append(event)

    backend = RetryReleaseBackend()
    key_down = key(KeyAction.DOWN)
    button_down = MouseButtonEvent(MouseButton.LEFT, KeyAction.DOWN)
    backend.inject(key_down)
    backend.inject(button_down)

    with pytest.raises(OSError, match="release failed"):
        backend.release_all()

    assert backend.pressed_keys == frozenset({key_down.usage})
    assert backend.pressed_buttons == frozenset({MouseButton.LEFT})

    backend.release_all()
    assert backend.injected[-2:] == [
        key(KeyAction.UP),
        MouseButtonEvent(MouseButton.LEFT, KeyAction.UP),
    ]
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
