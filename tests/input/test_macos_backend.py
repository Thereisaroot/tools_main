import sys
from types import ModuleType

import pytest

from shooklink.input.backend import PermissionStatus
from shooklink.input.events import KeyAction, KeyEvent
from shooklink.input.macos_backend import MacOSInputBackend


@pytest.mark.parametrize(
    "status",
    [
        PermissionStatus(False, True, "capture denied"),
        PermissionStatus(True, False, "injection denied"),
    ],
)
def test_macos_start_requires_capture_and_injection_permissions(monkeypatch, status):
    backend = MacOSInputBackend()
    monkeypatch.setattr(backend, "permission_status", lambda: status)

    def unexpected_thread(**_kwargs):
        raise AssertionError("event-tap thread must not start without both permissions")

    monkeypatch.setattr("shooklink.input.macos_backend.threading.Thread", unexpected_thread)

    with pytest.raises(PermissionError, match=status.detail):
        backend.start_capture(lambda _event: None, lambda _action: None)

    assert backend.capture_running is False
    assert backend._tap_thread is None


@pytest.mark.parametrize(
    "status",
    [
        PermissionStatus(False, True, "capture denied"),
        PermissionStatus(True, False, "injection denied"),
    ],
)
def test_macos_injection_requires_capture_and_injection_permissions(
    monkeypatch,
    status,
):
    class ForbiddenQuartz(ModuleType):
        def __getattr__(self, name):
            raise AssertionError(f"Quartz.{name} used without permissions")

    backend = MacOSInputBackend()
    monkeypatch.setattr(backend, "permission_status", lambda: status)
    monkeypatch.setitem(sys.modules, "Quartz", ForbiddenQuartz("Quartz"))

    with pytest.raises(PermissionError, match=status.detail):
        backend.inject(KeyEvent(KeyAction.DOWN, usage=0x04))

    assert backend.pressed_keys == frozenset()


def test_macos_stop_preserves_native_state_when_run_loop_stop_fails(monkeypatch):
    class LiveThread:
        def is_alive(self):
            return True

    core_foundation = ModuleType("CoreFoundation")

    def fail_stop(_run_loop):
        raise OSError("CFRunLoopStop failed")

    core_foundation.CFRunLoopStop = fail_stop
    monkeypatch.setitem(sys.modules, "CoreFoundation", core_foundation)

    backend = MacOSInputBackend()
    thread = LiveThread()
    run_loop = object()
    event_tap = object()
    callback = object()
    backend._tap_thread = thread
    backend._run_loop = run_loop
    backend._event_tap = event_tap
    backend._tap_callback_ref = callback

    with pytest.raises(RuntimeError, match="could not stop macOS run loop"):
        backend._stop_native_capture()

    assert backend._tap_thread is thread
    assert backend._run_loop is run_loop
    assert backend._event_tap is event_tap
    assert backend._tap_callback_ref is callback


def test_macos_stop_preserves_native_state_when_thread_does_not_exit(monkeypatch):
    class StuckThread:
        def is_alive(self):
            return True

        def join(self, _timeout):
            pass

    core_foundation = ModuleType("CoreFoundation")
    core_foundation.CFRunLoopStop = lambda _run_loop: None
    monkeypatch.setitem(sys.modules, "CoreFoundation", core_foundation)

    backend = MacOSInputBackend()
    thread = StuckThread()
    run_loop = object()
    event_tap = object()
    callback = object()
    backend._tap_thread = thread
    backend._run_loop = run_loop
    backend._event_tap = event_tap
    backend._tap_callback_ref = callback

    with pytest.raises(RuntimeError, match="did not stop"):
        backend._stop_native_capture()

    assert backend._tap_thread is thread
    assert backend._run_loop is run_loop
    assert backend._event_tap is event_tap
    assert backend._tap_callback_ref is callback


def test_macos_successful_stop_is_idempotent(monkeypatch):
    class JoiningThread:
        def __init__(self):
            self.alive = True
            self.joins = 0

        def is_alive(self):
            return self.alive

        def join(self, _timeout):
            self.joins += 1
            self.alive = False

    stopped = []
    core_foundation = ModuleType("CoreFoundation")
    core_foundation.CFRunLoopStop = stopped.append
    monkeypatch.setitem(sys.modules, "CoreFoundation", core_foundation)

    backend = MacOSInputBackend()
    thread = JoiningThread()
    run_loop = object()
    backend._tap_thread = thread
    backend._run_loop = run_loop
    backend._event_tap = object()
    backend._tap_callback_ref = object()

    backend._stop_native_capture()
    backend._stop_native_capture()

    assert stopped == [run_loop]
    assert thread.joins == 1
    assert backend._tap_thread is None
    assert backend._run_loop is None
    assert backend._event_tap is None
    assert backend._tap_callback_ref is None
