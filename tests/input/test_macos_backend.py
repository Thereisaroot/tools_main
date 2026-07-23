import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from shooklink.input.backend import PermissionStatus
from shooklink.input.events import KeyAction, KeyEvent, PointerMotionEvent
from shooklink.input.macos_backend import MacOSInputBackend
from shooklink.input.topology import Monitor, Rect


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


def test_macos_suppressed_capture_starts_at_current_monitor_center(monkeypatch):
    backend = MacOSInputBackend()
    monkeypatch.setattr(
        backend,
        "permission_status",
        lambda: PermissionStatus(True, True, "ready"),
    )
    monkeypatch.setattr(backend, "cursor_position", lambda: (1700, 900))
    monkeypatch.setattr(
        backend,
        "monitors",
        lambda: (
            Monitor("left", Rect(0, 0, 1920, 1080)),
            Monitor("right", Rect(1920, 0, 1920, 1080)),
        ),
    )
    warps = []
    monkeypatch.setattr(backend, "warp_cursor", lambda x, y: warps.append((x, y)))

    class ImmediateThread:
        def __init__(self, *, args, **_kwargs):
            self.generation = args[0]

        def start(self):
            self.generation.ready.set()

        def is_alive(self):
            return False

    monkeypatch.setattr("shooklink.input.macos_backend.threading.Thread", ImmediateThread)

    backend.start_capture(
        lambda _event: None,
        lambda _action: None,
        suppress=True,
    )

    assert warps == [(960, 540)]
    assert backend._mouse_anchor_position == (960, 540)
    assert backend._mouse_anchor_rect == Rect(0, 0, 1920, 1080)
    backend.stop_capture()


def test_macos_suppressed_motion_uses_event_location_and_reanchors(monkeypatch):
    class Quartz:
        kCGEventSourceUserData = 1
        kCGEventKeyDown = 2
        kCGEventKeyUp = 3
        kCGEventFlagsChanged = 4
        kCGEventMouseMoved = 5
        kCGEventLeftMouseDragged = 6
        kCGEventRightMouseDragged = 7
        kCGEventOtherMouseDragged = 8
        kCGEventScrollWheel = 9

        @staticmethod
        def CGEventGetIntegerValueField(_event, field):
            assert field == Quartz.kCGEventSourceUserData
            return 0

        @staticmethod
        def CGEventGetLocation(event):
            return SimpleNamespace(x=event[0], y=event[1])

    backend = MacOSInputBackend()
    backend._capture_suppress = True
    backend._mouse_anchor_position = (500, 500)
    backend._mouse_anchor_rect = Rect(0, 0, 1000, 1000)
    warps = []
    monkeypatch.setattr(backend, "warp_cursor", lambda x, y: warps.append((x, y)))

    event = backend._normalize_event(Quartz.kCGEventMouseMoved, (504, 497), Quartz)
    bogus = backend._normalize_event(Quartz.kCGEventMouseMoved, (0, 500), Quartz)

    assert isinstance(event, PointerMotionEvent)
    assert (event.dx, event.dy) == (4, -3)
    assert bogus is None
    assert warps == [(500, 500), (500, 500)]


def test_macos_suppressed_mouse_move_is_returned_so_quartz_applies_recenter(
    monkeypatch,
):
    quartz = ModuleType("Quartz")
    quartz.kCGEventTapDisabledByTimeout = 1
    quartz.kCGEventTapDisabledByUserInput = 2
    quartz.kCGEventMouseMoved = 3
    quartz.CGEventTapEnable = lambda *_args: None
    monkeypatch.setitem(sys.modules, "Quartz", quartz)

    backend = MacOSInputBackend()
    backend._capture_running = True
    backend._capture_suppress = True
    captured = []
    backend._capture_callback = captured.append
    normalized = PointerMotionEvent(3, -2)
    monkeypatch.setattr(backend, "_normalize_event", lambda *_args: normalized)
    native_event = object()

    result = backend._tap_callback(None, quartz.kCGEventMouseMoved, native_event, None)

    assert result is native_event
    assert captured == [normalized]


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


def test_macos_callback_thread_restart_isolated_from_stale_cleanup(monkeypatch):
    backend = MacOSInputBackend()
    monkeypatch.setattr(
        backend,
        "permission_status",
        lambda: PermissionStatus(True, True, "ready"),
    )

    real_thread = threading.Thread
    threads = []

    def thread_factory(**kwargs):
        thread = real_thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr("shooklink.input.macos_backend.threading.Thread", thread_factory)

    restart_complete = threading.Event()
    release_old_cleanup = threading.Event()
    errors = []
    thread_state = threading.local()
    stop_events = {}
    loop_count = 0

    core_foundation = ModuleType("CoreFoundation")

    def current_run_loop():
        nonlocal loop_count
        loop_count += 1
        run_loop = f"run-loop-{loop_count}"
        thread_state.run_loop = run_loop
        stop_events[run_loop] = threading.Event()
        return run_loop

    def run_loop():
        current = thread_state.run_loop
        if current == "run-loop-1":
            try:
                backend.stop_capture()
                backend.start_capture(lambda _event: None, lambda _action: None)
            except BaseException as error:
                errors.append(error)
            finally:
                restart_complete.set()
            assert release_old_cleanup.wait(2)
            return
        assert stop_events[current].wait(2)

    core_foundation.CFRunLoopGetCurrent = current_run_loop
    core_foundation.CFRunLoopAddSource = lambda *_args: None
    core_foundation.CFRunLoopRun = run_loop
    core_foundation.CFRunLoopStop = lambda value: stop_events[value].set()
    core_foundation.kCFRunLoopCommonModes = object()
    monkeypatch.setitem(sys.modules, "CoreFoundation", core_foundation)

    quartz = ModuleType("Quartz")
    event_names = (
        "kCGEventKeyDown",
        "kCGEventKeyUp",
        "kCGEventFlagsChanged",
        "kCGEventMouseMoved",
        "kCGEventLeftMouseDragged",
        "kCGEventRightMouseDragged",
        "kCGEventOtherMouseDragged",
        "kCGEventLeftMouseDown",
        "kCGEventLeftMouseUp",
        "kCGEventRightMouseDown",
        "kCGEventRightMouseUp",
        "kCGEventOtherMouseDown",
        "kCGEventOtherMouseUp",
        "kCGEventScrollWheel",
    )
    for event_type, name in enumerate(event_names, start=1):
        setattr(quartz, name, event_type)
    quartz.kCGHIDEventTap = 1
    quartz.kCGHeadInsertEventTap = 2
    quartz.kCGEventTapOptionDefault = 3
    quartz.kCGEventSourceStateCombinedSessionState = 4
    quartz.CGEventSourceKeyState = lambda *_args: False
    quartz.CGEventTapCreate = lambda *_args: object()
    quartz.CFMachPortCreateRunLoopSource = lambda *_args: object()
    quartz.CGEventTapEnable = lambda *_args: None
    monkeypatch.setitem(sys.modules, "Quartz", quartz)

    try:
        backend.start_capture(lambda _event: None, lambda _action: None)
        assert restart_complete.wait(2)
        assert errors == []
        assert len(threads) == 2

        new_thread = threads[1]
        new_run_loop = backend._run_loop
        new_event_tap = backend._event_tap
        new_callback = backend._tap_callback_ref
        assert new_run_loop == "run-loop-2"
        assert new_event_tap is not None
        assert new_callback is not None

        release_old_cleanup.set()
        threads[0].join(2)

        assert not threads[0].is_alive()
        assert backend._tap_thread is new_thread
        assert backend._run_loop is new_run_loop
        assert backend._event_tap is new_event_tap
        assert backend._tap_callback_ref is new_callback
    finally:
        release_old_cleanup.set()
        for event in stop_events.values():
            event.set()
        for thread in threads:
            thread.join(2)


def test_macos_new_generation_resets_modifier_key_tracking(monkeypatch):
    backend = MacOSInputBackend()
    backend._modifier_keys_down = {56}
    monkeypatch.setattr(
        backend,
        "permission_status",
        lambda: PermissionStatus(True, True, "ready"),
    )

    class ImmediateThread:
        def __init__(self, *, args, **_kwargs):
            self.generation = args[0]

        def start(self):
            self.generation.ready.set()

        def is_alive(self):
            return False

    monkeypatch.setattr("shooklink.input.macos_backend.threading.Thread", ImmediateThread)
    backend.start_capture(lambda _event: None, lambda _action: None)

    class Quartz:
        kCGEventKeyDown = 1
        kCGEventKeyUp = 2
        kCGEventFlagsChanged = 3
        kCGEventSourceUserData = 4
        kCGKeyboardEventKeycode = 5
        kCGKeyboardEventAutorepeat = 6
        kCGEventFlagMaskShift = 1 << 16
        kCGEventFlagMaskControl = 1 << 17
        kCGEventFlagMaskAlternate = 1 << 18
        kCGEventFlagMaskCommand = 1 << 19
        kCGEventFlagMaskAlphaShift = 1 << 20

        @staticmethod
        def CGEventGetIntegerValueField(_event, field):
            return {
                Quartz.kCGEventSourceUserData: 0,
                Quartz.kCGKeyboardEventKeycode: 56,
                Quartz.kCGKeyboardEventAutorepeat: 0,
            }[field]

        @staticmethod
        def CGEventKeyboardGetUnicodeString(*_args):
            return 0, ""

        @staticmethod
        def CGEventGetFlags(_event):
            return Quartz.kCGEventFlagMaskShift | 0x00000002

    event = backend._normalize_event(
        Quartz.kCGEventFlagsChanged,
        object(),
        Quartz,
    )

    assert event.action is KeyAction.DOWN
    assert backend._modifier_keys_down == {56}
    backend.stop_capture()


@pytest.mark.parametrize(
    ("keycode", "usage"),
    [
        (56, 0xE1),
        (60, 0xE5),
        (59, 0xE0),
        (62, 0xE4),
        (58, 0xE2),
        (61, 0xE6),
        (55, 0xE3),
        (54, 0xE7),
    ],
)
def test_macos_first_release_of_preheld_modifier_is_up(keycode, usage):
    backend = MacOSInputBackend()

    class Quartz:
        kCGEventKeyDown = 1
        kCGEventKeyUp = 2
        kCGEventFlagsChanged = 3
        kCGEventSourceStateCombinedSessionState = 4
        kCGEventSourceUserData = 5
        kCGKeyboardEventKeycode = 6
        kCGKeyboardEventAutorepeat = 7
        kCGEventFlagMaskShift = 1 << 16
        kCGEventFlagMaskControl = 1 << 17
        kCGEventFlagMaskAlternate = 1 << 18
        kCGEventFlagMaskCommand = 1 << 19
        kCGEventFlagMaskAlphaShift = 1 << 20

        @staticmethod
        def CGEventSourceKeyState(_state, candidate):
            return candidate == keycode

        @staticmethod
        def CGEventGetIntegerValueField(_event, field):
            return {
                Quartz.kCGEventSourceUserData: 0,
                Quartz.kCGKeyboardEventKeycode: keycode,
                Quartz.kCGKeyboardEventAutorepeat: 0,
            }[field]

        @staticmethod
        def CGEventKeyboardGetUnicodeString(*_args):
            return 0, ""

        @staticmethod
        def CGEventGetFlags(_event):
            return 0

    backend._initialize_modifier_key_state(Quartz)
    event = backend._normalize_event(
        Quartz.kCGEventFlagsChanged,
        object(),
        Quartz,
    )

    assert event.action is KeyAction.UP
    assert event.usage == usage
    assert keycode not in backend._modifier_keys_down
