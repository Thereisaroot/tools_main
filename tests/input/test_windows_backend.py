import ctypes
import threading
from types import SimpleNamespace

import pytest

from shooklink.input.events import KeyAction, KeyEvent, Modifiers, PointerMotionEvent
from shooklink.input.windows_backend import (
    INJECTION_MARKER,
    WindowsInputBackend,
    _Win32Api,
)


class _FakeFunction:
    def __init__(self, result=1):
        self.result = result
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if callable(self.result):
            return self.result(*args)
        return self.result


class _FakeLibrary:
    def __init__(self):
        self.functions = {}

    def __getattr__(self, name):
        return self.functions.setdefault(name, _FakeFunction())


@pytest.fixture
def win32_api(monkeypatch):
    user32 = _FakeLibrary()
    kernel32 = _FakeLibrary()

    def load_library(name, **_kwargs):
        return user32 if name == "user32" else kernel32

    monkeypatch.setattr(ctypes, "WinDLL", load_library, raising=False)
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(
        ctypes,
        "WinError",
        lambda code: OSError(code, "Win32 call failed"),
        raising=False,
    )
    return _Win32Api(), user32, kernel32


def test_win32_api_defines_prototypes_for_every_native_function(win32_api):
    api, user32, kernel32 = win32_api
    user32_names = {
        "SetWindowsHookExW",
        "UnhookWindowsHookEx",
        "CallNextHookEx",
        "PostThreadMessageW",
        "PeekMessageW",
        "GetMessageW",
        "TranslateMessage",
        "DispatchMessageW",
        "GetAsyncKeyState",
        "GetKeyState",
        "ToUnicodeEx",
        "GetKeyboardLayout",
        "GetCursorPos",
        "SetCursorPos",
        "GetMonitorInfoW",
        "EnumDisplayMonitors",
        "GetSystemMetrics",
        "SendInput",
    }
    kernel32_names = {"GetModuleHandleW", "GetCurrentThreadId"}

    for name in user32_names:
        function = user32.functions[name]
        assert function.argtypes is not None, name
        assert function.restype is not None, name
    for name in kernel32_names:
        function = kernel32.functions[name]
        assert function.argtypes is not None, name
        assert function.restype is not None, name

    assert user32.CallNextHookEx.restype is ctypes.c_ssize_t
    assert user32.SetWindowsHookExW.restype is api.wintypes.HANDLE
    assert dict(api.KeyboardData._fields_)["dwExtraInfo"] is ctypes.c_size_t
    assert dict(api.MouseData._fields_)["dwExtraInfo"] is ctypes.c_size_t


def test_win32_api_checks_hook_message_and_unhook_failures(win32_api):
    api, user32, kernel32 = win32_api
    kernel32.GetModuleHandleW.result = 0
    with pytest.raises(OSError):
        api.install_hook(13, lambda *_args: 0)

    user32.UnhookWindowsHookEx.result = 0
    with pytest.raises(OSError):
        api.unhook(123)

    user32.PostThreadMessageW.result = 0
    with pytest.raises(OSError):
        api.post_quit(456)

    user32.GetMessageW.result = -1
    with pytest.raises(OSError):
        api.message_loop()


def test_win32_api_checks_monitor_enumeration_failure(win32_api):
    api, user32, _kernel32 = win32_api
    user32.EnumDisplayMonitors.result = 0

    with pytest.raises(OSError):
        api.enumerate_monitors()


def test_windows_keyboard_callback_preserves_translated_text():
    class CallbackBackend(WindowsInputBackend):
        def _start_native_capture(self, suppress):
            pass

        def _stop_native_capture(self):
            pass

    class KeyboardApi:
        def keyboard_data(self, _pointer):
            return SimpleNamespace(
                vkCode=0xBF,
                scanCode=0x35,
                flags=0,
                dwExtraInfo=0,
            )

        def key_text(self, virtual_key, scan_code, key_state):
            assert (virtual_key, scan_code) == (0xBF, 0x35)
            assert key_state[0xBF] & 0x80
            return "?"

        def call_next(self, *_args):
            return 77

    backend = CallbackBackend()
    backend._api = KeyboardApi()
    captured = []
    backend.start_capture(captured.append, lambda _action: None)

    result = backend._keyboard_callback(0, 0x0100, object())

    assert result == 77
    assert captured[0].usage == 0x38
    assert captured[0].text == "?"


def test_windows_text_translation_uses_updated_left_right_shift_state():
    class CallbackBackend(WindowsInputBackend):
        def _start_native_capture(self, suppress):
            pass

        def _stop_native_capture(self):
            pass

    class KeyboardApi:
        def __init__(self):
            self.states = []

        def keyboard_data(self, pointer):
            return pointer

        def key_text(self, virtual_key, _scan_code, key_state):
            self.states.append((virtual_key, key_state))
            if virtual_key == 0xBF:
                return "?" if key_state[0x10] & 0x80 else "/"
            return ""

        def call_next(self, *_args):
            return 77

    def data(virtual_key, scan_code):
        return SimpleNamespace(
            vkCode=virtual_key,
            scanCode=scan_code,
            flags=0,
            dwExtraInfo=0,
        )

    backend = CallbackBackend()
    api = KeyboardApi()
    backend._api = api
    captured = []
    backend.start_capture(captured.append, lambda _action: None)

    backend._keyboard_callback(0, 0x0100, data(0xA0, 0x2A))
    backend._keyboard_callback(0, 0x0100, data(0xA1, 0x36))
    backend._keyboard_callback(0, 0x0101, data(0xA0, 0x2A))
    backend._keyboard_callback(0, 0x0100, data(0xBF, 0x35))
    backend._keyboard_callback(0, 0x0101, data(0xBF, 0x35))
    backend._keyboard_callback(0, 0x0101, data(0xA1, 0x36))
    backend._keyboard_callback(0, 0x0100, data(0xBF, 0x35))

    slash_events = [event for event in captured if event.virtual_key == 0xBF]
    assert [event.text for event in slash_events if event.action is KeyAction.DOWN] == [
        "?",
        "/",
    ]
    shifted_state = next(state for key, state in api.states if key == 0xBF)
    assert shifted_state[0x10] & 0x80
    assert shifted_state[0xA0] & 0x80 == 0
    assert shifted_state[0xA1] & 0x80
    assert shifted_state[0xBF] & 0x80


def test_windows_text_translation_tracks_caps_lock_transitions():
    class CallbackBackend(WindowsInputBackend):
        def _start_native_capture(self, suppress):
            pass

        def _stop_native_capture(self):
            pass

    class KeyboardApi:
        def keyboard_data(self, pointer):
            return pointer

        def key_text(self, virtual_key, _scan_code, key_state):
            if virtual_key != 0x41:
                return ""
            shifted = bool(key_state[0x10] & 0x80)
            caps_locked = bool(key_state[0x14] & 0x01)
            return "A" if shifted ^ caps_locked else "a"

        def call_next(self, *_args):
            return 77

    def data(virtual_key, scan_code):
        return SimpleNamespace(
            vkCode=virtual_key,
            scanCode=scan_code,
            flags=0,
            dwExtraInfo=0,
        )

    backend = CallbackBackend()
    backend._api = KeyboardApi()
    captured = []
    backend.start_capture(captured.append, lambda _action: None)

    backend._keyboard_callback(0, 0x0100, data(0x14, 0x3A))
    backend._keyboard_callback(0, 0x0101, data(0x14, 0x3A))
    backend._keyboard_callback(0, 0x0100, data(0x41, 0x1E))
    backend._keyboard_callback(0, 0x0101, data(0x41, 0x1E))
    backend._keyboard_callback(0, 0x0100, data(0x14, 0x3A))
    backend._keyboard_callback(0, 0x0101, data(0x14, 0x3A))
    backend._keyboard_callback(0, 0x0100, data(0x41, 0x1E))

    letter_events = [
        event
        for event in captured
        if event.virtual_key == 0x41 and event.action is KeyAction.DOWN
    ]
    assert [event.text for event in letter_events] == ["A", "a"]
    assert letter_events[0].modifiers & Modifiers.CAPS_LOCK
    assert letter_events[1].modifiers & Modifiers.CAPS_LOCK == 0


def test_win32_key_text_uses_supplied_state_without_get_keyboard_state(win32_api):
    api, user32, _kernel32 = win32_api
    key_state = bytearray(256)
    key_state[0x10] = 0x80
    key_state[0xA0] = 0x80
    key_state[0xBF] = 0x80
    user32.GetKeyboardState.result = lambda *_args: pytest.fail(
        "hook-thread keyboard state must not be read"
    )
    user32.GetKeyboardLayout.result = 1

    def translate(_virtual_key, _scan_code, state, buffer, *_args):
        assert state[0x10] == 0x80
        assert state[0xA0] == 0x80
        assert state[0xBF] == 0x80
        buffer[0] = "?"
        return 1

    user32.ToUnicodeEx.result = translate

    assert api.key_text(0xBF, 0x35, key_state) == "?"
    assert user32.GetKeyboardState.calls == []


@pytest.mark.parametrize(
    ("suppress", "expected_result"),
    [(False, 77), (True, 1)],
)
def test_windows_third_party_injected_keyboard_follows_capture_policy(
    suppress,
    expected_result,
):
    class CallbackBackend(WindowsInputBackend):
        def _start_native_capture(self, suppress):
            pass

        def _stop_native_capture(self):
            pass

    class KeyboardApi:
        def keyboard_data(self, _pointer):
            return SimpleNamespace(
                vkCode=0x41,
                scanCode=0x1E,
                flags=0x10,
                dwExtraInfo=0,
            )

        def key_text(self, _virtual_key, _scan_code, _key_state):
            return "a"

        def call_next(self, *_args):
            return 77

    backend = CallbackBackend()
    backend._api = KeyboardApi()
    captured = []
    backend.start_capture(
        captured.append,
        lambda _action: None,
        suppress=suppress,
    )

    result = backend._keyboard_callback(0, 0x0100, object())

    assert result == expected_result
    assert len(captured) == 1
    assert captured[0].injected is False
    assert captured[0].text == "a"


def test_windows_private_marker_keyboard_is_self_filtered():
    class CallbackBackend(WindowsInputBackend):
        def _start_native_capture(self, suppress):
            pass

        def _stop_native_capture(self):
            pass

    class KeyboardApi:
        def keyboard_data(self, _pointer):
            return SimpleNamespace(
                vkCode=0x41,
                scanCode=0x1E,
                flags=0x10,
                dwExtraInfo=INJECTION_MARKER,
            )

        def key_text(self, _virtual_key, _scan_code, _key_state):
            return "a"

        def call_next(self, *_args):
            return 77

    backend = CallbackBackend()
    backend._api = KeyboardApi()
    captured = []
    backend.start_capture(captured.append, lambda _action: None)

    result = backend._keyboard_callback(0, 0x0100, object())

    assert result == 1
    assert captured == []


def test_windows_third_party_injected_mouse_follows_capture_policy():
    class CallbackBackend(WindowsInputBackend):
        def _start_native_capture(self, suppress):
            pass

        def _stop_native_capture(self):
            pass

    class MouseApi:
        def mouse_data(self, _pointer):
            return SimpleNamespace(
                pt=SimpleNamespace(x=10, y=20),
                mouseData=0,
                flags=0x01,
                dwExtraInfo=0,
            )

        def call_next(self, *_args):
            return 77

    backend = CallbackBackend()
    backend._api = MouseApi()
    captured = []
    backend.start_capture(captured.append, lambda _action: None)

    result = backend._mouse_callback(0, 0x0201, object())

    assert result == 77
    assert len(captured) == 1
    assert captured[0].injected is False


def test_windows_private_marker_mouse_is_self_filtered():
    class CallbackBackend(WindowsInputBackend):
        def _start_native_capture(self, suppress):
            pass

        def _stop_native_capture(self):
            pass

    class MouseApi:
        def mouse_data(self, _pointer):
            return SimpleNamespace(
                pt=SimpleNamespace(x=10, y=20),
                mouseData=0,
                flags=0x01,
                dwExtraInfo=INJECTION_MARKER,
            )

        def call_next(self, *_args):
            return 77

    backend = CallbackBackend()
    backend._api = MouseApi()
    captured = []
    backend.start_capture(captured.append, lambda _action: None)

    result = backend._mouse_callback(0, 0x0201, object())

    assert result == 1
    assert captured == []


def test_windows_modifier_aggregation_keeps_other_side_pressed():
    backend = WindowsInputBackend()

    backend._update_modifiers(0xE1, KeyAction.DOWN)
    backend._update_modifiers(0xE5, KeyAction.DOWN)
    modifiers = backend._update_modifiers(0xE1, KeyAction.UP)

    assert modifiers & Modifiers.SHIFT
    assert backend._update_modifiers(0xE5, KeyAction.UP) & Modifiers.SHIFT == 0


def test_windows_initial_modifier_state_includes_caps_and_num_lock():
    class KeyStateApi:
        def key_is_down(self, virtual_key):
            return virtual_key in {0xA0, 0xA3}

        def key_is_toggled(self, virtual_key):
            return virtual_key in {0x14, 0x90}

    backend = WindowsInputBackend()

    backend._initialize_modifier_state(KeyStateApi())

    assert backend._modifiers == (
        Modifiers.SHIFT
        | Modifiers.CONTROL
        | Modifiers.CAPS_LOCK
        | Modifiers.NUM_LOCK
    )


@pytest.mark.parametrize(
    ("event", "scan_code", "expected_flags"),
    [
        (KeyEvent(KeyAction.DOWN, usage=0x38), 0x35, 0x0008),
        (KeyEvent(KeyAction.UP, usage=0xE4), 0x1D, 0x000B),
        (
            KeyEvent(KeyAction.DOWN, usage=0, scan_code=0x2C, virtual_key=0),
            0x2C,
            0x0008,
        ),
    ],
)
def test_windows_key_injection_prefers_physical_scan_codes(
    event,
    scan_code,
    expected_flags,
):
    class KeyboardInput:
        def __init__(self, virtual_key, scan, flags, time, extra_info):
            self.wVk = virtual_key
            self.wScan = scan
            self.dwFlags = flags
            self.time = time
            self.dwExtraInfo = extra_info

    class Input:
        def __init__(self, **values):
            vars(self).update(values)

    api = object.__new__(_Win32Api)
    api.KeyboardInput = KeyboardInput
    api.Input = Input
    sent = []
    api._send_inputs = sent.extend

    api.inject(event)

    keyboard = sent[0].ki
    assert keyboard.wVk == 0
    assert keyboard.wScan == scan_code
    assert keyboard.dwFlags == expected_flags


def test_windows_stop_preserves_thread_reference_when_thread_does_not_exit():
    class StuckThread:
        def is_alive(self):
            return True

        def join(self, _timeout):
            pass

    class StopApi:
        def __init__(self):
            self.quit_requests = []

        def post_quit(self, thread_id):
            self.quit_requests.append(thread_id)

    backend = WindowsInputBackend()
    thread = StuckThread()
    backend._api = StopApi()
    backend._hook_thread = thread
    backend._hook_thread_id = 99

    with pytest.raises(RuntimeError, match="did not stop"):
        backend._stop_native_capture()

    assert backend._hook_thread is thread
    assert backend._hook_thread_id == 99


def test_windows_stop_preserves_state_when_quit_request_fails():
    class LiveThread:
        def is_alive(self):
            return True

    class StopApi:
        def post_quit(self, _thread_id):
            raise OSError("post failed")

    backend = WindowsInputBackend()
    thread = LiveThread()
    backend._api = StopApi()
    backend._hook_thread = thread
    backend._hook_thread_id = 99

    with pytest.raises(RuntimeError, match="could not request"):
        backend._stop_native_capture()

    assert backend._hook_thread is thread
    assert backend._hook_thread_id == 99


def test_windows_stop_retries_failed_hook_cleanup_without_dropping_callbacks():
    class DeadThread:
        def is_alive(self):
            return False

    class CleanupApi:
        def __init__(self):
            self.fail_keyboard = True
            self.unhooked = []

        def unhook(self, hook):
            if hook == "keyboard" and self.fail_keyboard:
                raise OSError("unhook failed")
            self.unhooked.append(hook)

    backend = WindowsInputBackend()
    api = CleanupApi()
    thread = DeadThread()
    keyboard_callback = object()
    mouse_callback = object()
    backend._api = api
    backend._hook_thread = thread
    backend._hook_thread_id = 99
    backend._keyboard_hook = "keyboard"
    backend._mouse_hook = "mouse"
    backend._keyboard_callback_ref = keyboard_callback
    backend._mouse_callback_ref = mouse_callback

    with pytest.raises(RuntimeError, match="could not remove Windows input hooks"):
        backend._stop_native_capture()

    assert backend._hook_thread is thread
    assert backend._hook_thread_id == 99
    assert backend._keyboard_hook == "keyboard"
    assert backend._keyboard_callback_ref is keyboard_callback
    assert backend._mouse_hook is None
    assert backend._mouse_callback_ref is None

    api.fail_keyboard = False
    backend._stop_native_capture()

    assert api.unhooked == ["mouse", "keyboard"]
    assert backend._hook_thread is None
    assert backend._hook_thread_id is None
    assert backend._keyboard_hook is None
    assert backend._keyboard_callback_ref is None


def test_windows_callback_thread_restart_isolated_from_stale_cleanup(monkeypatch):
    backend = WindowsInputBackend()
    monkeypatch.setattr("shooklink.input.windows_backend.sys.platform", "win32")

    real_thread = threading.Thread
    threads = []

    def thread_factory(**kwargs):
        thread = real_thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr("shooklink.input.windows_backend.threading.Thread", thread_factory)

    class HookApi:
        def __init__(self):
            self.thread_state = threading.local()
            self.next_thread_id = 1
            self.next_hook_id = 1
            self.quit_events = {}
            self.unhooked = []
            self.unhook_attempts = []
            self.failed_old_cleanup = False
            self.restart_complete = threading.Event()
            self.release_old_cleanup = threading.Event()
            self.errors = []

        def current_thread_id(self):
            thread_id = self.next_thread_id
            self.next_thread_id += 1
            self.thread_state.thread_id = thread_id
            self.quit_events[thread_id] = threading.Event()
            return thread_id

        def ensure_message_queue(self):
            pass

        def key_is_down(self, _virtual_key):
            return False

        def key_is_toggled(self, _virtual_key):
            return False

        def hook_proc(self, callback):
            return callback

        def install_hook(self, _hook_type, _callback):
            hook = f"hook-{self.next_hook_id}"
            self.next_hook_id += 1
            return hook

        def post_quit(self, thread_id):
            self.quit_events[thread_id].set()

        def message_loop(self):
            thread_id = self.thread_state.thread_id
            if thread_id == 1:
                try:
                    backend.stop_capture()
                    backend.start_capture(lambda _event: None, lambda _action: None)
                except BaseException as error:
                    self.errors.append(error)
                finally:
                    self.restart_complete.set()
                assert self.release_old_cleanup.wait(2)
                return
            assert self.quit_events[thread_id].wait(2)

        def unhook(self, hook):
            self.unhook_attempts.append(hook)
            if hook == "hook-1" and not self.failed_old_cleanup:
                self.failed_old_cleanup = True
                raise OSError("old hook cleanup failed")
            self.unhooked.append(hook)

    api = HookApi()
    backend._api = api

    try:
        backend.start_capture(lambda _event: None, lambda _action: None)
        assert api.restart_complete.wait(2)
        assert api.errors == []
        assert len(threads) == 2

        new_thread = threads[1]
        new_keyboard_hook = backend._keyboard_hook
        new_mouse_hook = backend._mouse_hook
        new_keyboard_callback = backend._keyboard_callback_ref
        new_mouse_callback = backend._mouse_callback_ref
        assert new_keyboard_hook == "hook-3"
        assert new_mouse_hook == "hook-4"

        api.release_old_cleanup.set()
        threads[0].join(2)

        assert not threads[0].is_alive()
        assert backend._hook_thread is new_thread
        assert backend._keyboard_hook == new_keyboard_hook
        assert backend._mouse_hook == new_mouse_hook
        assert backend._keyboard_callback_ref is new_keyboard_callback
        assert backend._mouse_callback_ref is new_mouse_callback
        assert "hook-1" in api.unhook_attempts
        assert "hook-1" not in api.unhooked
        assert "hook-2" in api.unhooked
        assert new_keyboard_hook not in api.unhooked
        assert new_mouse_hook not in api.unhooked

        backend.stop_capture()
        assert "hook-1" in api.unhooked
    finally:
        api.release_old_cleanup.set()
        for event in api.quit_events.values():
            event.set()
        for thread in threads:
            thread.join(2)


def test_windows_new_generation_resets_capture_local_state(monkeypatch):
    backend = WindowsInputBackend()
    backend._modifier_keys_down = {0xE1}
    backend._lock_modifiers = Modifiers.CAPS_LOCK | Modifiers.NUM_LOCK
    backend._modifiers = Modifiers.SHIFT | backend._lock_modifiers
    backend._captured_keys_down = {0x04}
    backend._last_mouse_position = (100, 200)
    monkeypatch.setattr("shooklink.input.windows_backend.sys.platform", "win32")

    class ImmediateThread:
        def __init__(self, *, args, **_kwargs):
            self.generation = args[0]

        def start(self):
            self.generation.ready.set()

        def is_alive(self):
            return False

    class CaptureApi:
        def keyboard_data(self, pointer):
            return pointer

        def mouse_data(self, pointer):
            return pointer

        def key_text(self, _virtual_key, _scan_code, _key_state):
            return "a"

        def call_next(self, *_args):
            return 77

    monkeypatch.setattr(
        "shooklink.input.windows_backend.threading.Thread",
        ImmediateThread,
    )
    backend._api = CaptureApi()
    captured = []
    backend.start_capture(captured.append, lambda _action: None)

    backend._keyboard_callback(
        0,
        0x0100,
        SimpleNamespace(
            vkCode=0x41,
            scanCode=0x1E,
            flags=0,
            dwExtraInfo=0,
        ),
    )
    backend._mouse_callback(
        0,
        0x0200,
        SimpleNamespace(
            pt=SimpleNamespace(x=110, y=210),
            mouseData=0,
            flags=0,
            dwExtraInfo=0,
        ),
    )

    key_event = next(event for event in captured if isinstance(event, KeyEvent))
    motion_event = next(
        event for event in captured if isinstance(event, PointerMotionEvent)
    )
    assert key_event.repeat is False
    assert key_event.modifiers is Modifiers.NONE
    assert (motion_event.dx, motion_event.dy) == (0, 0)
    backend.stop_capture()
