import ctypes
from types import SimpleNamespace

import pytest

from shooklink.input.events import KeyAction, KeyEvent, Modifiers
from shooklink.input.windows_backend import WindowsInputBackend, _Win32Api


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
        "GetKeyboardState",
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

        def key_text(self, virtual_key, scan_code):
            assert (virtual_key, scan_code) == (0xBF, 0x35)
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
