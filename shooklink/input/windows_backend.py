"""Low-level Win32 hooks and SendInput injection for Windows."""

from __future__ import annotations

import sys
import threading

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
from shooklink.input.topology import Monitor, Rect

INJECTION_MARKER = 0x53484F4F4B4C4E4B

_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008

_MODIFIER_USAGE_TO_VK = {
    0xE0: 0xA2,
    0xE1: 0xA0,
    0xE2: 0xA4,
    0xE3: 0x5B,
    0xE4: 0xA3,
    0xE5: 0xA1,
    0xE6: 0xA5,
    0xE7: 0x5C,
}
_MODIFIER_GROUPS = {
    Modifiers.CONTROL: frozenset({0xE0, 0xE4}),
    Modifiers.SHIFT: frozenset({0xE1, 0xE5}),
    Modifiers.ALT: frozenset({0xE2, 0xE6}),
    Modifiers.META: frozenset({0xE3, 0xE7}),
}

_VK_TO_USAGE = {
    **{0x41 + index: 0x04 + index for index in range(26)},
    **{0x31 + index: 0x1E + index for index in range(9)},
    0x30: 0x27,
    0x0D: 0x28,
    0x1B: 0x29,
    0x08: 0x2A,
    0x09: 0x2B,
    0x20: 0x2C,
    0xBD: 0x2D,
    0xBB: 0x2E,
    0xDB: 0x2F,
    0xDD: 0x30,
    0xDC: 0x31,
    0xBA: 0x33,
    0xDE: 0x34,
    0xC0: 0x35,
    0xBC: 0x36,
    0xBE: 0x37,
    0xBF: 0x38,
    0x14: 0x39,
    **{0x70 + index: 0x3A + index for index in range(12)},
    0x2C: 0x46,
    0x91: 0x47,
    0x13: 0x48,
    0x2D: 0x49,
    0x24: 0x4A,
    0x21: 0x4B,
    0x2E: 0x4C,
    0x23: 0x4D,
    0x22: 0x4E,
    0x27: 0x4F,
    0x25: 0x50,
    0x28: 0x51,
    0x26: 0x52,
    0x90: 0x53,
    0x6F: 0x54,
    0x6A: 0x55,
    0x6D: 0x56,
    0x6B: 0x57,
    0x61: 0x59,
    0x62: 0x5A,
    0x63: 0x5B,
    0x64: 0x5C,
    0x65: 0x5D,
    0x66: 0x5E,
    0x67: 0x5F,
    0x68: 0x60,
    0x69: 0x61,
    0x60: 0x62,
    0x6E: 0x63,
    0xA2: 0xE0,
    0xA0: 0xE1,
    0xA4: 0xE2,
    0x5B: 0xE3,
    0xA3: 0xE4,
    0xA1: 0xE5,
    0xA5: 0xE6,
    0x5C: 0xE7,
}
_USAGE_TO_VK = {usage: virtual_key for virtual_key, usage in _VK_TO_USAGE.items()}
_USAGE_TO_VK[0x58] = 0x0D

_USAGE_TO_SCAN = {
    0x04: (0x1E, False),
    0x05: (0x30, False),
    0x06: (0x2E, False),
    0x07: (0x20, False),
    0x08: (0x12, False),
    0x09: (0x21, False),
    0x0A: (0x22, False),
    0x0B: (0x23, False),
    0x0C: (0x17, False),
    0x0D: (0x24, False),
    0x0E: (0x25, False),
    0x0F: (0x26, False),
    0x10: (0x32, False),
    0x11: (0x31, False),
    0x12: (0x18, False),
    0x13: (0x19, False),
    0x14: (0x10, False),
    0x15: (0x13, False),
    0x16: (0x1F, False),
    0x17: (0x14, False),
    0x18: (0x16, False),
    0x19: (0x2F, False),
    0x1A: (0x11, False),
    0x1B: (0x2D, False),
    0x1C: (0x15, False),
    0x1D: (0x2C, False),
    0x1E: (0x02, False),
    0x1F: (0x03, False),
    0x20: (0x04, False),
    0x21: (0x05, False),
    0x22: (0x06, False),
    0x23: (0x07, False),
    0x24: (0x08, False),
    0x25: (0x09, False),
    0x26: (0x0A, False),
    0x27: (0x0B, False),
    0x28: (0x1C, False),
    0x29: (0x01, False),
    0x2A: (0x0E, False),
    0x2B: (0x0F, False),
    0x2C: (0x39, False),
    0x2D: (0x0C, False),
    0x2E: (0x0D, False),
    0x2F: (0x1A, False),
    0x30: (0x1B, False),
    0x31: (0x2B, False),
    0x33: (0x27, False),
    0x34: (0x28, False),
    0x35: (0x29, False),
    0x36: (0x33, False),
    0x37: (0x34, False),
    0x38: (0x35, False),
    0x39: (0x3A, False),
    **{0x3A + index: (0x3B + index, False) for index in range(10)},
    0x44: (0x57, False),
    0x45: (0x58, False),
    0x46: (0x37, True),
    0x47: (0x46, False),
    0x48: (0x45, False),
    0x49: (0x52, True),
    0x4A: (0x47, True),
    0x4B: (0x49, True),
    0x4C: (0x53, True),
    0x4D: (0x4F, True),
    0x4E: (0x51, True),
    0x4F: (0x4D, True),
    0x50: (0x4B, True),
    0x51: (0x50, True),
    0x52: (0x48, True),
    0x53: (0x45, True),
    0x54: (0x35, True),
    0x55: (0x37, False),
    0x56: (0x4A, False),
    0x57: (0x4E, False),
    0x58: (0x1C, True),
    0x59: (0x4F, False),
    0x5A: (0x50, False),
    0x5B: (0x51, False),
    0x5C: (0x4B, False),
    0x5D: (0x4C, False),
    0x5E: (0x4D, False),
    0x5F: (0x47, False),
    0x60: (0x48, False),
    0x61: (0x49, False),
    0x62: (0x52, False),
    0x63: (0x53, False),
    0x64: (0x56, False),
    0x65: (0x5D, True),
    0x67: (0x59, False),
    0xE0: (0x1D, False),
    0xE1: (0x2A, False),
    0xE2: (0x38, False),
    0xE3: (0x5B, True),
    0xE4: (0x1D, True),
    0xE5: (0x36, False),
    0xE6: (0x38, True),
    0xE7: (0x5C, True),
}
_SCAN_TO_USAGE = {scan: usage for usage, scan in _USAGE_TO_SCAN.items()}


def windows_key_to_usage(virtual_key: int, scan_code: int, extended: bool) -> int:
    if any(type(value) is not int for value in (virtual_key, scan_code)):
        raise TypeError("Windows key values must be integers")
    usage = _SCAN_TO_USAGE.get((scan_code, bool(extended)))
    if usage is not None:
        return usage
    if virtual_key == 0x0D and extended:
        return 0x58
    return _VK_TO_USAGE.get(virtual_key, 0)


class WindowsInputBackend(BaseInputBackend):
    def __init__(self) -> None:
        super().__init__()
        self._api = None
        self._hook_thread: threading.Thread | None = None
        self._hook_ready = threading.Event()
        self._hook_error: BaseException | None = None
        self._hook_thread_id: int | None = None
        self._keyboard_hook = None
        self._mouse_hook = None
        self._keyboard_callback_ref = None
        self._mouse_callback_ref = None
        self._modifiers = Modifiers.NONE
        self._modifier_keys_down: set[int] = set()
        self._lock_modifiers = Modifiers.NONE
        self._captured_keys_down: set[int] = set()
        self._last_mouse_position: tuple[int, int] | None = None

    def permission_status(self) -> PermissionStatus:
        if sys.platform != "win32":
            return PermissionStatus(False, False, "Windows only")
        return PermissionStatus(True, True, "ready")

    def monitors(self) -> tuple[Monitor, ...]:
        if sys.platform != "win32":
            return ()
        return self._get_api().enumerate_monitors()

    def cursor_position(self) -> tuple[int, int]:
        if sys.platform != "win32":
            raise OSError("Windows only")
        return self._get_api().cursor_position()

    def warp_cursor(self, x: int, y: int) -> None:
        if sys.platform != "win32":
            raise OSError("Windows only")
        self._get_api().set_cursor_position(x, y)

    def _start_native_capture(self, suppress: bool) -> None:
        if sys.platform != "win32":
            raise OSError("Windows input capture is only available on Windows")
        self._hook_ready.clear()
        self._hook_error = None
        self._hook_thread = threading.Thread(
            target=self._hook_loop,
            name="shooklink-win32-input-hooks",
            daemon=True,
        )
        self._hook_thread.start()
        if not self._hook_ready.wait(3):
            raise RuntimeError("Windows input hooks did not start")
        if self._hook_error is not None:
            raise RuntimeError("Windows input hooks failed") from self._hook_error

    def _stop_native_capture(self) -> None:
        thread = self._hook_thread
        if thread is None:
            self._cleanup_hooks()
            self._hook_thread_id = None
            return
        if not thread.is_alive():
            self._finish_native_stop()
            return
        if self._hook_thread_id is None:
            raise RuntimeError("Windows hook thread has no message-loop identifier")
        try:
            self._get_api().post_quit(self._hook_thread_id)
        except Exception as error:
            if thread.is_alive():
                raise RuntimeError(
                    "could not request Windows hook thread shutdown"
                ) from error
        if thread is threading.current_thread():
            return
        thread.join(3)
        if thread.is_alive():
            raise RuntimeError("Windows hook thread did not stop within 3 seconds")
        self._finish_native_stop()

    def _finish_native_stop(self) -> None:
        self._cleanup_hooks()
        self._hook_thread = None
        self._hook_thread_id = None

    def _cleanup_hooks(self) -> None:
        failures = []
        for hook_name, callback_name in (
            ("_keyboard_hook", "_keyboard_callback_ref"),
            ("_mouse_hook", "_mouse_callback_ref"),
        ):
            hook = getattr(self, hook_name)
            if hook is None:
                setattr(self, callback_name, None)
                continue
            try:
                self._get_api().unhook(hook)
            except Exception as error:
                failures.append(error)
            else:
                setattr(self, hook_name, None)
                setattr(self, callback_name, None)
        if failures:
            raise RuntimeError("could not remove Windows input hooks") from failures[0]

    def _hook_loop(self) -> None:
        try:
            api = self._get_api()
            self._hook_thread_id = api.current_thread_id()
            api.ensure_message_queue()
            self._initialize_modifier_state(api)
            self._keyboard_callback_ref = api.hook_proc(self._keyboard_callback)
            self._mouse_callback_ref = api.hook_proc(self._mouse_callback)
            self._keyboard_hook = api.install_hook(13, self._keyboard_callback_ref)
            self._mouse_hook = api.install_hook(14, self._mouse_callback_ref)
            self._hook_ready.set()
            api.message_loop()
        except BaseException as error:
            self._hook_error = error
            self._hook_ready.set()
        finally:
            try:
                self._cleanup_hooks()
            except BaseException as error:
                if self._hook_error is None:
                    self._hook_error = error

    def _keyboard_callback(self, code, message, pointer):
        api = self._get_api()
        if code < 0:
            return api.call_next(self._keyboard_hook, code, message, pointer)
        data = api.keyboard_data(pointer)
        action = (
            KeyAction.DOWN
            if message in (0x0100, 0x0104)
            else KeyAction.UP
        )
        extended = bool(data.flags & 0x01)
        usage = windows_key_to_usage(data.vkCode, data.scanCode, extended)
        repeat = action is KeyAction.DOWN and usage in self._captured_keys_down
        modifiers = self._update_modifiers(usage, action, repeat=repeat)
        if action is KeyAction.DOWN:
            self._captured_keys_down.add(usage)
        else:
            self._captured_keys_down.discard(usage)
        text = (
            api.key_text(int(data.vkCode), int(data.scanCode))
            if action is KeyAction.DOWN
            else ""
        )
        event = KeyEvent(
            action,
            usage,
            scan_code=int(data.scanCode),
            virtual_key=int(data.vkCode),
            text=text,
            modifiers=modifiers,
            location=_usage_location(usage),
            repeat=repeat,
            extended=extended,
            injected=bool(data.flags & 0x10) or int(data.dwExtraInfo) == INJECTION_MARKER,
        )
        consumed = self.emit_captured(event)
        if consumed or self._capture_suppress:
            return 1
        return api.call_next(self._keyboard_hook, code, message, pointer)

    def _mouse_callback(self, code, message, pointer):
        api = self._get_api()
        if code < 0:
            return api.call_next(self._mouse_hook, code, message, pointer)
        data = api.mouse_data(pointer)
        injected = bool(data.flags & 0x01) or int(data.dwExtraInfo) == INJECTION_MARKER
        event: InputEvent | None = None
        position = (int(data.pt.x), int(data.pt.y))
        if message == 0x0200:
            previous = self._last_mouse_position or position
            self._last_mouse_position = position
            event = PointerMotionEvent(
                position[0] - previous[0],
                position[1] - previous[1],
                injected,
            )
        elif message in _WINDOWS_BUTTON_MESSAGES:
            button, action = _WINDOWS_BUTTON_MESSAGES[message]
            if message in (0x020B, 0x020C):
                high = (int(data.mouseData) >> 16) & 0xFFFF
                button = MouseButton.X1 if high == 1 else MouseButton.X2
            event = MouseButtonEvent(button, action, injected)
        elif message in (0x020A, 0x020E):
            delta = _signed_word((int(data.mouseData) >> 16) & 0xFFFF)
            event = WheelEvent(
                delta if message == 0x020E else 0,
                delta if message == 0x020A else 0,
                injected,
            )
        if event is not None:
            consumed = self.emit_captured(event)
            if consumed or self._capture_suppress:
                return 1
        return api.call_next(self._mouse_hook, code, message, pointer)

    def _initialize_modifier_state(self, api) -> None:
        self._modifier_keys_down = {
            usage
            for usage, virtual_key in _MODIFIER_USAGE_TO_VK.items()
            if api.key_is_down(virtual_key)
        }
        self._lock_modifiers = Modifiers.NONE
        if api.key_is_toggled(0x14):
            self._lock_modifiers |= Modifiers.CAPS_LOCK
        if api.key_is_toggled(0x90):
            self._lock_modifiers |= Modifiers.NUM_LOCK
        self._rebuild_modifiers()

    def _update_modifiers(
        self,
        usage: int,
        action: KeyAction,
        *,
        repeat: bool = False,
    ) -> Modifiers:
        if usage in _MODIFIER_USAGE_TO_VK:
            if action is KeyAction.DOWN:
                self._modifier_keys_down.add(usage)
            else:
                self._modifier_keys_down.discard(usage)
        if action is KeyAction.DOWN and not repeat:
            if usage == 0x39:
                self._lock_modifiers ^= Modifiers.CAPS_LOCK
            elif usage == 0x53:
                self._lock_modifiers ^= Modifiers.NUM_LOCK
        self._rebuild_modifiers()
        return self._modifiers

    def _rebuild_modifiers(self) -> None:
        modifiers = self._lock_modifiers
        for modifier, usages in _MODIFIER_GROUPS.items():
            if self._modifier_keys_down & usages:
                modifiers |= modifier
        self._modifiers = modifiers

    def _inject_native(self, event: InputEvent) -> None:
        if sys.platform != "win32":
            raise OSError("Windows input injection is only available on Windows")
        self._get_api().inject(event)

    def _get_api(self):
        if self._api is None:
            self._api = _Win32Api()
        return self._api


_WINDOWS_BUTTON_MESSAGES = {
    0x0201: (MouseButton.LEFT, KeyAction.DOWN),
    0x0202: (MouseButton.LEFT, KeyAction.UP),
    0x0204: (MouseButton.RIGHT, KeyAction.DOWN),
    0x0205: (MouseButton.RIGHT, KeyAction.UP),
    0x0207: (MouseButton.MIDDLE, KeyAction.DOWN),
    0x0208: (MouseButton.MIDDLE, KeyAction.UP),
    0x020B: (MouseButton.X1, KeyAction.DOWN),
    0x020C: (MouseButton.X1, KeyAction.UP),
}


class _Win32Api:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        if not hasattr(ctypes, "WinDLL"):
            raise OSError("Win32 APIs are unavailable")
        self.ctypes = ctypes
        self.wintypes = wintypes
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class Point(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        class KeyboardData(ctypes.Structure):
            _fields_ = [
                ("vkCode", wintypes.DWORD),
                ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class MouseData(ctypes.Structure):
            _fields_ = [
                ("pt", Point),
                ("mouseData", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class MouseInput(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class KeyboardInput(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        class InputUnion(ctypes.Union):
            _fields_ = [("mi", MouseInput), ("ki", KeyboardInput)]

        class Input(ctypes.Structure):
            _anonymous_ = ("union",)
            _fields_ = [("type", wintypes.DWORD), ("union", InputUnion)]

        class Message(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("message", wintypes.UINT),
                ("wParam", wintypes.WPARAM),
                ("lParam", wintypes.LPARAM),
                ("time", wintypes.DWORD),
                ("pt", Point),
                ("lPrivate", wintypes.DWORD),
            ]

        class MonitorInfo(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
            ]

        self.Point = Point
        self.KeyboardData = KeyboardData
        self.MouseData = MouseData
        self.MouseInput = MouseInput
        self.KeyboardInput = KeyboardInput
        self.Input = Input
        self.Message = Message
        self.MonitorInfo = MonitorInfo
        self.HookProc = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        self.MonitorEnumProc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HMONITOR,
            wintypes.HDC,
            ctypes.POINTER(wintypes.RECT),
            wintypes.LPARAM,
        )
        self._configure_functions()

    def _configure_functions(self) -> None:
        ctypes = self.ctypes
        wintypes = self.wintypes

        self.user32.SetWindowsHookExW.argtypes = [
            ctypes.c_int,
            self.HookProc,
            wintypes.HINSTANCE,
            wintypes.DWORD,
        ]
        self.user32.SetWindowsHookExW.restype = wintypes.HHOOK
        self.user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
        self.user32.UnhookWindowsHookEx.restype = wintypes.BOOL
        self.user32.CallNextHookEx.argtypes = [
            wintypes.HHOOK,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self.user32.CallNextHookEx.restype = ctypes.c_ssize_t
        self.user32.PostThreadMessageW.argtypes = [
            wintypes.DWORD,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self.user32.PostThreadMessageW.restype = wintypes.BOOL
        self.user32.PeekMessageW.argtypes = [
            ctypes.POINTER(self.Message),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self.user32.PeekMessageW.restype = wintypes.BOOL
        self.user32.GetMessageW.argtypes = [
            ctypes.POINTER(self.Message),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self.user32.GetMessageW.restype = wintypes.BOOL
        self.user32.TranslateMessage.argtypes = [ctypes.POINTER(self.Message)]
        self.user32.TranslateMessage.restype = wintypes.BOOL
        self.user32.DispatchMessageW.argtypes = [ctypes.POINTER(self.Message)]
        self.user32.DispatchMessageW.restype = ctypes.c_ssize_t
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = wintypes.SHORT
        self.user32.GetKeyState.argtypes = [ctypes.c_int]
        self.user32.GetKeyState.restype = wintypes.SHORT
        self.user32.GetKeyboardState.argtypes = [ctypes.POINTER(wintypes.BYTE)]
        self.user32.GetKeyboardState.restype = wintypes.BOOL
        self.user32.ToUnicodeEx.argtypes = [
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(wintypes.BYTE),
            wintypes.LPWSTR,
            ctypes.c_int,
            wintypes.UINT,
            wintypes.HKL,
        ]
        self.user32.ToUnicodeEx.restype = ctypes.c_int
        self.user32.GetKeyboardLayout.argtypes = [wintypes.DWORD]
        self.user32.GetKeyboardLayout.restype = wintypes.HKL
        self.user32.GetCursorPos.argtypes = [ctypes.POINTER(self.Point)]
        self.user32.GetCursorPos.restype = wintypes.BOOL
        self.user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        self.user32.SetCursorPos.restype = wintypes.BOOL
        self.user32.GetMonitorInfoW.argtypes = [
            wintypes.HMONITOR,
            ctypes.POINTER(self.MonitorInfo),
        ]
        self.user32.GetMonitorInfoW.restype = wintypes.BOOL
        self.user32.EnumDisplayMonitors.argtypes = [
            wintypes.HDC,
            ctypes.POINTER(wintypes.RECT),
            self.MonitorEnumProc,
            wintypes.LPARAM,
        ]
        self.user32.EnumDisplayMonitors.restype = wintypes.BOOL
        self.user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        self.user32.GetSystemMetrics.restype = ctypes.c_int
        self.user32.SendInput.argtypes = [
            wintypes.UINT,
            ctypes.POINTER(self.Input),
            ctypes.c_int,
        ]
        self.user32.SendInput.restype = wintypes.UINT

        self.kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self.kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        self.kernel32.GetCurrentThreadId.argtypes = []
        self.kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    def hook_proc(self, callback):
        return self.HookProc(callback)

    def install_hook(self, hook_type, callback):
        module = self.kernel32.GetModuleHandleW(None)
        if not module:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        hook = self.user32.SetWindowsHookExW(
            hook_type,
            callback,
            module,
            0,
        )
        if not hook:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return hook

    def unhook(self, hook) -> None:
        if hook and not self.user32.UnhookWindowsHookEx(hook):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def call_next(self, hook, code, message, pointer):
        return self.user32.CallNextHookEx(hook, code, message, pointer)

    def keyboard_data(self, pointer):
        return self.ctypes.cast(pointer, self.ctypes.POINTER(self.KeyboardData)).contents

    def mouse_data(self, pointer):
        return self.ctypes.cast(pointer, self.ctypes.POINTER(self.MouseData)).contents

    def current_thread_id(self) -> int:
        return int(self.kernel32.GetCurrentThreadId())

    def post_quit(self, thread_id: int) -> None:
        if not self.user32.PostThreadMessageW(thread_id, 0x0012, 0, 0):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def ensure_message_queue(self) -> None:
        message = self.Message()
        self.user32.PeekMessageW(self.ctypes.byref(message), None, 0, 0, 0)

    def message_loop(self) -> None:
        message = self.Message()
        while True:
            result = self.user32.GetMessageW(
                self.ctypes.byref(message),
                None,
                0,
                0,
            )
            if result == -1:
                raise self.ctypes.WinError(self.ctypes.get_last_error())
            if result == 0:
                return
            self.user32.TranslateMessage(self.ctypes.byref(message))
            self.user32.DispatchMessageW(self.ctypes.byref(message))

    def key_is_down(self, virtual_key: int) -> bool:
        return bool(self.user32.GetAsyncKeyState(virtual_key) & 0x8000)

    def key_is_toggled(self, virtual_key: int) -> bool:
        return bool(self.user32.GetKeyState(virtual_key) & 0x0001)

    def key_text(self, virtual_key: int, scan_code: int) -> str:
        state = (self.wintypes.BYTE * 256)()
        if not self.user32.GetKeyboardState(state):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        state[virtual_key & 0xFF] |= 0x80
        layout = self.user32.GetKeyboardLayout(0)
        if not layout:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        buffer = self.ctypes.create_unicode_buffer(32)
        length = self.user32.ToUnicodeEx(
            virtual_key,
            scan_code,
            state,
            buffer,
            len(buffer),
            0x0004,
            layout,
        )
        if length == 0:
            return ""
        return buffer[: min(abs(length), len(buffer) - 1)]

    def cursor_position(self) -> tuple[int, int]:
        point = self.Point()
        if not self.user32.GetCursorPos(self.ctypes.byref(point)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return int(point.x), int(point.y)

    def set_cursor_position(self, x: int, y: int) -> None:
        if not self.user32.SetCursorPos(x, y):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def enumerate_monitors(self) -> tuple[Monitor, ...]:
        monitors = []
        callback_error = None

        @self.MonitorEnumProc
        def callback(handle, _device, _rect, _data):
            nonlocal callback_error
            try:
                info = self.MonitorInfo()
                info.cbSize = self.ctypes.sizeof(info)
                if not self.user32.GetMonitorInfoW(handle, self.ctypes.byref(info)):
                    raise self.ctypes.WinError(self.ctypes.get_last_error())
                rect = info.rcMonitor
                raw_handle = getattr(handle, "value", handle)
                monitors.append(
                    Monitor(
                        str(int(raw_handle or 0)),
                        Rect(
                            int(rect.left),
                            int(rect.top),
                            int(rect.right - rect.left),
                            int(rect.bottom - rect.top),
                        ),
                    )
                )
                return True
            except BaseException as error:
                callback_error = error
                return False

        success = self.user32.EnumDisplayMonitors(None, None, callback, 0)
        if callback_error is not None:
            raise callback_error
        if not success:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return tuple(monitors)

    def inject(self, event: InputEvent) -> None:
        if isinstance(event, KeyEvent):
            scan = _USAGE_TO_SCAN.get(event.usage)
            if scan is None and event.scan_code:
                scan = (event.scan_code, event.extended)
            flags = _KEYEVENTF_KEYUP if event.action is KeyAction.UP else 0
            if scan is not None:
                scan_code, extended = scan
                if scan_code & 0xFF00 == 0xE000:
                    scan_code &= 0xFF
                    extended = True
                flags |= _KEYEVENTF_SCANCODE
                if extended:
                    flags |= _KEYEVENTF_EXTENDEDKEY
                virtual_key = 0
            else:
                scan_code = 0
                virtual_key = _USAGE_TO_VK.get(event.usage, event.virtual_key)
                if not virtual_key:
                    raise ValueError(f"unsupported Windows HID usage {event.usage}")
                if event.extended:
                    flags |= _KEYEVENTF_EXTENDEDKEY
            value = self.Input(
                type=1,
                ki=self.KeyboardInput(
                    virtual_key,
                    scan_code,
                    flags,
                    0,
                    INJECTION_MARKER,
                ),
            )
            self._send_inputs([value])
            return
        if isinstance(event, PointerPositionEvent):
            left = self.user32.GetSystemMetrics(76)
            top = self.user32.GetSystemMetrics(77)
            screen_width = self.user32.GetSystemMetrics(78)
            screen_height = self.user32.GetSystemMetrics(79)
            if screen_width <= 0 or screen_height <= 0:
                raise OSError("could not query Windows virtual-screen dimensions")
            width = max(1, screen_width - 1)
            height = max(1, screen_height - 1)
            x = round((event.x - left) * 65535 / width)
            y = round((event.y - top) * 65535 / height)
            self._send_mouse(x, y, 0, 0x0001 | 0x8000 | 0x4000)
            return
        if isinstance(event, PointerMotionEvent):
            self._send_mouse(event.dx, event.dy, 0, 0x0001)
            return
        if isinstance(event, MouseButtonEvent):
            flag, data = _windows_button_injection(event)
            self._send_mouse(0, 0, data, flag)
            return
        if isinstance(event, WheelEvent):
            if event.dy:
                self._send_mouse(0, 0, event.dy, 0x0800)
            if event.dx:
                self._send_mouse(0, 0, event.dx, 0x01000)
            return
        raise TypeError("unsupported Windows input event")

    def _send_mouse(self, dx: int, dy: int, data: int, flags: int) -> None:
        value = self.Input(
            type=0,
            mi=self.MouseInput(dx, dy, data & 0xFFFFFFFF, flags, 0, INJECTION_MARKER),
        )
        self._send_inputs([value])

    def _send_inputs(self, values) -> None:
        array = (self.Input * len(values))(*values)
        sent = self.user32.SendInput(
            len(values),
            array,
            self.ctypes.sizeof(self.Input),
        )
        if sent != len(values):
            raise self.ctypes.WinError(self.ctypes.get_last_error())


def _usage_location(usage: int) -> KeyLocation:
    if usage in {0xE0, 0xE1, 0xE2, 0xE3}:
        return KeyLocation.LEFT
    if usage in {0xE4, 0xE5, 0xE6, 0xE7}:
        return KeyLocation.RIGHT
    if 0x53 <= usage <= 0x63 or usage == 0x67:
        return KeyLocation.NUMPAD
    return KeyLocation.STANDARD


def _signed_word(value: int) -> int:
    return value - 0x10000 if value & 0x8000 else value


def _windows_button_injection(event: MouseButtonEvent) -> tuple[int, int]:
    return {
        (MouseButton.LEFT, KeyAction.DOWN): (0x0002, 0),
        (MouseButton.LEFT, KeyAction.UP): (0x0004, 0),
        (MouseButton.RIGHT, KeyAction.DOWN): (0x0008, 0),
        (MouseButton.RIGHT, KeyAction.UP): (0x0010, 0),
        (MouseButton.MIDDLE, KeyAction.DOWN): (0x0020, 0),
        (MouseButton.MIDDLE, KeyAction.UP): (0x0040, 0),
        (MouseButton.X1, KeyAction.DOWN): (0x0080, 1),
        (MouseButton.X1, KeyAction.UP): (0x0100, 1),
        (MouseButton.X2, KeyAction.DOWN): (0x0080, 2),
        (MouseButton.X2, KeyAction.UP): (0x0100, 2),
    }[(event.button, event.action)]


__all__ = ["WindowsInputBackend", "windows_key_to_usage"]
