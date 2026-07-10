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


def windows_key_to_usage(virtual_key: int, scan_code: int, extended: bool) -> int:
    if any(type(value) is not int for value in (virtual_key, scan_code)):
        raise TypeError("Windows key values must be integers")
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
        if self._hook_thread_id is not None:
            self._get_api().post_quit(self._hook_thread_id)
        if self._hook_thread is not None and self._hook_thread is not threading.current_thread():
            self._hook_thread.join(3)
        self._hook_thread = None
        self._hook_thread_id = None

    def _hook_loop(self) -> None:
        try:
            api = self._get_api()
            self._hook_thread_id = api.current_thread_id()
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
            if self._api is not None:
                self._api.unhook(self._keyboard_hook)
                self._api.unhook(self._mouse_hook)
            self._keyboard_hook = None
            self._mouse_hook = None
            self._keyboard_callback_ref = None
            self._mouse_callback_ref = None

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
        if action is KeyAction.DOWN:
            self._captured_keys_down.add(usage)
        else:
            self._captured_keys_down.discard(usage)
        modifiers = self._update_modifiers(usage, action)
        event = KeyEvent(
            action,
            usage,
            scan_code=int(data.scanCode),
            virtual_key=int(data.vkCode),
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

    def _update_modifiers(self, usage: int, action: KeyAction) -> Modifiers:
        modifier = {
            0xE0: Modifiers.CONTROL,
            0xE4: Modifiers.CONTROL,
            0xE1: Modifiers.SHIFT,
            0xE5: Modifiers.SHIFT,
            0xE2: Modifiers.ALT,
            0xE6: Modifiers.ALT,
            0xE3: Modifiers.META,
            0xE7: Modifiers.META,
        }.get(usage)
        if modifier is not None:
            if action is KeyAction.DOWN:
                self._modifiers |= modifier
            else:
                self._modifiers &= ~modifier
        if usage == 0x39 and action is KeyAction.DOWN:
            self._modifiers ^= Modifiers.CAPS_LOCK
        return self._modifiers

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
            ]

        self.Point = Point
        self.KeyboardData = KeyboardData
        self.MouseData = MouseData
        self.MouseInput = MouseInput
        self.KeyboardInput = KeyboardInput
        self.Input = Input
        self.Message = Message
        self.HookProc = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )

    def hook_proc(self, callback):
        return self.HookProc(callback)

    def install_hook(self, hook_type, callback):
        hook = self.user32.SetWindowsHookExW(
            hook_type,
            callback,
            self.kernel32.GetModuleHandleW(None),
            0,
        )
        if not hook:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return hook

    def unhook(self, hook) -> None:
        if hook:
            self.user32.UnhookWindowsHookEx(hook)

    def call_next(self, hook, code, message, pointer):
        return self.user32.CallNextHookEx(hook, code, message, pointer)

    def keyboard_data(self, pointer):
        return self.ctypes.cast(pointer, self.ctypes.POINTER(self.KeyboardData)).contents

    def mouse_data(self, pointer):
        return self.ctypes.cast(pointer, self.ctypes.POINTER(self.MouseData)).contents

    def current_thread_id(self) -> int:
        return int(self.kernel32.GetCurrentThreadId())

    def post_quit(self, thread_id: int) -> None:
        self.user32.PostThreadMessageW(thread_id, 0x0012, 0, 0)

    def message_loop(self) -> None:
        message = self.Message()
        while self.user32.GetMessageW(self.ctypes.byref(message), None, 0, 0) > 0:
            self.user32.TranslateMessage(self.ctypes.byref(message))
            self.user32.DispatchMessageW(self.ctypes.byref(message))

    def key_is_down(self, virtual_key: int) -> bool:
        return bool(self.user32.GetAsyncKeyState(virtual_key) & 0x4000)

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

        class MonitorInfo(self.ctypes.Structure):
            _fields_ = [
                ("cbSize", self.wintypes.DWORD),
                ("rcMonitor", self.wintypes.RECT),
                ("rcWork", self.wintypes.RECT),
                ("dwFlags", self.wintypes.DWORD),
            ]

        callback_type = self.ctypes.WINFUNCTYPE(
            self.wintypes.BOOL,
            self.wintypes.HMONITOR,
            self.wintypes.HDC,
            self.ctypes.POINTER(self.wintypes.RECT),
            self.wintypes.LPARAM,
        )

        @callback_type
        def callback(handle, _device, _rect, _data):
            info = MonitorInfo()
            info.cbSize = self.ctypes.sizeof(info)
            if self.user32.GetMonitorInfoW(handle, self.ctypes.byref(info)):
                rect = info.rcMonitor
                monitors.append(
                    Monitor(
                        str(int(handle)),
                        Rect(
                            int(rect.left),
                            int(rect.top),
                            int(rect.right - rect.left),
                            int(rect.bottom - rect.top),
                        ),
                    )
                )
            return True

        self.user32.EnumDisplayMonitors(None, None, callback, 0)
        return tuple(monitors)

    def inject(self, event: InputEvent) -> None:
        if isinstance(event, KeyEvent):
            virtual_key = _USAGE_TO_VK.get(event.usage)
            if virtual_key is None:
                raise ValueError(f"unsupported Windows HID usage {event.usage}")
            flags = 0x0002 if event.action is KeyAction.UP else 0
            if event.usage in {0x58, 0xE4, 0xE6, 0xE7, 0x4F, 0x50, 0x51, 0x52}:
                flags |= 0x0001
            value = self.Input(
                type=1,
                ki=self.KeyboardInput(
                    virtual_key,
                    0,
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
            width = max(1, self.user32.GetSystemMetrics(78) - 1)
            height = max(1, self.user32.GetSystemMetrics(79) - 1)
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
