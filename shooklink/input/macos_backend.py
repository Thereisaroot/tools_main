"""Quartz event-tap capture and injection for macOS."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from typing import Any

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

_MAC_TO_USAGE = {
    0: 0x04,
    1: 0x16,
    2: 0x07,
    3: 0x09,
    4: 0x0B,
    5: 0x0A,
    6: 0x1D,
    7: 0x1B,
    8: 0x06,
    9: 0x19,
    11: 0x05,
    12: 0x14,
    13: 0x1A,
    14: 0x08,
    15: 0x15,
    16: 0x1C,
    17: 0x17,
    18: 0x1E,
    19: 0x1F,
    20: 0x20,
    21: 0x21,
    22: 0x23,
    23: 0x22,
    24: 0x2E,
    25: 0x26,
    26: 0x24,
    27: 0x2D,
    28: 0x25,
    29: 0x27,
    30: 0x30,
    31: 0x12,
    32: 0x18,
    33: 0x2F,
    34: 0x0C,
    35: 0x13,
    36: 0x28,
    37: 0x0F,
    38: 0x0D,
    39: 0x34,
    40: 0x0E,
    41: 0x33,
    42: 0x31,
    43: 0x36,
    44: 0x38,
    45: 0x11,
    46: 0x10,
    47: 0x37,
    48: 0x2B,
    49: 0x2C,
    50: 0x35,
    51: 0x2A,
    53: 0x29,
    54: 0xE7,
    55: 0xE3,
    56: 0xE1,
    57: 0x39,
    58: 0xE2,
    59: 0xE0,
    60: 0xE5,
    61: 0xE6,
    62: 0xE4,
    65: 0x63,
    67: 0x55,
    69: 0x57,
    71: 0x53,
    75: 0x54,
    76: 0x58,
    78: 0x56,
    81: 0x67,
    82: 0x62,
    83: 0x59,
    84: 0x5A,
    85: 0x5B,
    86: 0x5C,
    87: 0x5D,
    88: 0x5E,
    89: 0x5F,
    91: 0x60,
    92: 0x61,
    96: 0x3E,
    97: 0x3F,
    98: 0x40,
    99: 0x3C,
    100: 0x41,
    101: 0x42,
    103: 0x44,
    109: 0x43,
    111: 0x45,
    115: 0x4A,
    116: 0x4B,
    117: 0x4C,
    118: 0x3D,
    119: 0x4D,
    120: 0x3B,
    121: 0x4E,
    122: 0x3A,
    123: 0x50,
    124: 0x4F,
    125: 0x51,
    126: 0x52,
}
_USAGE_TO_MAC = {usage: keycode for keycode, usage in _MAC_TO_USAGE.items()}


@dataclass(slots=True)
class _MacCaptureGeneration:
    number: int
    ready: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    error: BaseException | None = None
    run_loop: Any = None
    event_tap: Any = None
    callback_ref: Any = None


def mac_keycode_to_usage(keycode: int) -> int:
    if type(keycode) is not int:
        raise TypeError("macOS keycode must be an integer")
    return _MAC_TO_USAGE.get(keycode, 0)


class MacOSInputBackend(BaseInputBackend):
    def __init__(self) -> None:
        super().__init__()
        self._tap_thread: threading.Thread | None = None
        self._tap_ready = threading.Event()
        self._tap_error: BaseException | None = None
        self._run_loop = None
        self._event_tap = None
        self._tap_callback_ref = None
        self._modifier_keys_down: set[int] = set()
        self._tap_generation_number = 0
        self._tap_generation: _MacCaptureGeneration | None = None

    def permission_status(self) -> PermissionStatus:
        if sys.platform != "darwin":
            return PermissionStatus(False, False, "macOS only")
        import ApplicationServices
        import Quartz

        capture = bool(Quartz.CGPreflightListenEventAccess())
        inject = bool(ApplicationServices.AXIsProcessTrusted())
        detail = "ready" if capture and inject else "Accessibility/Input Monitoring required"
        return PermissionStatus(capture, inject, detail)

    def monitors(self) -> tuple[Monitor, ...]:
        if sys.platform != "darwin":
            return ()
        import Quartz

        error, displays, _count = Quartz.CGGetOnlineDisplayList(64, None, None)
        if error:
            return ()
        result = []
        for display_id in displays:
            bounds = Quartz.CGDisplayBounds(display_id)
            result.append(
                Monitor(
                    str(display_id),
                    Rect(
                        round(bounds.origin.x),
                        round(bounds.origin.y),
                        round(bounds.size.width),
                        round(bounds.size.height),
                    ),
                )
            )
        return tuple(result)

    def cursor_position(self) -> tuple[int, int]:
        import Quartz

        point = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return round(point.x), round(point.y)

    def warp_cursor(self, x: int, y: int) -> None:
        import Quartz

        Quartz.CGWarpMouseCursorPosition((x, y))

    def _start_native_capture(self, suppress: bool) -> None:
        self._require_input_permissions()
        self._reset_capture_local_state()
        self._tap_generation_number += 1
        generation = _MacCaptureGeneration(self._tap_generation_number)
        generation.thread = threading.Thread(
            target=self._tap_loop,
            args=(generation,),
            name="shooklink-quartz-event-tap",
            daemon=True,
        )
        self._tap_generation = generation
        self._publish_tap_generation(generation)
        generation.thread.start()
        if not generation.ready.wait(3):
            raise RuntimeError("macOS event tap did not start")
        if generation.error is not None:
            raise RuntimeError("macOS event tap failed") from generation.error

    def _reset_capture_local_state(self) -> None:
        self._modifier_keys_down.clear()

    def _stop_native_capture(self) -> None:
        generation = self._tap_generation
        thread = generation.thread if generation is not None else self._tap_thread
        if thread is None:
            self._clear_native_capture_refs(generation)
            return
        if not thread.is_alive():
            self._clear_native_capture_refs(generation)
            return
        run_loop = generation.run_loop if generation is not None else self._run_loop
        if run_loop is None:
            raise RuntimeError("macOS event-tap thread has no CFRunLoop")
        import CoreFoundation

        try:
            CoreFoundation.CFRunLoopStop(run_loop)
        except Exception as error:
            raise RuntimeError("could not stop macOS run loop") from error
        if thread is threading.current_thread():
            return
        thread.join(3)
        if thread.is_alive():
            raise RuntimeError("macOS event-tap thread did not stop within 3 seconds")
        self._clear_native_capture_refs(generation)

    def _clear_native_capture_refs(
        self,
        generation: _MacCaptureGeneration | None = None,
    ) -> None:
        if generation is not None:
            generation.thread = None
            generation.run_loop = None
            generation.event_tap = None
            generation.callback_ref = None
            if self._tap_generation is not generation:
                return
            self._tap_generation = None
        self._tap_thread = None
        self._run_loop = None
        self._event_tap = None
        self._tap_callback_ref = None

    def _publish_tap_generation(self, generation: _MacCaptureGeneration) -> None:
        if self._tap_generation is not generation:
            return
        self._tap_ready = generation.ready
        self._tap_error = generation.error
        self._tap_thread = generation.thread
        self._run_loop = generation.run_loop
        self._event_tap = generation.event_tap
        self._tap_callback_ref = generation.callback_ref

    def _require_input_permissions(self) -> None:
        status = self.permission_status()
        if not status.capture_allowed or not status.inject_allowed:
            raise PermissionError(status.detail)

    def _tap_loop(self, generation: _MacCaptureGeneration) -> None:
        try:
            import CoreFoundation
            import Quartz

            event_types = (
                Quartz.kCGEventKeyDown,
                Quartz.kCGEventKeyUp,
                Quartz.kCGEventFlagsChanged,
                Quartz.kCGEventMouseMoved,
                Quartz.kCGEventLeftMouseDragged,
                Quartz.kCGEventRightMouseDragged,
                Quartz.kCGEventOtherMouseDragged,
                Quartz.kCGEventLeftMouseDown,
                Quartz.kCGEventLeftMouseUp,
                Quartz.kCGEventRightMouseDown,
                Quartz.kCGEventRightMouseUp,
                Quartz.kCGEventOtherMouseDown,
                Quartz.kCGEventOtherMouseUp,
                Quartz.kCGEventScrollWheel,
            )
            mask = sum(1 << event_type for event_type in event_types)
            generation.callback_ref = (
                lambda proxy, event_type, event, context: self._tap_callback(
                    proxy,
                    event_type,
                    event,
                    context,
                    _generation=generation,
                )
            )
            self._publish_tap_generation(generation)
            generation.event_tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionDefault,
                mask,
                generation.callback_ref,
                None,
            )
            self._publish_tap_generation(generation)
            if generation.event_tap is None:
                raise PermissionError("could not create macOS event tap")
            source = Quartz.CFMachPortCreateRunLoopSource(
                None,
                generation.event_tap,
                0,
            )
            generation.run_loop = CoreFoundation.CFRunLoopGetCurrent()
            self._publish_tap_generation(generation)
            CoreFoundation.CFRunLoopAddSource(
                generation.run_loop,
                source,
                CoreFoundation.kCFRunLoopCommonModes,
            )
            Quartz.CGEventTapEnable(generation.event_tap, True)
            generation.ready.set()
            CoreFoundation.CFRunLoopRun()
        except BaseException as error:
            generation.error = error
            self._publish_tap_generation(generation)
            generation.ready.set()
        finally:
            generation.run_loop = None
            generation.event_tap = None
            generation.callback_ref = None
            self._publish_tap_generation(generation)

    def _tap_callback(
        self,
        _proxy,
        event_type,
        event,
        _context,
        *,
        _generation: _MacCaptureGeneration | None = None,
    ):
        if _generation is not None and self._tap_generation is not _generation:
            return event
        import Quartz

        if event_type in (
            Quartz.kCGEventTapDisabledByTimeout,
            Quartz.kCGEventTapDisabledByUserInput,
        ):
            event_tap = (
                _generation.event_tap
                if _generation is not None
                else self._event_tap
            )
            if event_tap is not None:
                Quartz.CGEventTapEnable(event_tap, True)
            return event
        normalized = self._normalize_event(event_type, event, Quartz)
        if normalized is None:
            return event
        consumed = self.emit_captured(normalized)
        if consumed or self._capture_suppress:
            return None
        return event

    def _normalize_event(self, event_type: int, event: Any, quartz) -> InputEvent | None:
        marker = quartz.CGEventGetIntegerValueField(
            event,
            quartz.kCGEventSourceUserData,
        )
        injected = marker == INJECTION_MARKER
        key_types = {
            quartz.kCGEventKeyDown,
            quartz.kCGEventKeyUp,
            quartz.kCGEventFlagsChanged,
        }
        if event_type in key_types:
            keycode = int(
                quartz.CGEventGetIntegerValueField(event, quartz.kCGKeyboardEventKeycode)
            )
            if event_type == quartz.kCGEventFlagsChanged:
                action = (
                    KeyAction.UP
                    if keycode in self._modifier_keys_down
                    else KeyAction.DOWN
                )
                if action is KeyAction.DOWN:
                    self._modifier_keys_down.add(keycode)
                else:
                    self._modifier_keys_down.discard(keycode)
            else:
                action = (
                    KeyAction.DOWN
                    if event_type == quartz.kCGEventKeyDown
                    else KeyAction.UP
                )
            unicode_result = quartz.CGEventKeyboardGetUnicodeString(event, 16, None, None)
            text = unicode_result[1] if unicode_result else ""
            usage = mac_keycode_to_usage(keycode)
            return KeyEvent(
                action,
                usage,
                scan_code=keycode,
                virtual_key=keycode,
                text=text,
                modifiers=_mac_modifiers(quartz.CGEventGetFlags(event), quartz),
                location=_usage_location(usage),
                repeat=bool(
                    quartz.CGEventGetIntegerValueField(
                        event,
                        quartz.kCGKeyboardEventAutorepeat,
                    )
                ),
                injected=injected,
            )
        if event_type in {
            quartz.kCGEventMouseMoved,
            quartz.kCGEventLeftMouseDragged,
            quartz.kCGEventRightMouseDragged,
            quartz.kCGEventOtherMouseDragged,
        }:
            return PointerMotionEvent(
                int(quartz.CGEventGetIntegerValueField(event, quartz.kCGMouseEventDeltaX)),
                int(quartz.CGEventGetIntegerValueField(event, quartz.kCGMouseEventDeltaY)),
                injected,
            )
        if event_type == quartz.kCGEventScrollWheel:
            return WheelEvent(
                int(
                    quartz.CGEventGetIntegerValueField(
                        event,
                        quartz.kCGScrollWheelEventPointDeltaAxis2,
                    )
                ),
                int(
                    quartz.CGEventGetIntegerValueField(
                        event,
                        quartz.kCGScrollWheelEventPointDeltaAxis1,
                    )
                ),
                injected,
            )
        button = _mac_mouse_button(event_type, event, quartz)
        if button is not None:
            action = (
                KeyAction.DOWN
                if event_type
                in {
                    quartz.kCGEventLeftMouseDown,
                    quartz.kCGEventRightMouseDown,
                    quartz.kCGEventOtherMouseDown,
                }
                else KeyAction.UP
            )
            return MouseButtonEvent(button, action, injected)
        return None

    def _inject_native(self, event: InputEvent) -> None:
        self._require_input_permissions()
        import Quartz

        source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStatePrivate)
        quartz_event = None
        if isinstance(event, KeyEvent):
            keycode = _USAGE_TO_MAC.get(event.usage)
            if keycode is None:
                raise ValueError(f"unsupported macOS HID usage {event.usage}")
            quartz_event = Quartz.CGEventCreateKeyboardEvent(
                source,
                keycode,
                event.action is KeyAction.DOWN,
            )
            if event.text and event.action is KeyAction.DOWN and event.usage == 0:
                Quartz.CGEventKeyboardSetUnicodeString(
                    quartz_event,
                    len(event.text),
                    event.text,
                )
        elif isinstance(event, PointerPositionEvent):
            quartz_event = Quartz.CGEventCreateMouseEvent(
                source,
                Quartz.kCGEventMouseMoved,
                event.position,
                Quartz.kCGMouseButtonLeft,
            )
        elif isinstance(event, PointerMotionEvent):
            x, y = self.cursor_position()
            quartz_event = Quartz.CGEventCreateMouseEvent(
                source,
                Quartz.kCGEventMouseMoved,
                (x + event.dx, y + event.dy),
                Quartz.kCGMouseButtonLeft,
            )
        elif isinstance(event, MouseButtonEvent):
            quartz_event = _create_mac_button_event(source, event, self.cursor_position(), Quartz)
        elif isinstance(event, WheelEvent):
            quartz_event = Quartz.CGEventCreateScrollWheelEvent(
                source,
                Quartz.kCGScrollEventUnitPixel,
                2,
                event.dy,
                event.dx,
            )
        if quartz_event is None:
            raise TypeError("unsupported macOS input event")
        Quartz.CGEventSetIntegerValueField(
            quartz_event,
            Quartz.kCGEventSourceUserData,
            INJECTION_MARKER,
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, quartz_event)


def _mac_modifiers(flags: int, quartz) -> Modifiers:
    result = Modifiers.NONE
    for flag, modifier in (
        (quartz.kCGEventFlagMaskShift, Modifiers.SHIFT),
        (quartz.kCGEventFlagMaskControl, Modifiers.CONTROL),
        (quartz.kCGEventFlagMaskAlternate, Modifiers.ALT),
        (quartz.kCGEventFlagMaskCommand, Modifiers.META),
        (quartz.kCGEventFlagMaskAlphaShift, Modifiers.CAPS_LOCK),
    ):
        if flags & flag:
            result |= modifier
    return result


def _usage_location(usage: int) -> KeyLocation:
    if usage in {0xE0, 0xE1, 0xE2, 0xE3}:
        return KeyLocation.LEFT
    if usage in {0xE4, 0xE5, 0xE6, 0xE7}:
        return KeyLocation.RIGHT
    if 0x53 <= usage <= 0x63 or usage == 0x67:
        return KeyLocation.NUMPAD
    return KeyLocation.STANDARD


def _mac_mouse_button(event_type: int, event, quartz) -> MouseButton | None:
    if event_type in {quartz.kCGEventLeftMouseDown, quartz.kCGEventLeftMouseUp}:
        return MouseButton.LEFT
    if event_type in {quartz.kCGEventRightMouseDown, quartz.kCGEventRightMouseUp}:
        return MouseButton.RIGHT
    if event_type in {quartz.kCGEventOtherMouseDown, quartz.kCGEventOtherMouseUp}:
        number = int(
            quartz.CGEventGetIntegerValueField(event, quartz.kCGMouseEventButtonNumber)
        )
        return {2: MouseButton.MIDDLE, 3: MouseButton.X1, 4: MouseButton.X2}.get(
            number,
            MouseButton.MIDDLE,
        )
    return None


def _create_mac_button_event(source, event, position, quartz):
    button, down_type, up_type = {
        MouseButton.LEFT: (
            quartz.kCGMouseButtonLeft,
            quartz.kCGEventLeftMouseDown,
            quartz.kCGEventLeftMouseUp,
        ),
        MouseButton.RIGHT: (
            quartz.kCGMouseButtonRight,
            quartz.kCGEventRightMouseDown,
            quartz.kCGEventRightMouseUp,
        ),
        MouseButton.MIDDLE: (
            quartz.kCGMouseButtonCenter,
            quartz.kCGEventOtherMouseDown,
            quartz.kCGEventOtherMouseUp,
        ),
        MouseButton.X1: (3, quartz.kCGEventOtherMouseDown, quartz.kCGEventOtherMouseUp),
        MouseButton.X2: (4, quartz.kCGEventOtherMouseDown, quartz.kCGEventOtherMouseUp),
    }[event.button]
    event_type = down_type if event.action is KeyAction.DOWN else up_type
    return quartz.CGEventCreateMouseEvent(source, event_type, position, button)


__all__ = ["MacOSInputBackend", "mac_keycode_to_usage"]
