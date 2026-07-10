"""Platform-neutral physical keyboard and pointer events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntFlag


class KeyAction(str, Enum):
    DOWN = "down"
    UP = "up"


class KeyLocation(str, Enum):
    STANDARD = "standard"
    LEFT = "left"
    RIGHT = "right"
    NUMPAD = "numpad"


class Modifiers(IntFlag):
    NONE = 0
    SHIFT = 1 << 0
    CONTROL = 1 << 1
    ALT = 1 << 2
    META = 1 << 3
    CAPS_LOCK = 1 << 4
    NUM_LOCK = 1 << 5


class MouseButton(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"
    X1 = "x1"
    X2 = "x2"


@dataclass(frozen=True, slots=True)
class KeyEvent:
    action: KeyAction
    usage: int
    scan_code: int = 0
    virtual_key: int = 0
    text: str = ""
    modifiers: Modifiers = Modifiers.NONE
    location: KeyLocation = KeyLocation.STANDARD
    repeat: bool = False
    extended: bool = False
    injected: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.action, KeyAction):
            raise TypeError("key action must be a KeyAction")
        _validate_uint("usage", self.usage, (1 << 16) - 1)
        _validate_uint("scan_code", self.scan_code, (1 << 16) - 1)
        _validate_uint("virtual_key", self.virtual_key, (1 << 16) - 1)
        if not isinstance(self.text, str):
            raise TypeError("key text must be a string")
        if len(self.text) > 32:
            raise ValueError("key text cannot exceed 32 characters")
        if not isinstance(self.modifiers, Modifiers):
            raise TypeError("key modifiers must be Modifiers")
        if not isinstance(self.location, KeyLocation):
            raise TypeError("key location must be a KeyLocation")
        for name, value in (
            ("repeat", self.repeat),
            ("extended", self.extended),
            ("injected", self.injected),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class PointerMotionEvent:
    dx: int
    dy: int
    injected: bool = False

    def __post_init__(self) -> None:
        _validate_int("dx", self.dx)
        _validate_int("dy", self.dy)
        if not isinstance(self.injected, bool):
            raise TypeError("injected must be a boolean")


@dataclass(frozen=True, slots=True)
class PointerPositionEvent:
    x: int
    y: int
    injected: bool = False

    def __post_init__(self) -> None:
        _validate_int("x", self.x)
        _validate_int("y", self.y)
        if not isinstance(self.injected, bool):
            raise TypeError("injected must be a boolean")

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True, slots=True)
class MouseButtonEvent:
    button: MouseButton
    action: KeyAction
    injected: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.button, MouseButton):
            raise TypeError("mouse button must be a MouseButton")
        if not isinstance(self.action, KeyAction):
            raise TypeError("mouse action must be a KeyAction")
        if not isinstance(self.injected, bool):
            raise TypeError("injected must be a boolean")


@dataclass(frozen=True, slots=True)
class WheelEvent:
    dx: int
    dy: int
    injected: bool = False

    def __post_init__(self) -> None:
        _validate_int("dx", self.dx)
        _validate_int("dy", self.dy)
        if not isinstance(self.injected, bool):
            raise TypeError("injected must be a boolean")


InputEvent = (
    KeyEvent
    | PointerMotionEvent
    | PointerPositionEvent
    | MouseButtonEvent
    | WheelEvent
)


def _validate_uint(name: str, value: int, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 0 to {maximum}")


def _validate_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")


__all__ = [
    "InputEvent",
    "KeyAction",
    "KeyEvent",
    "KeyLocation",
    "Modifiers",
    "MouseButton",
    "MouseButtonEvent",
    "PointerMotionEvent",
    "PointerPositionEvent",
    "WheelEvent",
]
