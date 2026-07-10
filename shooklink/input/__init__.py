"""Cross-platform input-sharing models and backends."""

from .backend import BaseInputBackend, PermissionStatus
from .events import (
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
from .pointer import LogicalPointer, PointerTransition, TransitionKind
from .service import (
    InputProtocolError,
    InputService,
    InputSessionState,
    InputStateChange,
    InputUnavailable,
)
from .topology import EdgeSegment, Monitor, Rect, Side, Topology

__all__ = [
    "EdgeSegment",
    "BaseInputBackend",
    "KeyAction",
    "KeyEvent",
    "KeyLocation",
    "LogicalPointer",
    "InputProtocolError",
    "InputService",
    "InputSessionState",
    "InputStateChange",
    "InputUnavailable",
    "Modifiers",
    "Monitor",
    "MouseButton",
    "MouseButtonEvent",
    "PermissionStatus",
    "PointerMotionEvent",
    "PointerPositionEvent",
    "PointerTransition",
    "Rect",
    "Side",
    "Topology",
    "TransitionKind",
    "WheelEvent",
]
