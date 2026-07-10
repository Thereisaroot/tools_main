"""Cross-platform input-sharing models and backends."""

from .pointer import LogicalPointer, PointerTransition, TransitionKind
from .topology import EdgeSegment, Monitor, Rect, Side, Topology

__all__ = [
    "EdgeSegment",
    "LogicalPointer",
    "Monitor",
    "PointerTransition",
    "Rect",
    "Side",
    "Topology",
    "TransitionKind",
]
