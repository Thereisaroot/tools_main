"""Logical absolute pointer that discards impossible relative movement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from shooklink.input.topology import Side, Topology

RETURN_EDGE_RESISTANCE_PIXELS = 32


class TransitionKind(str, Enum):
    ENTER = "enter"
    MOVE = "move"
    LEAVE = "leave"


@dataclass(frozen=True, slots=True)
class PointerTransition:
    kind: TransitionKind
    x: int
    y: int

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y


class LogicalPointer:
    def __init__(
        self,
        topology: Topology,
        *,
        return_side: Side | None = None,
    ) -> None:
        if not isinstance(topology, Topology):
            raise TypeError("topology must be a Topology")
        if return_side is not None and not isinstance(return_side, Side):
            raise TypeError("return_side must be a Side or None")
        self.topology = topology
        self.return_side = return_side
        self._position: tuple[int, int] | None = None
        self._return_pressure = 0

    @property
    def active(self) -> bool:
        return self._position is not None

    @property
    def position(self) -> tuple[int, int]:
        if self._position is None:
            raise RuntimeError("pointer has not entered the destination")
        return self._position

    def enter(self, side: Side, cross_axis_fraction: float) -> PointerTransition:
        self._position = self.topology.map_fraction_to_edge(side, cross_axis_fraction)
        self._return_pressure = 0
        return PointerTransition(TransitionKind.ENTER, *self._position)

    def set_position(self, x: int, y: int) -> None:
        if type(x) is not int or type(y) is not int:
            raise TypeError("pointer coordinates must be integers")
        if not self.topology.contains(x, y):
            raise ValueError("pointer position must be on a connected monitor")
        if self._position != (x, y):
            self._return_pressure = 0
        self._position = (x, y)

    def move(self, dx: int, dy: int) -> PointerTransition:
        if type(dx) is not int or type(dy) is not int:
            raise TypeError("pointer deltas must be integers")
        x, y = self.position
        crossing = (
            None
            if self.return_side is None
            else _return_edge_crossing(
                self.topology,
                self.return_side,
                x,
                y,
                dx,
                dy,
            )
        )
        if crossing is not None:
            self._return_pressure += _outward_overflow(
                self.return_side,
                x,
                y,
                dx,
                dy,
                crossing,
            )
            if self._return_pressure >= RETURN_EDGE_RESISTANCE_PIXELS:
                self._position = None
                self._return_pressure = 0
                return PointerTransition(TransitionKind.LEAVE, *crossing)
            self._position = crossing
            return PointerTransition(TransitionKind.MOVE, *crossing)
        self._position = self.topology.move_point(x, y, dx, dy)
        if (
            self.return_side is not None
            and (
                _moves_outward(self.return_side.opposite, dx, dy)
                or not self.topology.is_on_outer_edge(
                    self.return_side,
                    *self._position,
                )
            )
        ):
            self._return_pressure = 0
        return PointerTransition(TransitionKind.MOVE, *self._position)


def _moves_outward(side: Side, dx: int, dy: int) -> bool:
    return {
        Side.LEFT: dx < 0,
        Side.RIGHT: dx > 0,
        Side.TOP: dy < 0,
        Side.BOTTOM: dy > 0,
    }[side]


def _return_edge_crossing(
    topology: Topology,
    side: Side,
    x: int,
    y: int,
    dx: int,
    dy: int,
) -> tuple[int, int] | None:
    if not _moves_outward(side, dx, dy):
        return None
    for segment in topology.edge_segments(side):
        if side in (Side.LEFT, Side.RIGHT):
            target = x + dx
            crossed = (
                x == segment.coordinate
                or side is Side.LEFT and target < segment.coordinate < x
                or side is Side.RIGHT and x < segment.coordinate < target
            )
            if not crossed:
                continue
            fraction = 0.0 if x == segment.coordinate else (segment.coordinate - x) / dx
            cross_axis = round(y + dy * fraction)
            if segment.start <= cross_axis < segment.end:
                return segment.coordinate, cross_axis
        else:
            target = y + dy
            crossed = (
                y == segment.coordinate
                or side is Side.TOP and target < segment.coordinate < y
                or side is Side.BOTTOM and y < segment.coordinate < target
            )
            if not crossed:
                continue
            fraction = 0.0 if y == segment.coordinate else (segment.coordinate - y) / dy
            cross_axis = round(x + dx * fraction)
            if segment.start <= cross_axis < segment.end:
                return cross_axis, segment.coordinate
    return None


def _outward_overflow(
    side: Side,
    x: int,
    y: int,
    dx: int,
    dy: int,
    crossing: tuple[int, int],
) -> int:
    target_x = x + dx
    target_y = y + dy
    edge_x, edge_y = crossing
    return {
        Side.LEFT: max(0, edge_x - target_x),
        Side.RIGHT: max(0, target_x - edge_x),
        Side.TOP: max(0, edge_y - target_y),
        Side.BOTTOM: max(0, target_y - edge_y),
    }[side]


__all__ = [
    "RETURN_EDGE_RESISTANCE_PIXELS",
    "LogicalPointer",
    "PointerTransition",
    "TransitionKind",
]
