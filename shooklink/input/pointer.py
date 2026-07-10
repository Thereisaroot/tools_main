"""Logical absolute pointer that discards impossible relative movement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from shooklink.input.topology import Side, Topology


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
        return PointerTransition(TransitionKind.ENTER, *self._position)

    def set_position(self, x: int, y: int) -> None:
        if not self.topology.contains(x, y):
            raise ValueError("pointer position must be on a connected monitor")
        self._position = (x, y)

    def move(self, dx: int, dy: int) -> PointerTransition:
        if type(dx) is not int or type(dy) is not int:
            raise TypeError("pointer deltas must be integers")
        x, y = self.position
        if (
            self.return_side is not None
            and self.topology.is_on_outer_edge(self.return_side, x, y)
            and _moves_outward(self.return_side, dx, dy)
        ):
            self._position = None
            return PointerTransition(TransitionKind.LEAVE, x, y)
        self._position = self.topology.nearest_point(x + dx, y + dy)
        return PointerTransition(TransitionKind.MOVE, *self._position)


def _moves_outward(side: Side, dx: int, dy: int) -> bool:
    return {
        Side.LEFT: dx < 0,
        Side.RIGHT: dx > 0,
        Side.TOP: dy < 0,
        Side.BOTTOM: dy > 0,
    }[side]


__all__ = ["LogicalPointer", "PointerTransition", "TransitionKind"]
