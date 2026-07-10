"""Immutable monitor geometry and connected outer-edge mapping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

    @property
    def opposite(self) -> Side:
        return {
            Side.LEFT: Side.RIGHT,
            Side.RIGHT: Side.LEFT,
            Side.TOP: Side.BOTTOM,
            Side.BOTTOM: Side.TOP,
        }[self]


@dataclass(frozen=True, slots=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if type(self.x) is not int or type(self.y) is not int:
            raise TypeError("rectangle coordinates must be integers")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("rectangle width must be positive")
        if type(self.height) is not int or self.height <= 0:
            raise ValueError("rectangle height must be positive")

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def clamp_point(self, x: int, y: int) -> tuple[int, int]:
        return (
            min(max(x, self.x), self.right - 1),
            min(max(y, self.y), self.bottom - 1),
        )


@dataclass(frozen=True, slots=True)
class Monitor:
    monitor_id: str
    rect: Rect

    def __post_init__(self) -> None:
        if not isinstance(self.monitor_id, str) or not self.monitor_id:
            raise ValueError("monitor_id must be a non-empty string")
        if not isinstance(self.rect, Rect):
            raise TypeError("monitor rect must be a Rect")


@dataclass(frozen=True, slots=True)
class EdgeSegment:
    side: Side
    coordinate: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.side, Side):
            raise TypeError("edge side must be a Side")
        if any(type(value) is not int for value in (self.coordinate, self.start, self.end)):
            raise TypeError("edge coordinates must be integers")
        if self.end <= self.start:
            raise ValueError("edge segment must have positive length")

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains_point(self, x: int, y: int) -> bool:
        if self.side in (Side.LEFT, Side.RIGHT):
            return x == self.coordinate and self.start <= y < self.end
        return y == self.coordinate and self.start <= x < self.end


class Topology:
    def __init__(self, monitors: tuple[Monitor, ...] | list[Monitor]) -> None:
        self.monitors = tuple(monitors)
        if not self.monitors:
            raise ValueError("topology must contain at least one monitor")
        if not all(isinstance(monitor, Monitor) for monitor in self.monitors):
            raise TypeError("topology monitors must be Monitor values")
        identifiers = [monitor.monitor_id for monitor in self.monitors]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("monitor IDs must be unique")
        self._edges = {
            side: self._build_edge_segments(side)
            for side in Side
        }

    def contains(self, x: int, y: int) -> bool:
        return any(monitor.rect.contains(x, y) for monitor in self.monitors)

    def monitor_at(self, x: int, y: int) -> Monitor | None:
        return next(
            (monitor for monitor in self.monitors if monitor.rect.contains(x, y)),
            None,
        )

    def edge_segments(self, side: Side) -> tuple[EdgeSegment, ...]:
        if not isinstance(side, Side):
            raise TypeError("side must be a Side")
        return self._edges[side]

    def is_on_outer_edge(self, side: Side, x: int, y: int) -> bool:
        return any(
            segment.contains_point(x, y)
            for segment in self.edge_segments(side)
        )

    def map_fraction_to_edge(self, side: Side, fraction: float) -> tuple[int, int]:
        if not isinstance(fraction, (int, float)) or not math.isfinite(fraction):
            raise TypeError("edge fraction must be a finite number")
        if not 0 <= fraction <= 1:
            raise ValueError("edge fraction must be from 0 to 1")
        segments = self.edge_segments(side)
        total = sum(segment.length for segment in segments)
        offset = min(total - 1, int(float(fraction) * total))
        for segment in segments:
            if offset < segment.length:
                cross_axis = segment.start + offset
                if side in (Side.LEFT, Side.RIGHT):
                    return segment.coordinate, cross_axis
                return cross_axis, segment.coordinate
            offset -= segment.length
        raise AssertionError("edge fraction mapping failed")

    def edge_fraction(self, side: Side, x: int, y: int) -> float:
        segments = self.edge_segments(side)
        total = sum(segment.length for segment in segments)
        preceding = 0
        for segment in segments:
            if segment.contains_point(x, y):
                cross_axis = y if side in (Side.LEFT, Side.RIGHT) else x
                return (preceding + cross_axis - segment.start) / max(1, total - 1)
            preceding += segment.length
        raise ValueError("point is not on the requested outer edge")

    def nearest_point(self, x: int, y: int) -> tuple[int, int]:
        if type(x) is not int or type(y) is not int:
            raise TypeError("pointer coordinates must be integers")
        if self.contains(x, y):
            return x, y
        candidates = []
        for order, monitor in enumerate(self.monitors):
            candidate_x, candidate_y = monitor.rect.clamp_point(x, y)
            distance = (candidate_x - x) ** 2 + (candidate_y - y) ** 2
            candidates.append((distance, order, candidate_x, candidate_y))
        _distance, _order, nearest_x, nearest_y = min(candidates)
        return nearest_x, nearest_y

    def _build_edge_segments(self, side: Side) -> tuple[EdgeSegment, ...]:
        if side is Side.LEFT:
            boundary = min(monitor.rect.x for monitor in self.monitors)
            spans = [
                (monitor.rect.y, monitor.rect.bottom)
                for monitor in self.monitors
                if monitor.rect.x == boundary
            ]
            coordinate = boundary
        elif side is Side.RIGHT:
            boundary = max(monitor.rect.right for monitor in self.monitors)
            spans = [
                (monitor.rect.y, monitor.rect.bottom)
                for monitor in self.monitors
                if monitor.rect.right == boundary
            ]
            coordinate = boundary - 1
        elif side is Side.TOP:
            boundary = min(monitor.rect.y for monitor in self.monitors)
            spans = [
                (monitor.rect.x, monitor.rect.right)
                for monitor in self.monitors
                if monitor.rect.y == boundary
            ]
            coordinate = boundary
        else:
            boundary = max(monitor.rect.bottom for monitor in self.monitors)
            spans = [
                (monitor.rect.x, monitor.rect.right)
                for monitor in self.monitors
                if monitor.rect.bottom == boundary
            ]
            coordinate = boundary - 1
        merged = _merge_spans(spans)
        return tuple(
            EdgeSegment(side, coordinate, start, end)
            for start, end in merged
        )


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


__all__ = ["EdgeSegment", "Monitor", "Rect", "Side", "Topology"]
