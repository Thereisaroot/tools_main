"""Thread-safe priority scheduling for one shared serial link."""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, replace
from enum import IntEnum
from threading import Condition

UINT8_MAX = (1 << 8) - 1
UINT32_MAX = (1 << 32) - 1


class MultiplexerClosed(RuntimeError):
    """Raised when work is submitted after a multiplexer closes."""


class Priority(IntEnum):
    INTERACTIVE = 0
    NORMAL = 1
    MOTION = 2
    FILE = 3


@dataclass(frozen=True, slots=True)
class OutboundItem:
    priority: Priority
    stream_id: int
    payload: bytes
    message_type: int = 0
    flags: int = 0
    acknowledgement: int = 0
    sequence: int | None = None


def _validate_uint(name: str, value: int, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 0 to {maximum}")


def _validate_item(item: OutboundItem) -> None:
    if not isinstance(item, OutboundItem):
        raise TypeError("item must be an OutboundItem")
    if not isinstance(item.priority, Priority):
        raise TypeError("priority must be a Priority")
    _validate_uint("stream_id", item.stream_id, UINT32_MAX)
    _validate_uint("message_type", item.message_type, UINT8_MAX)
    _validate_uint("flags", item.flags, UINT8_MAX)
    _validate_uint("acknowledgement", item.acknowledgement, UINT32_MAX)
    if item.sequence is not None:
        _validate_uint("sequence", item.sequence, UINT32_MAX)
    if not isinstance(item.payload, bytes):
        raise TypeError("payload must be bytes")


class Multiplexer:
    def __init__(self) -> None:
        self._condition = Condition()
        self._queue: list[tuple[int, int, OutboundItem]] = []
        self._pointer_items: dict[int, tuple[int, OutboundItem]] = {}
        self._order = itertools.count()
        self._last_sequences: dict[int, int] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def reserve_sequence(self, stream_id: int) -> int:
        _validate_uint("stream_id", stream_id, UINT32_MAX)
        with self._condition:
            self._ensure_open()
            return self._reserve_sequence_locked(stream_id)

    def _reserve_sequence_locked(self, stream_id: int) -> int:
        previous = self._last_sequences.get(stream_id, 0)
        if previous >= UINT32_MAX:
            raise OverflowError(f"sequence exhausted for stream {stream_id}")
        sequence = previous + 1
        self._last_sequences[stream_id] = sequence
        return sequence

    def enqueue(self, item: OutboundItem) -> OutboundItem:
        _validate_item(item)
        with self._condition:
            self._ensure_open()
            if item.sequence is None:
                item = replace(
                    item,
                    sequence=self._reserve_sequence_locked(item.stream_id),
                )
            order = next(self._order)
            heapq.heappush(self._queue, (int(item.priority), order, item))
            self._condition.notify()
            return item

    def enqueue_pointer(
        self,
        stream_id: int,
        payload: bytes,
        *,
        message_type: int = 0,
        flags: int = 0,
        acknowledgement: int = 0,
        sequence: int | None = None,
    ) -> OutboundItem:
        item = OutboundItem(
            Priority.MOTION,
            stream_id,
            payload,
            message_type,
            flags,
            acknowledgement,
            sequence,
        )
        _validate_item(item)
        with self._condition:
            self._ensure_open()
            if item.sequence is None:
                item = replace(
                    item,
                    sequence=self._reserve_sequence_locked(item.stream_id),
                )
            self._pointer_items[stream_id] = (next(self._order), item)
            self._condition.notify()
            return item

    def pop(self, timeout: float | None = None) -> OutboundItem | None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while not self._queue and not self._pointer_items and not self._closed:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

            if self._closed:
                return None

            pointer_stream: int | None = None
            pointer_order: int | None = None
            pointer_item: OutboundItem | None = None
            if self._pointer_items:
                pointer_stream, (pointer_order, pointer_item) = min(
                    self._pointer_items.items(),
                    key=lambda entry: entry[1][0],
                )

            if self._queue and (
                pointer_item is None
                or self._queue[0][0] <= int(pointer_item.priority)
            ):
                return heapq.heappop(self._queue)[2]

            if pointer_stream is None or pointer_order is None or pointer_item is None:
                return None
            del self._pointer_items[pointer_stream]
            return pointer_item

    def empty(self) -> bool:
        with self._condition:
            return not self._queue and not self._pointer_items

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._queue.clear()
            self._pointer_items.clear()
            self._condition.notify_all()

    def _ensure_open(self) -> None:
        if self._closed:
            raise MultiplexerClosed("multiplexer is closed")


__all__ = [
    "Multiplexer",
    "MultiplexerClosed",
    "OutboundItem",
    "Priority",
]
