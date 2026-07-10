"""Thread-safe priority scheduling for one shared serial link."""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, replace
from enum import IntEnum
from threading import Condition

from shooklink.protocol.framing import MAX_PAYLOAD_SIZE

UINT8_MAX = (1 << 8) - 1
UINT32_MAX = (1 << 32) - 1
DEFAULT_MAX_ITEMS = 4_096
DEFAULT_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_PRIORITY_BURST = 32
DEFAULT_MAX_TRACKED_STREAMS = 4_096


class MultiplexerClosed(RuntimeError):
    """Raised when work is submitted after a multiplexer closes."""


class QueueFullError(RuntimeError):
    """Raised when bounded outbound capacity is exhausted."""


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
    if len(item.payload) > MAX_PAYLOAD_SIZE:
        raise ValueError(f"payload cannot exceed {MAX_PAYLOAD_SIZE} bytes")


class Multiplexer:
    def __init__(
        self,
        *,
        max_items: int = DEFAULT_MAX_ITEMS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_priority_burst: int = DEFAULT_MAX_PRIORITY_BURST,
        max_tracked_streams: int = DEFAULT_MAX_TRACKED_STREAMS,
    ) -> None:
        if type(max_items) is not int or max_items <= 0:
            raise ValueError("max_items must be positive")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if type(max_priority_burst) is not int or max_priority_burst <= 0:
            raise ValueError("max_priority_burst must be positive")
        if type(max_tracked_streams) is not int or max_tracked_streams <= 0:
            raise ValueError("max_tracked_streams must be positive")
        self._condition = Condition()
        self._queue: list[tuple[int, int, OutboundItem]] = []
        self._pointer_items: dict[int, tuple[int, OutboundItem]] = {}
        self._order = itertools.count()
        self._last_sequences: dict[int, int] = {}
        self._stream_priorities: dict[int, Priority] = {}
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._max_priority_burst = max_priority_burst
        self._max_tracked_streams = max_tracked_streams
        self._queued_bytes = 0
        self._priority_streak = 0
        self._fair_priority_cursor = int(Priority.NORMAL)
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def queued_items(self) -> int:
        with self._condition:
            return len(self._queue) + len(self._pointer_items)

    @property
    def queued_bytes(self) -> int:
        with self._condition:
            return self._queued_bytes

    @property
    def tracked_streams(self) -> int:
        with self._condition:
            return self._tracked_stream_count_locked()

    def reserve_sequence(self, stream_id: int) -> int:
        _validate_uint("stream_id", stream_id, UINT32_MAX)
        with self._condition:
            self._ensure_open()
            self._ensure_stream_capacity_locked(stream_id)
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
            self._validate_stream_priority_locked(item)
            self._ensure_capacity_locked(1, len(item.payload))
            self._ensure_stream_capacity_locked(item.stream_id)
            if item.sequence is None:
                item = replace(
                    item,
                    sequence=self._reserve_sequence_locked(item.stream_id),
                )
            else:
                self._record_explicit_sequence_locked(item.stream_id, item.sequence)
            self._stream_priorities.setdefault(item.stream_id, item.priority)
            order = next(self._order)
            heapq.heappush(self._queue, (int(item.priority), order, item))
            self._queued_bytes += len(item.payload)
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
            self._validate_stream_priority_locked(item)
            previous = self._pointer_items.get(stream_id)
            previous_size = 0 if previous is None else len(previous[1].payload)
            item_delta = 1 if previous is None else 0
            self._ensure_capacity_locked(
                item_delta,
                len(item.payload) - previous_size,
            )
            self._ensure_stream_capacity_locked(item.stream_id)
            if item.sequence is None:
                item = replace(
                    item,
                    sequence=self._reserve_sequence_locked(item.stream_id),
                )
            else:
                self._record_explicit_sequence_locked(item.stream_id, item.sequence)
            self._stream_priorities.setdefault(item.stream_id, item.priority)
            self._pointer_items[stream_id] = (next(self._order), item)
            self._queued_bytes += len(item.payload) - previous_size
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

            available = self._available_priorities_locked()
            best_priority = min(available)
            lower_priorities = [
                priority for priority in available if priority > best_priority
            ]
            if (
                lower_priorities
                and self._priority_streak >= self._max_priority_burst
            ):
                selected_priority = self._next_fair_priority(lower_priorities)
                self._priority_streak = 0
            else:
                selected_priority = best_priority
                if lower_priorities:
                    self._priority_streak += 1
                else:
                    self._priority_streak = 0
            return self._pop_priority_locked(selected_priority)

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
            self._last_sequences.clear()
            self._stream_priorities.clear()
            self._queued_bytes = 0
            self._condition.notify_all()

    def release_stream(self, stream_id: int) -> None:
        _validate_uint("stream_id", stream_id, UINT32_MAX)
        with self._condition:
            if stream_id in self._pointer_items or any(
                item.stream_id == stream_id for _priority, _order, item in self._queue
            ):
                raise ValueError(f"stream {stream_id} still has queued work")
            self._last_sequences.pop(stream_id, None)
            self._stream_priorities.pop(stream_id, None)

    def _ensure_open(self) -> None:
        if self._closed:
            raise MultiplexerClosed("multiplexer is closed")

    def _validate_stream_priority_locked(self, item: OutboundItem) -> None:
        priority = self._stream_priorities.get(item.stream_id)
        if priority is not None and priority is not item.priority:
            raise ValueError(
                f"stream {item.stream_id} priority cannot change from "
                f"{priority.name} to {item.priority.name}"
            )

    def _ensure_capacity_locked(self, item_delta: int, byte_delta: int) -> None:
        if self.queued_items + item_delta > self._max_items:
            raise QueueFullError("outbound item capacity is full")
        if self._queued_bytes + byte_delta > self._max_bytes:
            raise QueueFullError("outbound byte capacity is full")

    def _ensure_stream_capacity_locked(self, stream_id: int) -> None:
        if stream_id in self._last_sequences or stream_id in self._stream_priorities:
            return
        if self._tracked_stream_count_locked() >= self._max_tracked_streams:
            raise QueueFullError("tracked stream capacity is full")

    def _tracked_stream_count_locked(self) -> int:
        return len(self._last_sequences) + sum(
            stream_id not in self._last_sequences
            for stream_id in self._stream_priorities
        )

    def _record_explicit_sequence_locked(self, stream_id: int, sequence: int) -> None:
        previous = self._last_sequences.get(stream_id, 0)
        if sequence > previous:
            self._last_sequences[stream_id] = sequence

    def _available_priorities_locked(self) -> list[int]:
        priorities = {entry[0] for entry in self._queue}
        if self._pointer_items:
            priorities.add(int(Priority.MOTION))
        return sorted(priorities)

    def _next_fair_priority(self, available: list[int]) -> int:
        selected = next(
            (
                priority
                for priority in available
                if priority >= self._fair_priority_cursor
            ),
            available[0],
        )
        self._fair_priority_cursor = selected + 1
        if self._fair_priority_cursor > int(Priority.FILE):
            self._fair_priority_cursor = int(Priority.NORMAL)
        return selected

    def _pop_priority_locked(self, priority: int) -> OutboundItem:
        candidates: list[tuple[int, str, int, OutboundItem]] = []
        for index, (item_priority, order, item) in enumerate(self._queue):
            if item_priority == priority:
                candidates.append((order, "queue", index, item))
        if priority == int(Priority.MOTION):
            for stream_id, (order, item) in self._pointer_items.items():
                candidates.append((order, "pointer", stream_id, item))
        _order, source, key, item = min(candidates, key=lambda candidate: candidate[0])
        if source == "queue":
            last = self._queue.pop()
            if key < len(self._queue):
                self._queue[key] = last
                heapq.heapify(self._queue)
        else:
            del self._pointer_items[key]
        self._queued_bytes -= len(item.payload)
        self._condition.notify_all()
        return item


__all__ = [
    "Multiplexer",
    "MultiplexerClosed",
    "OutboundItem",
    "Priority",
    "QueueFullError",
]
