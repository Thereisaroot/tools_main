"""Reader/writer lifecycle for a framed serial connection."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import Protocol

from shooklink.protocol.framing import Frame, FrameParser, encode_frame
from shooklink.transport.multiplexer import (
    Multiplexer,
    MultiplexerClosed,
    OutboundItem,
)

logger = logging.getLogger(__name__)


class LinkClosedError(RuntimeError):
    """Raised when a serial link cannot accept more work."""


class LinkCloseError(RuntimeError):
    """Raised when the serial endpoint cannot be closed cleanly."""


class LinkCloseTimeout(LinkCloseError):
    """Raised when serial worker threads do not stop before the deadline."""


class _LinkState(Enum):
    NEW = auto()
    STARTING = auto()
    ACTIVE = auto()
    STOPPING = auto()
    CLOSED = auto()


class SerialEndpoint(Protocol):
    def read(self, size: int) -> bytes: ...

    def write(self, data: bytes | memoryview) -> int: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class SerialLink:
    def __init__(
        self,
        endpoint: SerialEndpoint,
        on_frame: Callable[[Frame], None],
        on_disconnect: Callable[[BaseException | None], None],
        *,
        read_size: int = 4096,
        multiplexer: Multiplexer | None = None,
    ) -> None:
        if type(read_size) is not int or read_size <= 0:
            raise ValueError("read_size must be positive")
        self._endpoint = endpoint
        self._on_frame = on_frame
        self._on_disconnect = on_disconnect
        self._read_size = read_size
        self._multiplexer = multiplexer or Multiplexer()
        self._parser = FrameParser()
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._stop_finalized = threading.Event()
        self._disconnect_callback_completed = threading.Event()
        self._state = _LinkState.NEW
        self._endpoint_close_error: BaseException | None = None
        self._finalizer_start_error: BaseException | None = None
        self._terminal_cause: BaseException | None = None
        self._finalization_in_progress = False
        self._disconnect_notified = False
        self._disconnect_callback_thread_id: int | None = None
        self._reader_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._finalizer_thread: threading.Thread | None = None
        self._started_threads: list[threading.Thread] = []

    @classmethod
    def open_port(
        cls,
        port: str,
        baud_rate: int,
        on_frame: Callable[[Frame], None],
        on_disconnect: Callable[[BaseException | None], None],
    ) -> SerialLink:
        import serial

        endpoint = serial.serial_for_url(
            port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            write_timeout=1.0,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        return cls(endpoint, on_frame, on_disconnect)

    @property
    def parser(self) -> FrameParser:
        return self._parser

    @property
    def threads_alive(self) -> bool:
        with self._lifecycle_lock:
            threads = tuple(self._started_threads)
        return any(thread.is_alive() for thread in threads)

    @property
    def closed(self) -> bool:
        with self._lifecycle_lock:
            return self._state in (_LinkState.STOPPING, _LinkState.CLOSED)

    def start(self) -> None:
        startup_error: BaseException | None = None
        with self._lifecycle_lock:
            if self._state is not _LinkState.NEW:
                if self._state in (_LinkState.STOPPING, _LinkState.CLOSED):
                    raise LinkClosedError("serial link is closed")
                raise RuntimeError("serial link can only be started once")
            self._state = _LinkState.STARTING
            if self._multiplexer.closed:
                startup_error = LinkClosedError("outbound multiplexer is closed")
            else:
                self._reader_thread = threading.Thread(
                    target=self._read_loop,
                    name="shooklink-serial-reader",
                    daemon=True,
                )
                self._writer_thread = threading.Thread(
                    target=self._write_loop,
                    name="shooklink-serial-writer",
                    daemon=True,
                )
                try:
                    self._writer_thread.start()
                    self._started_threads.append(self._writer_thread)
                    self._reader_thread.start()
                    self._started_threads.append(self._reader_thread)
                    self._state = _LinkState.ACTIVE
                except BaseException as error:
                    startup_error = error

        if startup_error is not None:
            self._request_stop(
                startup_error,
                allow_synchronous_fallback=True,
            )
            self._stop_finalized.wait(2.0)
            self.wait_closed(2.0)
            self._disconnect_callback_completed.wait(2.0)
            raise startup_error

    def reserve_sequence(self, stream_id: int) -> int:
        self._ensure_active()
        try:
            return self._multiplexer.reserve_sequence(stream_id)
        except MultiplexerClosed as error:
            raise LinkClosedError("serial link is closed") from error

    def cancel_sequence(self, stream_id: int, sequence: int) -> None:
        self._ensure_active()
        try:
            self._multiplexer.cancel_sequence(stream_id, sequence)
        except MultiplexerClosed as error:
            raise LinkClosedError("serial link is closed") from error

    def send(self, item: OutboundItem) -> OutboundItem:
        self._ensure_active()
        try:
            return self._multiplexer.enqueue(item)
        except MultiplexerClosed as error:
            raise LinkClosedError("serial link is closed") from error

    def send_pointer(self, stream_id: int, payload: bytes, **metadata) -> OutboundItem:
        self._ensure_active()
        try:
            return self._multiplexer.enqueue_pointer(stream_id, payload, **metadata)
        except MultiplexerClosed as error:
            raise LinkClosedError("serial link is closed") from error

    def close(self, timeout: float = 2.0) -> None:
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        deadline = time.monotonic() + timeout
        current = threading.current_thread()
        with self._lifecycle_lock:
            called_from_worker = current in self._started_threads
        self._request_stop(
            None,
            allow_synchronous_fallback=called_from_worker,
        )
        remaining = max(0.0, deadline - time.monotonic())
        if not self._stop_finalized.wait(remaining):
            timeout_error = LinkCloseTimeout(
                "serial endpoint shutdown did not finish"
            )
            with self._lifecycle_lock:
                start_error = self._finalizer_start_error
            if start_error is not None:
                raise timeout_error from start_error
            raise timeout_error
        with self._lifecycle_lock:
            is_callback_thread = (
                self._disconnect_callback_thread_id == threading.get_ident()
            )
            close_error = self._endpoint_close_error
        if not is_callback_thread:
            remaining = max(0.0, deadline - time.monotonic())
            if not self.wait_closed(remaining):
                raise LinkCloseTimeout("serial worker threads did not stop")
            remaining = max(0.0, deadline - time.monotonic())
            if not self._disconnect_callback_completed.wait(remaining):
                raise LinkCloseTimeout("serial disconnect callback did not finish")
        if close_error is not None:
            raise LinkCloseError(str(close_error)) from close_error

    def wait_closed(self, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        current = threading.current_thread()
        with self._lifecycle_lock:
            threads = tuple(self._started_threads)
        for thread in threads:
            if thread is current:
                continue
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(remaining)
        return not any(
            thread is not current and thread.is_alive()
            for thread in threads
        )

    def _ensure_active(self) -> None:
        with self._lifecycle_lock:
            if self._state is not _LinkState.ACTIVE:
                raise LinkClosedError("serial link is not active")

    def _read_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                data = self._endpoint.read(self._read_size)
                if not data:
                    continue
                for frame in self._parser.feed(data):
                    if self._stop_event.is_set():
                        return
                    self._on_frame(frame)
        except BaseException as error:
            if not self._stop_event.is_set():
                self._request_stop(error, allow_synchronous_fallback=True)

    def _write_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                item = self._multiplexer.pop(timeout=0.1)
                if item is None:
                    if self._multiplexer.closed:
                        raise MultiplexerClosed("outbound multiplexer was closed")
                    continue
                try:
                    if item.sequence is None:
                        raise RuntimeError("multiplexer returned an item without a sequence")
                    frame = Frame(
                        message_type=item.message_type,
                        flags=item.flags,
                        priority=int(item.priority),
                        stream_id=item.stream_id,
                        sequence=item.sequence,
                        acknowledgement=item.acknowledgement,
                        payload=item.payload,
                    )
                    self._write_all(encode_frame(frame))
                    if item.on_written is not None:
                        try:
                            item.on_written()
                        except BaseException:
                            logger.exception("serial write completion callback failed")
                finally:
                    self._multiplexer.task_done(item)
        except BaseException as error:
            if not self._stop_event.is_set():
                self._request_stop(error, allow_synchronous_fallback=True)

    def _write_all(self, encoded: bytes) -> None:
        view = memoryview(encoded)
        offset = 0
        while offset < len(view) and not self._stop_event.is_set():
            written = self._endpoint.write(view[offset:])
            if type(written) is not int or written <= 0:
                raise OSError("serial endpoint made no write progress")
            if written > len(view) - offset:
                raise OSError("serial endpoint reported an invalid write length")
            offset += written
        if offset != len(view):
            raise LinkClosedError("serial link closed during write")

    def _request_stop(
        self,
        error: BaseException | None,
        *,
        allow_synchronous_fallback: bool = False,
    ) -> None:
        use_synchronous_fallback = False
        with self._lifecycle_lock:
            if self._state is _LinkState.CLOSED:
                return
            if self._state is not _LinkState.STOPPING:
                self._state = _LinkState.STOPPING
                self._terminal_cause = error
                self._stop_event.set()
                self._multiplexer.close()
            elif self._finalization_in_progress:
                return
            self._finalization_in_progress = True
            try:
                self._finalizer_thread = threading.Thread(
                    target=self._finalize_stop,
                    name="shooklink-serial-finalizer",
                    daemon=True,
                )
                self._finalizer_thread.start()
                self._finalizer_start_error = None
            except BaseException as start_error:
                self._finalizer_thread = None
                self._finalizer_start_error = start_error
                if self._terminal_cause is None:
                    self._terminal_cause = start_error
                use_synchronous_fallback = allow_synchronous_fallback
                if not use_synchronous_fallback:
                    self._finalization_in_progress = False

        if use_synchronous_fallback:
            self._finalize_stop()

    def _finalize_stop(self) -> None:
        close_error: BaseException | None = None
        try:
            self._endpoint.close()
        except BaseException as endpoint_error:
            close_error = endpoint_error

        with self._lifecycle_lock:
            if close_error is not None:
                self._endpoint_close_error = close_error
                if self._terminal_cause is None:
                    self._terminal_cause = close_error
            self._finalization_in_progress = False
            self._state = _LinkState.CLOSED
            callback_error = self._terminal_cause
            callback = None
            if not self._disconnect_notified:
                self._disconnect_notified = True
                callback = self._on_disconnect

        self._stop_finalized.set()
        if callback is not None:
            self._invoke_disconnect(callback, callback_error)
        else:
            self._disconnect_callback_completed.set()

    def _invoke_disconnect(
        self,
        callback: Callable[[BaseException | None], None],
        error: BaseException | None,
    ) -> None:
        with self._lifecycle_lock:
            self._disconnect_callback_thread_id = threading.get_ident()
        try:
            callback(error)
        except BaseException:
            logger.exception("serial disconnect callback failed")
        finally:
            with self._lifecycle_lock:
                self._disconnect_callback_thread_id = None
            self._disconnect_callback_completed.set()


__all__ = [
    "LinkCloseError",
    "LinkCloseTimeout",
    "LinkClosedError",
    "SerialEndpoint",
    "SerialLink",
]
