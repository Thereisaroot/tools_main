"""Reader/writer lifecycle for a framed serial connection."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

from shooklink.protocol.framing import Frame, FrameParser, encode_frame
from shooklink.transport.multiplexer import (
    Multiplexer,
    MultiplexerClosed,
    OutboundItem,
)


class LinkClosedError(RuntimeError):
    """Raised when a serial link cannot accept more work."""


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
        if read_size <= 0:
            raise ValueError("read_size must be positive")
        self._endpoint = endpoint
        self._on_frame = on_frame
        self._on_disconnect = on_disconnect
        self._read_size = read_size
        self._multiplexer = multiplexer or Multiplexer()
        self._parser = FrameParser()
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._started = False
        self._closed = False
        self._endpoint_closed = False
        self._disconnect_notified = False
        self._reader_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None

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
        threads = (self._reader_thread, self._writer_thread)
        return any(thread is not None and thread.is_alive() for thread in threads)

    @property
    def closed(self) -> bool:
        with self._lifecycle_lock:
            return self._closed

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("serial link can only be started once")
            if self._closed:
                raise LinkClosedError("serial link is closed")
            self._started = True
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
            reader = self._reader_thread
            writer = self._writer_thread
        writer.start()
        reader.start()

    def reserve_sequence(self, stream_id: int) -> int:
        with self._lifecycle_lock:
            if not self._started or self._closed:
                raise LinkClosedError("serial link is not active")
        try:
            return self._multiplexer.reserve_sequence(stream_id)
        except MultiplexerClosed as error:
            raise LinkClosedError("serial link is closed") from error

    def send(self, item: OutboundItem) -> OutboundItem:
        with self._lifecycle_lock:
            if not self._started or self._closed:
                raise LinkClosedError("serial link is not active")
        try:
            return self._multiplexer.enqueue(item)
        except MultiplexerClosed as error:
            raise LinkClosedError("serial link is closed") from error

    def send_pointer(self, stream_id: int, payload: bytes, **metadata) -> OutboundItem:
        with self._lifecycle_lock:
            if not self._started or self._closed:
                raise LinkClosedError("serial link is not active")
        try:
            return self._multiplexer.enqueue_pointer(stream_id, payload, **metadata)
        except MultiplexerClosed as error:
            raise LinkClosedError("serial link is closed") from error

    def close(self) -> None:
        self._request_stop(None)
        self.wait_closed(2.0)

    def wait_closed(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        current = threading.current_thread()
        for thread in (self._writer_thread, self._reader_thread):
            if thread is None or thread is current:
                continue
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(remaining)
        return not self.threads_alive

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
                self._request_stop(error)

    def _write_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                item = self._multiplexer.pop(timeout=0.1)
                if item is None:
                    continue
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
        except BaseException as error:
            if not self._stop_event.is_set():
                self._request_stop(error)

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

    def _request_stop(self, error: BaseException | None) -> None:
        endpoint_to_close = None
        with self._lifecycle_lock:
            if not self._closed:
                self._closed = True
                self._stop_event.set()
                self._multiplexer.close()
            if not self._endpoint_closed:
                self._endpoint_closed = True
                endpoint_to_close = self._endpoint

        close_error: BaseException | None = None
        if endpoint_to_close is not None:
            try:
                endpoint_to_close.close()
            except BaseException as endpoint_error:
                close_error = endpoint_error

        callback = None
        callback_error = error if error is not None else close_error
        with self._lifecycle_lock:
            if not self._disconnect_notified:
                self._disconnect_notified = True
                callback = self._on_disconnect
        if callback is not None:
            try:
                callback(callback_error)
            except BaseException:
                pass


__all__ = ["LinkClosedError", "SerialEndpoint", "SerialLink"]
