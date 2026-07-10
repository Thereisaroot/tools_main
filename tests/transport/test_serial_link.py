import threading
import time

import pytest

from shooklink.transport.multiplexer import OutboundItem, Priority
from shooklink.transport.serial_link import LinkClosedError, SerialLink


class MemoryEndpoint:
    def __init__(self, *, max_read=7, max_write=5):
        self.max_read = max_read
        self.max_write = max_write
        self.peer = None
        self._buffer = bytearray()
        self._condition = threading.Condition()
        self._closed = False
        self.close_calls = 0

    def connect(self, peer):
        self.peer = peer

    @property
    def closed(self):
        with self._condition:
            return self._closed

    def read(self, size):
        with self._condition:
            self._condition.wait_for(lambda: self._buffer or self._closed, timeout=0.05)
            if self._closed:
                raise OSError("endpoint closed")
            if not self._buffer:
                return b""
            count = min(size, self.max_read, len(self._buffer))
            result = bytes(self._buffer[:count])
            del self._buffer[:count]
            return result

    def write(self, data):
        if self.closed or self.peer is None or self.peer.closed:
            raise OSError("endpoint unavailable")
        count = min(len(data), self.max_write)
        with self.peer._condition:
            self.peer._buffer.extend(bytes(data[:count]))
            self.peer._condition.notify_all()
        return count

    def flush(self):
        return None

    def close(self):
        with self._condition:
            self.close_calls += 1
            self._closed = True
            self._condition.notify_all()


class BrokenWriteEndpoint(MemoryEndpoint):
    def write(self, _data):
        raise OSError("write failed")


def endpoint_pair(**kwargs):
    left = MemoryEndpoint(**kwargs)
    right = MemoryEndpoint(**kwargs)
    left.connect(right)
    right.connect(left)
    return left, right


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_links_exchange_fragmented_frames_and_sequence_per_stream():
    left_endpoint, right_endpoint = endpoint_pair(max_read=3, max_write=2)
    received = []
    left_disconnects = []
    right_disconnects = []
    left = SerialLink(left_endpoint, lambda frame: None, left_disconnects.append)
    right = SerialLink(right_endpoint, received.append, right_disconnects.append)
    left.start()
    right.start()

    left.send(OutboundItem(Priority.NORMAL, 4, b"first", message_type=10))
    left.send(OutboundItem(Priority.INTERACTIVE, 4, b"second", message_type=45))
    left.send(OutboundItem(Priority.NORMAL, 9, b"other", message_type=11))

    assert wait_for(lambda: len(received) == 3)
    by_payload = {frame.payload: frame for frame in received}
    assert by_payload[b"first"].sequence == 1
    assert by_payload[b"second"].sequence == 2
    assert by_payload[b"other"].sequence == 1
    assert by_payload[b"second"].message_type == 45

    left.close()
    right.close()

    assert left.wait_closed(1)
    assert right.wait_closed(1)
    assert left_disconnects == [None]
    assert right_disconnects == [None]
    assert left_endpoint.close_calls == 1
    assert right_endpoint.close_calls == 1
    assert not left.threads_alive
    assert not right.threads_alive


def test_explicit_sequence_is_preserved_for_retransmission():
    left_endpoint, right_endpoint = endpoint_pair()
    received = []
    left = SerialLink(left_endpoint, lambda frame: None, lambda error: None)
    right = SerialLink(right_endpoint, received.append, lambda error: None)
    left.start()
    right.start()

    left.send(OutboundItem(Priority.FILE, 7, b"retry", message_type=22, sequence=99))

    assert wait_for(lambda: len(received) == 1)
    assert received[0].sequence == 99
    left.close()
    right.close()


def test_writer_error_notifies_disconnect_once_and_stops_threads():
    endpoint = BrokenWriteEndpoint()
    endpoint.connect(MemoryEndpoint())
    disconnects = []
    disconnected = threading.Event()

    def on_disconnect(error):
        disconnects.append(error)
        disconnected.set()

    link = SerialLink(endpoint, lambda frame: None, on_disconnect)
    link.start()
    link.send(OutboundItem(Priority.NORMAL, 1, b"fail"))

    assert disconnected.wait(2)
    assert isinstance(disconnects[0], OSError)
    link.close()
    assert link.wait_closed(1)
    assert len(disconnects) == 1
    assert endpoint.close_calls == 1


def test_close_is_idempotent_and_send_after_close_fails():
    endpoint, _peer = endpoint_pair()
    disconnects = []
    link = SerialLink(endpoint, lambda frame: None, disconnects.append)
    link.start()

    link.close()
    link.close()

    with pytest.raises(LinkClosedError):
        link.send(OutboundItem(Priority.NORMAL, 1, b"late"))
    assert disconnects == [None]
    assert endpoint.close_calls == 1


def test_start_is_single_use():
    endpoint, _peer = endpoint_pair()
    link = SerialLink(endpoint, lambda frame: None, lambda error: None)
    link.start()

    with pytest.raises(RuntimeError):
        link.start()

    link.close()
