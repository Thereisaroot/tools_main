import threading
import time

import pytest

from shooklink.transport.multiplexer import Multiplexer, OutboundItem, Priority
from shooklink.transport.serial_link import (
    LinkCloseError,
    LinkCloseTimeout,
    LinkClosedError,
    SerialLink,
)


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


class BlockingBrokenWriteEndpoint(BrokenWriteEndpoint):
    def __init__(self):
        super().__init__()
        self.close_started = threading.Event()
        self.release_close = threading.Event()

    def close(self):
        self.close_started.set()
        self.release_close.wait(1)
        super().close()


class StuckReadEndpoint(MemoryEndpoint):
    def __init__(self):
        super().__init__()
        self.read_started = threading.Event()
        self.release_read = threading.Event()

    def read(self, _size):
        self.read_started.set()
        self.release_read.wait(2)
        return b""

    def close(self):
        with self._condition:
            self.close_calls += 1
            self._closed = True


class FailingCloseEndpoint(MemoryEndpoint):
    def close(self):
        super().close()
        raise OSError("close failed")


class BlockingFailingCloseEndpoint(MemoryEndpoint):
    def __init__(self):
        super().__init__()
        self.close_started = threading.Event()
        self.release_close = threading.Event()

    def close(self):
        self.close_started.set()
        self.release_close.wait(1)
        with self._condition:
            self.close_calls += 1
            self._closed = True
            self._condition.notify_all()
        raise OSError("delayed close failed")


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


def test_start_rejects_a_preclosed_multiplexer_and_closes_the_endpoint():
    endpoint = MemoryEndpoint()
    multiplexer = Multiplexer()
    multiplexer.close()
    disconnects = []
    link = SerialLink(
        endpoint,
        lambda frame: None,
        disconnects.append,
        multiplexer=multiplexer,
    )

    with pytest.raises(LinkClosedError):
        link.start()

    assert link.wait_closed(1)
    assert endpoint.closed
    assert len(disconnects) == 1
    assert isinstance(disconnects[0], LinkClosedError)


def test_externally_closed_multiplexer_stops_the_link_without_spinning():
    endpoint = MemoryEndpoint()
    endpoint.connect(endpoint)
    multiplexer = Multiplexer()
    disconnects = []
    link = SerialLink(
        endpoint,
        lambda frame: None,
        disconnects.append,
        multiplexer=multiplexer,
    )
    link.start()

    multiplexer.close()

    assert wait_for(lambda: link.closed)
    assert link.wait_closed(1)
    assert endpoint.closed
    assert len(disconnects) == 1
    assert disconnects[0] is not None


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
    left.send(OutboundItem(Priority.NORMAL, 4, b"second", message_type=45))
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
    left.send(OutboundItem(Priority.FILE, 7, b"next", message_type=22))

    assert wait_for(lambda: len(received) == 2)
    assert received[0].sequence == 99
    assert received[1].sequence == 100
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


def test_concurrent_start_and_close_never_join_unstarted_thread(monkeypatch):
    endpoint, _peer = endpoint_pair()
    disconnects = []
    link = SerialLink(endpoint, lambda frame: None, disconnects.append)
    original_start = threading.Thread.start
    reader_waiting = threading.Event()
    release_reader = threading.Event()
    errors = []

    def delayed_start(thread):
        if thread.name == "shooklink-serial-reader":
            reader_waiting.set()
            release_reader.wait(1)
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", delayed_start)
    starter = threading.Thread(
        target=lambda: _capture_error(link.start, errors),
        name="test-link-starter",
    )
    original_start(starter)
    assert reader_waiting.wait(1)
    closer = threading.Thread(
        target=lambda: _capture_error(link.close, errors),
        name="test-link-closer",
    )
    original_start(closer)
    time.sleep(0.02)
    release_reader.set()
    starter.join(1)
    closer.join(1)

    assert errors == []
    assert disconnects == [None]
    assert not link.threads_alive


def test_thread_start_failure_rolls_back_and_closes_endpoint(monkeypatch):
    endpoint, _peer = endpoint_pair()
    disconnects = []
    link = SerialLink(endpoint, lambda frame: None, disconnects.append)
    original_start = threading.Thread.start

    def failing_start(thread):
        if thread.name == "shooklink-serial-reader":
            raise RuntimeError("cannot start reader")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", failing_start)

    with pytest.raises(RuntimeError, match="cannot start reader"):
        link.start()

    assert link.wait_closed(1)
    assert endpoint.close_calls == 1
    assert isinstance(disconnects[0], RuntimeError)
    with pytest.raises(LinkClosedError):
        link.send(OutboundItem(Priority.NORMAL, 1, b"late"))


def test_finalizer_start_failure_is_retriable_without_breaking_deadline(
    monkeypatch,
):
    endpoint, _peer = endpoint_pair()
    disconnects = []
    link = SerialLink(endpoint, lambda frame: None, disconnects.append)
    original_start = threading.Thread.start

    failed_once = False

    def failing_start(thread):
        nonlocal failed_once
        if thread.name == "shooklink-serial-finalizer" and not failed_once:
            failed_once = True
            raise RuntimeError("cannot start finalizer")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", failing_start)

    started_at = time.monotonic()
    with pytest.raises(LinkCloseTimeout) as captured:
        link.close(timeout=0.05)

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert time.monotonic() - started_at < 0.15
    assert endpoint.close_calls == 0
    link.close(timeout=1)
    assert endpoint.close_calls == 1
    assert endpoint.closed
    assert isinstance(disconnects[0], RuntimeError)


def test_worker_error_finalizes_if_finalizer_thread_cannot_start(monkeypatch):
    endpoint = BrokenWriteEndpoint()
    endpoint.connect(MemoryEndpoint())
    disconnects = []
    disconnected = threading.Event()
    original_start = threading.Thread.start

    def failing_start(thread):
        if thread.name == "shooklink-serial-finalizer":
            raise RuntimeError("cannot start finalizer")
        return original_start(thread)

    def on_disconnect(error):
        disconnects.append(error)
        disconnected.set()

    monkeypatch.setattr(threading.Thread, "start", failing_start)
    link = SerialLink(endpoint, lambda frame: None, on_disconnect)
    link.start()
    link.send(OutboundItem(Priority.NORMAL, 1, b"fail"))

    assert disconnected.wait(1)
    assert endpoint.close_calls == 1
    assert len(disconnects) == 1
    assert isinstance(disconnects[0], OSError)
    assert link.wait_closed(1)


def test_worker_fallback_owns_finalization_against_concurrent_close(monkeypatch):
    endpoint = BlockingBrokenWriteEndpoint()
    endpoint.connect(MemoryEndpoint())
    disconnects = []
    original_start = threading.Thread.start
    failed_once = False

    def failing_start(thread):
        nonlocal failed_once
        if thread.name == "shooklink-serial-finalizer" and not failed_once:
            failed_once = True
            raise RuntimeError("cannot start finalizer")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", failing_start)
    link = SerialLink(endpoint, lambda frame: None, disconnects.append)
    link.start()
    link.send(OutboundItem(Priority.NORMAL, 1, b"fail"))
    assert endpoint.close_started.wait(1)
    close_errors = []
    closer = threading.Thread(target=lambda: _capture_error(link.close, close_errors))
    closer.start()
    time.sleep(0.02)
    endpoint.release_close.set()
    closer.join(1)

    assert close_errors == []
    assert endpoint.close_calls == 1
    assert len(disconnects) == 1


def test_first_stop_owns_disconnect_cause_during_concurrent_close():
    endpoint = BlockingBrokenWriteEndpoint()
    endpoint.connect(MemoryEndpoint())
    disconnects = []
    link = SerialLink(endpoint, lambda frame: None, disconnects.append)
    link.start()
    link.send(OutboundItem(Priority.NORMAL, 1, b"fail"))
    assert endpoint.close_started.wait(1)
    close_errors = []
    closer = threading.Thread(target=lambda: _capture_error(link.close, close_errors))
    closer.start()
    time.sleep(0.02)
    endpoint.release_close.set()
    closer.join(1)

    assert close_errors == []
    assert len(disconnects) == 1
    assert isinstance(disconnects[0], OSError)
    assert str(disconnects[0]) == "write failed"


def test_close_reports_worker_timeout_and_can_finish_after_unblock():
    endpoint = StuckReadEndpoint()
    endpoint.connect(MemoryEndpoint())
    link = SerialLink(endpoint, lambda frame: None, lambda error: None)
    link.start()
    assert endpoint.read_started.wait(1)

    with pytest.raises(LinkCloseTimeout):
        link.close(timeout=0.05)

    endpoint.release_read.set()
    assert link.wait_closed(1)


def test_close_failure_is_reported_to_callback_and_caller():
    endpoint = FailingCloseEndpoint()
    endpoint.connect(MemoryEndpoint())
    disconnects = []
    link = SerialLink(endpoint, lambda frame: None, disconnects.append)
    link.start()

    with pytest.raises(LinkCloseError, match="close failed"):
        link.close()

    assert isinstance(disconnects[0], OSError)


def test_concurrent_close_waits_for_endpoint_finalization_and_cause():
    endpoint = BlockingFailingCloseEndpoint()
    disconnects = []
    link = SerialLink(endpoint, lambda frame: None, disconnects.append)
    first_errors = []
    first_close = threading.Thread(
        target=lambda: _capture_error(link.close, first_errors),
    )
    first_close.start()
    assert endpoint.close_started.wait(1)

    with pytest.raises(LinkCloseTimeout):
        link.close(timeout=0.05)

    endpoint.release_close.set()
    first_close.join(1)
    assert len(first_errors) == 1
    assert isinstance(first_errors[0], LinkCloseError)
    assert isinstance(disconnects[0], OSError)
    with pytest.raises(LinkCloseError, match="delayed close failed"):
        link.close()


def test_initiating_close_timeout_includes_blocking_endpoint_close():
    endpoint = BlockingBrokenWriteEndpoint()
    link = SerialLink(endpoint, lambda frame: None, lambda error: None)
    started_at = time.monotonic()

    with pytest.raises(LinkCloseTimeout):
        link.close(timeout=0.05)

    assert time.monotonic() - started_at < 0.15
    endpoint.release_close.set()
    link.close(timeout=1)


def test_on_frame_callback_can_close_its_own_link():
    left_endpoint, right_endpoint = endpoint_pair()
    callback_errors = []
    callback_finished = threading.Event()
    holder = {}

    def on_frame(_frame):
        try:
            holder["right"].close()
        except BaseException as error:
            callback_errors.append(error)
        finally:
            callback_finished.set()

    left = SerialLink(left_endpoint, lambda frame: None, lambda error: None)
    right = SerialLink(right_endpoint, on_frame, lambda error: None)
    holder["right"] = right
    left.start()
    right.start()
    left.send(OutboundItem(Priority.NORMAL, 1, b"close peer"))

    assert callback_finished.wait(2)
    assert callback_errors == []
    assert right.wait_closed(1)
    left.close()


def test_on_disconnect_callback_can_close_its_own_link():
    endpoint = BrokenWriteEndpoint()
    endpoint.connect(MemoryEndpoint())
    callback_errors = []
    callback_finished = threading.Event()
    holder = {}

    def on_disconnect(_error):
        try:
            holder["link"].close()
        except BaseException as error:
            callback_errors.append(error)
        finally:
            callback_finished.set()

    link = SerialLink(endpoint, lambda frame: None, on_disconnect)
    holder["link"] = link
    link.start()
    link.send(OutboundItem(Priority.NORMAL, 1, b"fail"))

    assert callback_finished.wait(2)
    assert callback_errors == []
    assert link.wait_closed(1)


def test_external_close_waits_for_disconnect_callback_completion():
    endpoint, _peer = endpoint_pair()
    callback_started = threading.Event()
    release_callback = threading.Event()
    close_errors = []

    def on_disconnect(_error):
        callback_started.set()
        release_callback.wait(1)

    link = SerialLink(endpoint, lambda frame: None, on_disconnect)
    closer = threading.Thread(target=lambda: _capture_error(link.close, close_errors))
    closer.start()
    assert callback_started.wait(1)
    time.sleep(0.02)

    assert closer.is_alive()
    release_callback.set()
    closer.join(1)
    assert close_errors == []


def test_on_frame_and_on_disconnect_can_both_close_without_cycle():
    left_endpoint, right_endpoint = endpoint_pair()
    errors = []
    frame_finished = threading.Event()
    disconnect_finished = threading.Event()
    holder = {}

    def on_frame(_frame):
        try:
            holder["right"].close()
        except BaseException as error:
            errors.append(error)
        finally:
            frame_finished.set()

    def on_disconnect(_error):
        try:
            holder["right"].close()
        except BaseException as error:
            errors.append(error)
        finally:
            disconnect_finished.set()

    left = SerialLink(left_endpoint, lambda frame: None, lambda error: None)
    right = SerialLink(right_endpoint, on_frame, on_disconnect)
    holder["right"] = right
    left.start()
    right.start()
    left.send(OutboundItem(Priority.NORMAL, 1, b"close"))

    assert disconnect_finished.wait(3)
    assert frame_finished.wait(3)
    assert errors == []
    assert right.wait_closed(1)
    left.close()


def _capture_error(callback, errors):
    try:
        callback()
    except BaseException as error:
        errors.append(error)
