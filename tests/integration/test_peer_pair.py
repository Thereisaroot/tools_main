from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from shooklink.core import CoreError, CoreState, ShookLinkCore
from shooklink.files.service import CHUNK_SIZE, WINDOW_SIZE
from shooklink.input.backend import BaseInputBackend, PermissionStatus
from shooklink.input.events import KeyAction, KeyEvent, PointerMotionEvent
from shooklink.input.service import InputSessionState
from shooklink.input.topology import Monitor, Rect
from shooklink.protocol.crypto import IdentityStore, TrustStore
from shooklink.protocol.framing import SECURE_FLAG, Frame, pack_frame_header
from shooklink.protocol.messages import Message, MessageType, encode_message
from shooklink.transport.multiplexer import Priority
from shooklink.transport.serial_link import LinkCloseTimeout


class MemoryEndpoint:
    def __init__(self, *, max_read=257, max_write=509):
        self.max_read = max_read
        self.max_write = max_write
        self.peer = None
        self._buffer = bytearray()
        self._condition = threading.Condition()
        self._closed = False

    def connect(self, peer):
        self.peer = peer

    def read(self, size):
        with self._condition:
            self._condition.wait_for(
                lambda: self._buffer or self._closed,
                timeout=0.02,
            )
            if self._closed:
                raise OSError("endpoint closed")
            if not self._buffer:
                return b""
            count = min(size, self.max_read, len(self._buffer))
            result = bytes(self._buffer[:count])
            del self._buffer[:count]
            return result

    def write(self, data):
        if self._closed or self.peer is None or self.peer._closed:
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
            self._closed = True
            self._condition.notify_all()


class BlockingCloseEndpoint(MemoryEndpoint):
    def __init__(self):
        super().__init__()
        self.close_started = threading.Event()
        self.release_close = threading.Event()

    def close(self):
        self.close_started.set()
        self.release_close.wait(2)
        super().close()


class GatedMemoryEndpoint(MemoryEndpoint):
    def __init__(self):
        super().__init__()
        self.accepting = True
        self.dropped_bytes = 0

    def write(self, data):
        if self.peer is not None and not self.peer.accepting:
            count = min(len(data), self.max_write)
            self.dropped_bytes += count
            return count
        return super().write(data)


def endpoint_pair():
    left = MemoryEndpoint()
    right = MemoryEndpoint()
    left.connect(right)
    right.connect(left)
    return left, right


def wait_for(predicate, *, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class FakeInputBackend(BaseInputBackend):
    def __init__(self, *, position=(99, 50)):
        super().__init__()
        self.position = position
        self.injected = []
        self.warps = []
        self.capture_starts = []
        self.capture_stops = 0

    def permission_status(self):
        return PermissionStatus(True, True, "ready")

    def monitors(self):
        return (Monitor("display", Rect(0, 0, 100, 100)),)

    def cursor_position(self):
        return self.position

    def warp_cursor(self, x, y):
        self.position = (x, y)
        self.warps.append((x, y))

    def _start_native_capture(self, suppress):
        self.capture_starts.append(suppress)

    def _stop_native_capture(self):
        self.capture_stops += 1

    def _inject_native(self, event):
        self.injected.append(event)
        if hasattr(event, "position"):
            self.position = event.position

    def capture(self, event):
        self.emit_captured(event)


class FakeProcess:
    def __init__(self, on_output, on_exit):
        self.on_output = on_output
        self.on_exit = on_exit
        self.started = []
        self.writes = []
        self.resizes = []
        self.terminated = False

    def start(self, columns, rows):
        self.started.append((columns, rows))

    def write(self, data):
        self.writes.append(data)

    def resize(self, columns, rows):
        self.resizes.append((columns, rows))

    def terminate(self):
        self.terminated = True

    def is_running(self):
        return bool(self.started) and not self.terminated

    def emit_output(self, data):
        self.on_output(data)


class FakeProcessFactory:
    def __init__(self):
        self.processes = []

    def __call__(self, on_output, on_exit, term):
        process = FakeProcess(on_output, on_exit)
        self.processes.append(process)
        return process


def build_core(tmp_path, name, *, auto_edge_enabled=False):
    backend = FakeInputBackend()
    processes = FakeProcessFactory()
    core = ShookLinkCore(
        identity=IdentityStore(tmp_path / f"{name}.identity").load_or_create(),
        trust_store=TrustStore(tmp_path / f"{name}.trust.json"),
        download_dir=tmp_path / f"{name}-downloads",
        local_peer_id=name,
        input_backend=backend,
        process_factory=processes,
        auto_edge_enabled=auto_edge_enabled,
    )
    return core, backend, processes


def approve_pair(left, right):
    left_snapshot = left.snapshot
    right_snapshot = right.snapshot
    left.approve_peer(left_snapshot.connection_id, left_snapshot.fingerprint)
    right.approve_peer(right_snapshot.connection_id, right_snapshot.fingerprint)
    assert wait_for(
        lambda: left.snapshot.state is CoreState.READY
        and right.snapshot.state is CoreState.READY
    )


def make_secure_frame(sender, outer_type, inner_message, priority=Priority.NORMAL):
    connection = sender._connection
    assert connection is not None
    stream_id = int(outer_type) * 4 + int(priority)
    encoded = encode_message(inner_message)
    sequence = connection.link.reserve_sequence(stream_id)
    associated_data = pack_frame_header(
        message_type=int(outer_type),
        flags=SECURE_FLAG,
        priority=int(priority),
        stream_id=stream_id,
        sequence=sequence,
        acknowledgement=0,
        payload_length=len(encoded) + 16,
    )
    try:
        payload = connection.secure_session.encrypt(
            stream_id,
            sequence,
            encoded,
            associated_data=associated_data,
        )
    finally:
        connection.link.cancel_sequence(stream_id, sequence)
    return Frame(
        message_type=int(outer_type),
        flags=SECURE_FLAG,
        priority=int(priority),
        stream_id=stream_id,
        sequence=sequence,
        acknowledgement=0,
        payload=payload,
    )


def test_fragmented_peer_pair_runs_all_services_and_reconnects_cleanly(tmp_path):
    left, left_input, left_processes = build_core(tmp_path, "left")
    right, right_input, right_processes = build_core(tmp_path, "right")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)

    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        assert left.snapshot.peer_id == "right"
        assert right.snapshot.peer_id == "left"

        plain_messages = []
        right.chat.add_message_listener(plain_messages.append)
        left.chat.send_plain("plain 한글 ?/=")
        assert wait_for(lambda: [item.text for item in plain_messages] == ["plain 한글 ?/="])

        left_snapshot = left.snapshot
        left.approve_peer(left_snapshot.connection_id, left_snapshot.fingerprint)
        time.sleep(0.02)
        assert not left.trusted
        assert not right.trusted
        right_snapshot = right.snapshot
        right.approve_peer(right_snapshot.connection_id, right_snapshot.fingerprint)
        assert wait_for(lambda: left.trusted and right.trusted)

        secure_messages = []
        right.chat.add_message_listener(secure_messages.append)
        left.chat.send_secure("secure punctuation ?_+ 한글")
        assert wait_for(
            lambda: any(
                item.secure and item.text == "secure punctuation ?_+ 한글"
                for item in secure_messages
            )
        )

        file_data = bytes(range(251)) * (
            ((WINDOW_SIZE + 2) * CHUNK_SIZE) // 251 + 1
        )
        file_data = file_data[: (WINDOW_SIZE + 2) * CHUNK_SIZE + 123]
        source = tmp_path / "multi-window.bin"
        source.write_bytes(file_data)
        right_progress = []
        right.files.add_progress_listener(right_progress.append)
        transfer_id = left.files.send_file(source).result(timeout=2)
        assert wait_for(
            lambda: any(
                item.transfer_id == transfer_id and item.state == "complete"
                for item in right_progress
            ),
            timeout=10,
        )
        assert (right.files.download_dir / source.name).read_bytes() == file_data

        shell_output = []
        left.shell.add_output_listener(shell_output.append)
        right.shell.set_allow_remote_shell(True)
        shell_id = left.shell.open_remote(columns=90, rows=30)
        assert wait_for(
            lambda: right_processes.processes
            and right_processes.processes[-1].started == [(90, 30)]
        )
        remote_process = right_processes.processes[-1]
        left.shell.send_input(shell_id, b"echo peer\r")
        assert wait_for(lambda: remote_process.writes == [b"echo peer\r"])
        remote_process.emit_output(b"peer output")
        assert wait_for(
            lambda: shell_output
            and shell_output[-1].session_id == shell_id
            and shell_output[-1].data == b"peer output"
        )
        left.shell.close_session(shell_id)
        assert wait_for(lambda: remote_process.terminated)

        right.input.set_allow_remote_input(True)
        right_input.position = (50, 50)
        input_id = left.input.request_control()
        assert wait_for(
            lambda: left.input.state is InputSessionState.CONTROLLING
            and right.input.state is InputSessionState.BEING_CONTROLLED
        )
        left_input.capture(PointerMotionEvent(10, 0))
        assert wait_for(lambda: right_input.position == (60, 50))
        left_input.capture(PointerMotionEvent(-1000, 0))
        assert wait_for(
            lambda: left.input.state is InputSessionState.IDLE
            and right.input.state is InputSessionState.IDLE
        )
        assert left.input.active_session_id is None
        assert right.input.active_session_id is None

        right.shell.set_allow_remote_shell(True)
        active_shell = left.shell.open_remote(columns=80, rows=24)
        assert wait_for(lambda: left.shell.active_session_id == active_shell)
        right.input.set_allow_remote_input(True)
        active_input = left.input.request_control()
        assert wait_for(
            lambda: left.input.active_session_id == active_input
            and right.input.state is InputSessionState.BEING_CONTROLLED
        )
        left_input.capture(KeyEvent(KeyAction.DOWN, usage=4, text="a"))
        assert wait_for(lambda: right_input.pressed_keys == frozenset({4}))

        left.disconnect()
        right.disconnect()
        assert wait_for(
            lambda: left.snapshot.state is CoreState.DISCONNECTED
            and right.snapshot.state is CoreState.DISCONNECTED
        )
        assert left.shell.active_session_id is None
        assert right.shell.active_session_id is None
        assert left.input.state is InputSessionState.IDLE
        assert right.input.state is InputSessionState.IDLE
        assert not left_input.capture_running
        assert right_input.pressed_keys == frozenset()
        assert all(process.terminated for process in right_processes.processes)
        assert not list(right.files.download_dir.glob("*.part"))
        assert not left.threads_alive
        assert not right.threads_alive

        new_left_endpoint, new_right_endpoint = endpoint_pair()
        left.connect_endpoint(new_left_endpoint)
        right.connect_endpoint(new_right_endpoint)
        assert wait_for(
            lambda: left.snapshot.state is CoreState.READY
            and right.snapshot.state is CoreState.READY
        )
        reconnected = []
        right.chat.add_message_listener(reconnected.append)
        left.chat.send_secure("reconnected")
        assert wait_for(
            lambda: any(item.secure and item.text == "reconnected" for item in reconnected)
        )
    finally:
        left.close()
        right.close()

    assert not left.threads_alive
    assert not right.threads_alive
    assert not left.files._timer_thread.is_alive()
    assert not right.files._timer_thread.is_alive()
    assert left_processes.processes == []


def test_core_rejects_spoofed_tampered_mismatched_and_replayed_frames(tmp_path):
    left, *_ = build_core(tmp_path, "left-security")
    right, *_ = build_core(tmp_path, "right-security")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)

    received = []
    right.chat.add_message_listener(received.append)
    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )

        spoofed_trust = Message(
            MessageType.TRUST,
            {
                "protocol": 1,
                "app_version": "spoofed",
                "platform": "spoofed",
                "peer_id": "spoofed",
                "features": ["chat"],
                "max_frame_size": 65_535,
                "accepts_fingerprint": right.identity.fingerprint,
            },
        )
        right._on_frame(
            right.snapshot.connection_id,
            Frame(
                message_type=int(MessageType.TRUST),
                flags=0,
                priority=int(Priority.INTERACTIVE),
                stream_id=int(MessageType.TRUST) * 4 + int(Priority.INTERACTIVE),
                sequence=50,
                acknowledgement=0,
                payload=encode_message(spoofed_trust),
            ),
        )
        assert right.snapshot.peer_id == "left-security"
        assert not right.snapshot.remote_approved

        approve_pair(left, right)

        tampered = make_secure_frame(
            left,
            MessageType.CHAT_SECURE,
            Message(MessageType.CHAT_SECURE, {}, b"tampered"),
        )
        right._on_frame(
            right.snapshot.connection_id,
            replace(tampered, acknowledgement=1),
        )
        assert received == []

        mismatched = make_secure_frame(
            left,
            MessageType.FILE_ACK,
            Message(MessageType.CHAT_SECURE, {}, b"mismatched"),
        )
        right._on_frame(right.snapshot.connection_id, mismatched)
        assert received == []

        replayed = make_secure_frame(
            left,
            MessageType.CHAT_SECURE,
            Message(MessageType.CHAT_SECURE, {}, b"once"),
        )
        right._on_frame(right.snapshot.connection_id, replayed)
        right._on_frame(right.snapshot.connection_id, replayed)
        assert [item.text for item in received] == ["once"]
    finally:
        left.close()
        right.close()


def test_peer_approval_is_scoped_to_the_current_connection(tmp_path):
    left, *_ = build_core(tmp_path, "left-stale")
    right, *_ = build_core(tmp_path, "right-stale")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)

    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        stale = left.snapshot
        left.disconnect()
        right.disconnect()
        assert wait_for(
            lambda: left.snapshot.state is CoreState.DISCONNECTED
            and right.snapshot.state is CoreState.DISCONNECTED
        )

        new_left_endpoint, new_right_endpoint = endpoint_pair()
        left.connect_endpoint(new_left_endpoint)
        right.connect_endpoint(new_right_endpoint)
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )

        with pytest.raises(CoreError, match="stale"):
            left.approve_peer(stale.connection_id, stale.fingerprint)
    finally:
        left.close()
        right.close()


def test_approval_persisting_during_disconnect_does_not_report_stale_failure(
    tmp_path,
):
    left, *_ = build_core(tmp_path, "left-approval-race")
    right, *_ = build_core(tmp_path, "right-approval-race")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)
    release_approval = threading.Event()

    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        approval = left.snapshot
        original_accept = left.trust_store.accept
        trust_written = threading.Event()

        def blocking_accept(peer_id, fingerprint):
            original_accept(peer_id, fingerprint)
            trust_written.set()
            assert release_approval.wait(2)

        left.trust_store.accept = blocking_accept
        approval_errors = []

        def approve():
            try:
                left.approve_peer(
                    approval.connection_id,
                    approval.fingerprint,
                )
            except BaseException as error:
                approval_errors.append(error)

        approver = threading.Thread(target=approve)
        approver.start()
        assert trust_written.wait(1)
        left_disconnector = threading.Thread(target=left.disconnect)
        right_disconnector = threading.Thread(target=right.disconnect)
        left_disconnector.start()
        right_disconnector.start()
        time.sleep(0.02)
        release_approval.set()
        approver.join(2)
        left_disconnector.join(2)
        right_disconnector.join(2)

        assert not approver.is_alive()
        assert not left_disconnector.is_alive()
        assert not right_disconnector.is_alive()
        assert approval_errors == []
        assert left.trust_store.check("right-approval-race", approval.fingerprint)
        assert left.snapshot.state is CoreState.DISCONNECTED
    finally:
        release_approval.set()
        left.close()
        right.close()


def test_approval_send_racing_disconnect_does_not_report_persisted_trust_as_failure(
    tmp_path,
):
    left, *_ = build_core(tmp_path, "left-approval-send-race")
    right, *_ = build_core(tmp_path, "right-approval-send-race")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)
    release_send = threading.Event()
    approver = None

    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        approval = left.snapshot
        original_send_internal = left._send_internal
        send_started = threading.Event()

        def blocking_send_internal(*args, **kwargs):
            send_started.set()
            assert release_send.wait(2)
            return original_send_internal(*args, **kwargs)

        left._send_internal = blocking_send_internal
        approval_errors = []

        def approve():
            try:
                left.approve_peer(
                    approval.connection_id,
                    approval.fingerprint,
                )
            except BaseException as error:
                approval_errors.append(error)

        approver = threading.Thread(target=approve)
        approver.start()
        assert send_started.wait(1)
        left.disconnect()
        right.disconnect()
        release_send.set()
        approver.join(2)

        assert not approver.is_alive()
        assert approval_errors == []
        assert left.trust_store.check(
            "right-approval-send-race",
            approval.fingerprint,
        )
    finally:
        release_send.set()
        if approver is not None:
            approver.join(2)
        left.close()
        right.close()


def test_explicit_disconnect_wins_over_error_reported_while_closing(tmp_path):
    left, *_ = build_core(tmp_path, "left-explicit-disconnect")
    right, *_ = build_core(tmp_path, "right-explicit-disconnect")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)

    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        connection = left._connection
        assert connection is not None
        original_close = connection.link.close

        def close_with_concurrent_error(*args, **kwargs):
            left._on_disconnect(
                connection.connection_id,
                OSError("endpoint unavailable"),
            )
            return original_close(*args, **kwargs)

        connection.link.close = close_with_concurrent_error
        left.disconnect()

        assert left.snapshot.state is CoreState.DISCONNECTED
        assert left.snapshot.error is None
    finally:
        left.close()
        right.close()


def test_explicit_disconnect_clears_error_callback_already_in_progress(
    tmp_path,
    monkeypatch,
):
    left, *_ = build_core(tmp_path, "left-error-callback-race")
    right, *_ = build_core(tmp_path, "right-error-callback-race")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)
    callback_sampled_intent = threading.Event()
    release_callback = threading.Event()
    callback_thread = None
    disconnector = None
    link = None

    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        connection = left._connection
        assert connection is not None
        link = connection.link
        original_is_set = left._disconnect_requested.is_set
        blocked_once = False

        def blocking_is_set():
            nonlocal blocked_once
            sampled = original_is_set()
            if (
                not blocked_once
                and threading.current_thread().name == "test-error-callback"
            ):
                blocked_once = True
                callback_sampled_intent.set()
                assert release_callback.wait(2)
            return sampled

        monkeypatch.setattr(
            left._disconnect_requested,
            "is_set",
            blocking_is_set,
        )
        callback_thread = threading.Thread(
            target=left._on_disconnect,
            args=(connection.connection_id, OSError("endpoint unavailable")),
            name="test-error-callback",
        )
        callback_thread.start()
        assert callback_sampled_intent.wait(1)

        disconnector = threading.Thread(target=left.disconnect)
        disconnector.start()
        assert left._disconnect_requested.wait(1)
        release_callback.set()
        callback_thread.join(2)
        disconnector.join(2)

        assert not callback_thread.is_alive()
        assert not disconnector.is_alive()
        assert left.snapshot.state is CoreState.DISCONNECTED
        assert left.snapshot.error is None
    finally:
        release_callback.set()
        if callback_thread is not None:
            callback_thread.join(2)
        if disconnector is not None:
            disconnector.join(2)
        if link is not None:
            link.close()
        left.close()
        right.close()


def test_disconnect_timeout_exposes_disconnecting_state(tmp_path):
    left, *_ = build_core(tmp_path, "left-disconnect-timeout")
    right, *_ = build_core(tmp_path, "right-disconnect-timeout")
    left_endpoint = BlockingCloseEndpoint()
    right_endpoint = MemoryEndpoint()
    left_endpoint.connect(right_endpoint)
    right_endpoint.connect(left_endpoint)
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)
    connection = None

    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        approve_pair(left, right)
        connection = left._connection
        assert connection is not None
        original_close = connection.link.close
        connection.link.close = lambda: original_close(timeout=0.02)

        with pytest.raises(LinkCloseTimeout):
            left.disconnect()

        assert left_endpoint.close_started.is_set()
        assert left.snapshot.state is CoreState.DISCONNECTING
        assert not left.trusted
    finally:
        left_endpoint.release_close.set()
        if connection is not None:
            connection.link.close = original_close
        left.close()
        right.close()


def test_secure_sequence_reservation_encryption_and_enqueue_are_serialized(tmp_path):
    left, *_ = build_core(tmp_path, "left-concurrent")
    right, *_ = build_core(tmp_path, "right-concurrent")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)

    received = []
    right.chat.add_message_listener(received.append)
    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        approve_pair(left, right)
        connection = left._connection
        assert connection is not None
        original_encrypt = connection.secure_session.encrypt
        first_encrypt_started = threading.Event()
        release_first_encrypt = threading.Event()

        def delayed_encrypt(
            stream_id,
            sequence,
            plaintext,
            *,
            associated_data=b"",
        ):
            if stream_id == int(MessageType.CHAT_SECURE) * 4 + int(Priority.NORMAL):
                if not first_encrypt_started.is_set():
                    first_encrypt_started.set()
                    assert release_first_encrypt.wait(2)
            return original_encrypt(
                stream_id,
                sequence,
                plaintext,
                associated_data=associated_data,
            )

        connection.secure_session.encrypt = delayed_encrypt
        errors = []

        def send(text):
            try:
                left.chat.send_secure(text)
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=send, args=("first",))
        second = threading.Thread(target=send, args=("second",))
        first.start()
        assert first_encrypt_started.wait(1)
        second.start()
        time.sleep(0.02)
        release_first_encrypt.set()
        first.join(2)
        second.join(2)

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert wait_for(
            lambda: [item.text for item in received] == ["first", "second"]
        )
    finally:
        left.close()
        right.close()


def test_trust_waits_until_local_hello_has_been_queued(tmp_path):
    left, *_ = build_core(tmp_path, "left-hello-order")
    right, *_ = build_core(tmp_path, "right-hello-order")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)

    original_send_internal = right._send_internal
    local_hello_started = threading.Event()
    release_local_hello = threading.Event()

    def delay_local_hello(
        connection_id,
        message,
        *,
        secure,
        priority,
        allow_untrusted,
    ):
        if message.message_type is MessageType.HELLO:
            local_hello_started.set()
            assert release_local_hello.wait(2)
        return original_send_internal(
            connection_id,
            message,
            secure=secure,
            priority=priority,
            allow_untrusted=allow_untrusted,
        )

    right._send_internal = delay_local_hello
    connector = threading.Thread(
        target=right.connect_endpoint,
        args=(right_endpoint,),
    )
    connector.start()
    try:
        assert local_hello_started.wait(1)
        time.sleep(0.02)
        release_local_hello.set()
        connector.join(2)

        assert not connector.is_alive()
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
    finally:
        release_local_hello.set()
        left.close()
        right.close()


def test_handshake_recovers_when_the_first_hello_is_lost(tmp_path):
    left, *_ = build_core(tmp_path, "left-lost-hello")
    right, *_ = build_core(tmp_path, "right-lost-hello")
    left_endpoint = GatedMemoryEndpoint()
    right_endpoint = GatedMemoryEndpoint()
    left_endpoint.connect(right_endpoint)
    right_endpoint.connect(left_endpoint)
    right_endpoint.accepting = False

    try:
        left.connect_endpoint(left_endpoint)
        assert wait_for(lambda: left_endpoint.dropped_bytes > 0, timeout=1)
        assert left.snapshot.state is CoreState.HANDSHAKING

        right_endpoint.accepting = True
        right.connect_endpoint(right_endpoint)

        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED,
            timeout=1.5,
        )
    finally:
        left.close()
        right.close()


def test_peer_reconnects_while_other_serial_port_stays_open(tmp_path):
    left, *_ = build_core(tmp_path, "left-asymmetric-reconnect")
    right, *_ = build_core(tmp_path, "right-asymmetric-reconnect")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)

    received = []
    left.chat.add_message_listener(received.append)
    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        approve_pair(left, right)

        right.disconnect()
        assert wait_for(lambda: right.snapshot.state is CoreState.DISCONNECTED)
        assert left.snapshot.state is CoreState.READY

        replacement = MemoryEndpoint()
        left_endpoint.connect(replacement)
        replacement.connect(left_endpoint)
        right.connect_endpoint(replacement)

        assert wait_for(
            lambda: left.snapshot.state is CoreState.READY
            and right.snapshot.state is CoreState.READY,
            timeout=2,
        )
        right.chat.send_secure("asymmetric reconnect")
        assert wait_for(
            lambda: any(
                item.secure and item.text == "asymmetric reconnect"
                for item in received
            )
        )
    finally:
        left.close()
        right.close()


def test_ready_transition_starts_persisted_auto_edge_capture(tmp_path):
    left, left_input, _ = build_core(
        tmp_path,
        "left-auto-edge",
        auto_edge_enabled=True,
    )
    right, right_input, _ = build_core(
        tmp_path,
        "right-auto-edge",
        auto_edge_enabled=True,
    )
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)

    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        assert not left_input.capture_running
        assert not right_input.capture_running

        approve_pair(left, right)

        assert wait_for(
            lambda: left_input.capture_running
            and right_input.capture_running
        )
    finally:
        left.close()
        right.close()


def test_replayed_plain_chat_frame_is_delivered_only_once(tmp_path):
    left, *_ = build_core(tmp_path, "left-plain-replay")
    right, *_ = build_core(tmp_path, "right-plain-replay")
    left_endpoint, right_endpoint = endpoint_pair()
    left.connect_endpoint(left_endpoint)
    right.connect_endpoint(right_endpoint)
    received = []
    right.chat.add_message_listener(received.append)

    try:
        assert wait_for(
            lambda: left.snapshot.state is CoreState.UNTRUSTED
            and right.snapshot.state is CoreState.UNTRUSTED
        )
        priority = Priority.NORMAL
        message_type = MessageType.CHAT_PLAIN
        replayed = Frame(
            message_type=int(message_type),
            flags=0,
            priority=int(priority),
            stream_id=int(message_type) * 4 + int(priority),
            sequence=99,
            acknowledgement=0,
            payload=encode_message(Message(message_type, {}, b"once")),
        )

        right._on_frame(right.snapshot.connection_id, replayed)
        right._on_frame(right.snapshot.connection_id, replayed)

        assert [message.text for message in received] == ["once"]
    finally:
        left.close()
        right.close()
