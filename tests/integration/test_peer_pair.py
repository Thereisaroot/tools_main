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


def build_core(tmp_path, name):
    backend = FakeInputBackend()
    processes = FakeProcessFactory()
    core = ShookLinkCore(
        identity=IdentityStore(tmp_path / f"{name}.identity").load_or_create(),
        trust_store=TrustStore(tmp_path / f"{name}.trust.json"),
        download_dir=tmp_path / f"{name}-downloads",
        local_peer_id=name,
        input_backend=backend,
        process_factory=processes,
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
        input_id = left.input.request_control()
        assert wait_for(
            lambda: left.input.state is InputSessionState.CONTROLLING
            and right.input.state is InputSessionState.BEING_CONTROLLED
        )
        left_input.capture(PointerMotionEvent(10, 0))
        assert wait_for(lambda: right_input.position == (10, 50))
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
