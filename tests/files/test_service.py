import hashlib
import threading
import time
from collections import deque
from pathlib import Path

import pytest

from shooklink.files.service import (
    FileHashMismatch,
    FileOffer,
    FileProtocolError,
    RETRANSMIT_TIMEOUT,
    FileService,
    IncomingTransfer,
    OutgoingTransfer,
    build_ack_metadata,
    sanitize_filename,
)
from shooklink.protocol.messages import Message, MessageType
from shooklink.transport.multiplexer import Priority


class QueueBus:
    def __init__(self, trusted=True, authenticated=True):
        self.trusted = trusted
        self.authenticated = authenticated
        self.queue = deque()
        self.sent = []
        self.decrypt_calls = []

    def send(self, message, *, secure=True, priority=Priority.NORMAL):
        item = (message, secure, priority)
        self.queue.append(item)
        self.sent.append(item)

    def decrypt_secure(self, message):
        self.decrypt_calls.append(message)
        if not self.trusted or not self.authenticated:
            raise RuntimeError("file frame authentication failed")
        return message.body


def make_offer(name, data, *, transfer_id="a" * 32, chunk_size=8):
    return FileOffer(
        transfer_id=transfer_id,
        name=name,
        size=len(data),
        mtime_ns=1_700_000_000_000_000_000,
        sha256=hashlib.sha256(data).hexdigest(),
        chunk_size=chunk_size,
    )


@pytest.mark.parametrize(
    ("untrusted", "expected"),
    [
        ("../../escape.txt", "escape.txt"),
        (r"..\..\windows.txt", "windows.txt"),
        ("/absolute/path/data.bin", "data.bin"),
        ("..", "download"),
        ("\x00bad.txt", "bad.txt"),
    ],
)
def test_sanitize_filename_reduces_input_to_a_safe_basename(untrusted, expected):
    assert sanitize_filename(untrusted) == expected


def test_incoming_data_remains_partial_until_valid_final_hash(tmp_path):
    data = b"complete payload"
    transfer = IncomingTransfer.create(make_offer("payload.bin", data), tmp_path)

    assert transfer.partial_path.name.endswith(".part")
    assert transfer.partial_path.exists()
    assert not transfer.final_path.exists()

    for index in range(transfer.chunk_count):
        start = index * transfer.offer.chunk_size
        transfer.write_chunk(index, data[start : start + transfer.offer.chunk_size])

    final_path = transfer.finalize()

    assert final_path.read_bytes() == data
    assert not transfer.partial_path.exists()


def test_bad_final_hash_removes_partial_file(tmp_path):
    data = b"payload"
    offer = make_offer("payload.bin", data)
    offer = FileOffer(
        offer.transfer_id,
        offer.name,
        offer.size,
        offer.mtime_ns,
        "0" * 64,
        offer.chunk_size,
    )
    transfer = IncomingTransfer.create(offer, tmp_path)
    transfer.write_chunk(0, data)

    with pytest.raises(FileHashMismatch):
        transfer.finalize()

    assert not transfer.partial_path.exists()
    assert not transfer.final_path.exists()


def test_existing_destination_is_not_overwritten(tmp_path):
    (tmp_path / "same.txt").write_text("old", encoding="utf-8")
    data = b"new"
    transfer = IncomingTransfer.create(make_offer("same.txt", data), tmp_path)
    transfer.write_chunk(0, data)

    final_path = transfer.finalize()

    assert final_path.name == "same (1).txt"
    assert (tmp_path / "same.txt").read_text(encoding="utf-8") == "old"


def test_destination_created_during_transfer_is_not_overwritten(tmp_path):
    data = b"remote"
    transfer = IncomingTransfer.create(make_offer("race.txt", data), tmp_path)
    transfer.write_chunk(0, data)
    (tmp_path / "race.txt").write_text("local", encoding="utf-8")

    final_path = transfer.finalize()

    assert (tmp_path / "race.txt").read_text(encoding="utf-8") == "local"
    assert final_path.name == "race (1).txt"
    assert final_path.read_bytes() == data


def test_selective_repeat_requeues_only_reported_holes_and_bounds_window(tmp_path):
    chunk_size = 8
    data = bytes(index % 256 for index in range(40 * chunk_size))
    path = tmp_path / "forty.bin"
    path.write_bytes(data)
    sender = OutgoingTransfer.from_path(
        path,
        transfer_id="b" * 32,
        chunk_size=chunk_size,
        window_size=16,
    )
    sender.accept()
    seen = set()
    received = set()
    retransmitted = []

    messages = sender.next_messages(now=1.0)
    assert len(messages) == 16
    assert sender.in_flight_count <= 16

    for message in messages:
        index = message.metadata["index"]
        seen.add(index)
        if index != 3:
            received.add(index)
    messages = sender.acknowledge(build_ack_metadata(received), now=2.0)
    for message in messages:
        if message.message_type is not MessageType.FILE_CHUNK:
            continue
        index = message.metadata["index"]
        if index in seen:
            retransmitted.append(index)
        seen.add(index)
        if index != 17:
            received.add(index)
    assert sender.in_flight_count <= 16

    messages = sender.acknowledge(build_ack_metadata(received), now=3.0)
    for message in messages:
        if message.message_type is not MessageType.FILE_CHUNK:
            continue
        index = message.metadata["index"]
        if index in seen:
            retransmitted.append(index)
        seen.add(index)
        received.add(index)
    assert sender.in_flight_count <= 16

    messages = sender.acknowledge(build_ack_metadata(received), now=4.0)

    assert retransmitted == [3, 17]
    assert seen == set(range(40))
    assert sender.in_flight_count == 0
    assert [message.message_type for message in messages] == [MessageType.FILE_FINISH]


def test_retransmit_timeout_only_resends_expired_chunks(tmp_path):
    path = tmp_path / "timeout.bin"
    path.write_bytes(b"0123456789abcdef")
    sender = OutgoingTransfer.from_path(
        path,
        transfer_id="c" * 32,
        chunk_size=8,
        window_size=2,
        retransmit_timeout=0.5,
    )
    sender.accept()
    sender.next_messages(now=10.0)

    assert sender.retransmit_expired(now=10.49) == []
    retried = sender.retransmit_expired(now=10.5)

    assert [message.metadata["index"] for message in retried] == [0, 1]


def test_late_ack_and_timer_after_finish_are_harmless(tmp_path):
    path = tmp_path / "finished.bin"
    path.write_bytes(b"finished")
    sender = OutgoingTransfer.from_path(
        path,
        transfer_id="d" * 32,
        chunk_size=8,
    )
    sender.accept()
    sender.next_messages(now=1.0)
    finished = sender.acknowledge(
        {"base": 1, "span": 0, "bitmap": "0"},
        now=2.0,
    )

    assert [message.message_type for message in finished] == [MessageType.FILE_FINISH]
    assert sender.acknowledge(
        {"base": 1, "span": 0, "bitmap": "0"},
        now=3.0,
    ) == []
    assert sender.retransmit_expired(now=4.0) == []


def test_ack_bitmap_cannot_set_bits_outside_its_span(tmp_path):
    path = tmp_path / "bitmap.bin"
    path.write_bytes(b"bitmap")
    sender = OutgoingTransfer.from_path(
        path,
        transfer_id="e" * 32,
        chunk_size=8,
    )
    sender.accept()
    sender.next_messages(now=1.0)

    with pytest.raises(FileProtocolError):
        sender.acknowledge(
            {"base": 0, "span": 0, "bitmap": "1"},
            now=2.0,
        )


def test_cancel_removes_only_its_partial_file(tmp_path):
    first = IncomingTransfer.create(
        make_offer("first.bin", b"first", transfer_id="1" * 32),
        tmp_path,
    )
    second = IncomingTransfer.create(
        make_offer("second.bin", b"second", transfer_id="2" * 32),
        tmp_path,
    )

    first.cancel()

    assert not first.partial_path.exists()
    assert second.partial_path.exists()
    second.cancel()


def test_service_hashes_offer_off_thread_and_starts_with_a_bounded_window(tmp_path):
    path = tmp_path / "window.bin"
    path.write_bytes(b"x" * (20 * 16 * 1024))
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")

    transfer_id = service.send_file(path).result(timeout=2)

    offer, secure, priority = bus.queue.popleft()
    assert offer.message_type is MessageType.FILE_OFFER
    assert offer.metadata["transfer_id"] == transfer_id
    assert offer.metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert secure is True
    assert priority is Priority.NORMAL

    service.handle_message(
        type(offer)(MessageType.FILE_ACCEPT, {"transfer_id": transfer_id})
    )
    chunks = [item for item in bus.sent if item[0].message_type is MessageType.FILE_CHUNK]

    assert len(chunks) == 16
    assert all(item[1] is True for item in chunks)
    assert all(item[2] is Priority.FILE for item in chunks)
    service.close()


def test_incoming_file_offer_requires_authenticated_transport(tmp_path):
    data = b"blocked"
    offer = make_offer("blocked.bin", data)
    bus = QueueBus(authenticated=False)
    downloads = tmp_path / "downloads"
    service = FileService(bus, downloads)

    with pytest.raises(FileProtocolError, match="authentication"):
        service.handle_message(
            Message(MessageType.FILE_OFFER, offer.to_metadata())
        )

    assert not downloads.exists()
    assert len(bus.decrypt_calls) == 1
    service.close()


def test_incoming_offer_count_is_bounded(tmp_path):
    bus = QueueBus()
    service = FileService(
        bus,
        tmp_path / "downloads",
        max_incoming_transfers=2,
    )
    for index in range(3):
        offer = make_offer(
            f"{index}.bin",
            b"data",
            transfer_id=f"{index + 1:032x}",
        )
        service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))

    assert [item[0].message_type for item in bus.sent] == [
        MessageType.FILE_ACCEPT,
        MessageType.FILE_ACCEPT,
        MessageType.FILE_CANCEL,
    ]
    assert bus.sent[-1][0].metadata["reason"] == "busy"
    service.close()


def test_two_services_complete_an_encrypted_file_transfer(tmp_path):
    source = tmp_path / "source.dat"
    data = (b"serial-windowed-transfer\n" * 2000) + bytes(range(256))
    source.write_bytes(data)
    left_bus = QueueBus()
    right_bus = QueueBus()
    left = FileService(left_bus, tmp_path / "left-downloads")
    right_downloads = tmp_path / "right-downloads"
    right = FileService(right_bus, right_downloads)
    transfer_id = left.send_file(source).result(timeout=2)
    deadline = time.monotonic() + 5

    while time.monotonic() < deadline:
        progressed = False
        while left_bus.queue:
            message, secure, _priority = left_bus.queue.popleft()
            assert secure is True
            right.handle_message(message)
            progressed = True
        while right_bus.queue:
            message, secure, _priority = right_bus.queue.popleft()
            assert secure is True
            left.handle_message(message)
            progressed = True
        target = right_downloads / source.name
        if target.exists() and not left_bus.queue and not right_bus.queue:
            break
        if not progressed:
            time.sleep(0.005)
    else:
        pytest.fail("file transfer did not complete")

    assert (right_downloads / source.name).read_bytes() == data
    assert any(
        item[0].message_type is MessageType.FILE_FINISH
        and item[0].metadata.get("status") == "ok"
        for item in right_bus.sent
    )
    assert all(
        priority is Priority.FILE
        for message, _secure, priority in left_bus.sent
        if message.message_type is MessageType.FILE_CHUNK
    )
    assert transfer_id
    left.close()
    right.close()


def test_service_drives_retransmission_timer_without_manual_poll(tmp_path):
    source = tmp_path / "retry.bin"
    source.write_bytes(b"retry me")
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    transfer_id = service.send_file(source).result(timeout=2)
    bus.queue.clear()
    service.handle_message(
        Message(MessageType.FILE_ACCEPT, {"transfer_id": transfer_id})
    )
    bus.queue.clear()
    deadline = time.monotonic() + RETRANSMIT_TIMEOUT + 1

    while time.monotonic() < deadline:
        if any(
            item[0].message_type is MessageType.FILE_CHUNK
            for item in bus.queue
        ):
            break
        time.sleep(0.01)
    else:
        pytest.fail("file chunk was not retransmitted automatically")
    service.close()


def test_service_retries_offer_and_finish_control_frames(tmp_path):
    source = tmp_path / "control.bin"
    source.write_bytes(b"control")
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    transfer_id = service.send_file(source).result(timeout=2)
    bus.queue.clear()
    deadline = time.monotonic() + RETRANSMIT_TIMEOUT + 1
    while time.monotonic() < deadline and not bus.queue:
        time.sleep(0.01)
    assert bus.queue.popleft()[0].message_type is MessageType.FILE_OFFER

    service.handle_message(Message(MessageType.FILE_ACCEPT, {"transfer_id": transfer_id}))
    service.handle_message(
        Message(
            MessageType.FILE_ACK,
            {
                "transfer_id": transfer_id,
                "base": 1,
                "span": 0,
                "bitmap": "0",
            },
        )
    )
    bus.queue.clear()
    deadline = time.monotonic() + RETRANSMIT_TIMEOUT + 1
    while time.monotonic() < deadline and not bus.queue:
        time.sleep(0.01)
    assert bus.queue.popleft()[0].message_type is MessageType.FILE_FINISH
    service.close()


def test_incoming_offer_policy_can_reject_without_reserving_a_file(tmp_path):
    offer = make_offer("reject.bin", b"rejected")
    bus = QueueBus()
    downloads = tmp_path / "downloads"
    service = FileService(bus, downloads, accept_offer=lambda _offer: False)

    service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))

    rejection = bus.sent[-1][0]
    assert rejection.message_type is MessageType.FILE_CANCEL
    assert rejection.metadata == {
        "transfer_id": offer.transfer_id,
        "reason": "rejected",
    }
    assert not downloads.exists()
    service.close()


def test_local_and_remote_cancel_emit_terminal_progress(tmp_path):
    source = tmp_path / "cancel.bin"
    source.write_bytes(b"cancel")
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    progress = []
    service.add_progress_listener(progress.append)
    outgoing_id = service.send_file(source).result(timeout=2)

    service.cancel(outgoing_id)
    assert progress[-1].transfer_id == outgoing_id
    assert progress[-1].state == "cancelled"

    incoming = make_offer("remote.bin", b"remote", transfer_id="8" * 32)
    service.handle_message(Message(MessageType.FILE_OFFER, incoming.to_metadata()))
    service.handle_message(
        Message(MessageType.FILE_CANCEL, {"transfer_id": incoming.transfer_id})
    )
    assert progress[-1].transfer_id == incoming.transfer_id
    assert progress[-1].state == "cancelled"
    service.close()


def test_cancelled_incoming_transfer_cannot_be_resurrected_by_late_offer(tmp_path):
    offer = make_offer("cancelled.bin", b"cancelled", transfer_id="3" * 32)
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    message = Message(MessageType.FILE_OFFER, offer.to_metadata())
    service.handle_message(message)
    service.handle_message(
        Message(MessageType.FILE_CANCEL, {"transfer_id": offer.transfer_id})
    )
    bus.sent.clear()

    service.handle_message(message)

    assert bus.sent[-1][0].message_type is MessageType.FILE_CANCEL
    assert bus.sent[-1][0].metadata["reason"] == "cancelled"
    assert not any((tmp_path / "downloads").glob("*.part"))
    service.close()


def test_conflicting_duplicate_offer_is_explicitly_rejected(tmp_path):
    first = make_offer("same.bin", b"first", transfer_id="2" * 32)
    conflicting = make_offer("same.bin", b"other", transfer_id="2" * 32)
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    service.handle_message(Message(MessageType.FILE_OFFER, first.to_metadata()))

    service.handle_message(Message(MessageType.FILE_OFFER, conflicting.to_metadata()))

    assert bus.sent[-1][0].message_type is MessageType.FILE_CANCEL
    assert bus.sent[-1][0].metadata["reason"] == "duplicate"
    service.close()


def test_cancel_during_final_hash_cannot_commit_or_report_success(tmp_path, monkeypatch):
    from shooklink.files import service as service_module

    data = b"finalize race"
    offer = make_offer(
        "race.bin",
        data,
        transfer_id="7" * 32,
        chunk_size=len(data),
    )
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))
    service.handle_message(
        Message(
            MessageType.FILE_CHUNK,
            {"transfer_id": offer.transfer_id, "index": 0},
            data,
        )
    )
    started = threading.Event()
    release = threading.Event()
    original_hash = service_module._sha256_path

    def blocking_hash(path):
        started.set()
        release.wait(2)
        return original_hash(path)

    monkeypatch.setattr(service_module, "_sha256_path", blocking_hash)
    service.handle_message(
        Message(
            MessageType.FILE_FINISH,
            {"transfer_id": offer.transfer_id, "sha256": offer.sha256},
        )
    )
    assert started.wait(1)

    service.cancel(offer.transfer_id)
    release.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not any((tmp_path / "downloads").glob("*.part")):
            break
        time.sleep(0.01)

    assert not (tmp_path / "downloads" / "race.bin").exists()
    assert not any(
        message.message_type is MessageType.FILE_FINISH
        and message.metadata.get("status") == "ok"
        for message, _secure, _priority in bus.sent
    )
    service.close()


def test_duplicate_finish_and_late_ack_are_idempotent(tmp_path):
    data = b"duplicate"
    offer = make_offer(
        "duplicate.bin",
        data,
        transfer_id="6" * 32,
        chunk_size=len(data),
    )
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))
    service.handle_message(
        Message(
            MessageType.FILE_CHUNK,
            {"transfer_id": offer.transfer_id, "index": 0},
            data,
        )
    )
    finish = Message(
        MessageType.FILE_FINISH,
        {"transfer_id": offer.transfer_id, "sha256": offer.sha256},
    )
    service.handle_message(finish)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if any(
            message.message_type is MessageType.FILE_FINISH
            and message.metadata.get("status") == "ok"
            for message, _secure, _priority in bus.sent
        ):
            break
        time.sleep(0.01)
    service.handle_message(finish)
    statuses = [
        message.metadata.get("status")
        for message, _secure, _priority in bus.sent
        if message.message_type is MessageType.FILE_FINISH
        and "status" in message.metadata
    ]
    assert statuses == ["ok", "ok"]

    service.handle_message(
        Message(
            MessageType.FILE_ACK,
            {
                "transfer_id": "5" * 32,
                "base": 0,
                "span": 0,
                "bitmap": "0",
            },
        )
    )
    service.close()


def test_cancelled_preparation_future_never_sends_offer(tmp_path, monkeypatch):
    from shooklink.files import service as service_module

    source = tmp_path / "slow.bin"
    source.write_bytes(b"slow")
    started = threading.Event()
    release = threading.Event()
    original_hash = service_module._sha256_path

    def blocking_hash(path):
        started.set()
        release.wait(2)
        return original_hash(path)

    monkeypatch.setattr(service_module, "_sha256_path", blocking_hash)
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    future = service.send_file(source)
    assert started.wait(1)

    assert future.cancel()
    release.set()
    time.sleep(0.05)

    assert bus.sent == []
    service.close()


def test_timestamp_and_windows_filename_edge_cases_are_safe(tmp_path):
    with pytest.raises(ValueError, match="mtime"):
        FileOffer("4" * 32, "time.bin", 0, 1 << 80, "0" * 64)

    assert sanitize_filename("CON .txt").upper().split(".", 1)[0].strip() != "CON"
    sanitized = sanitize_filename("bad\ud800name.txt")
    sanitized.encode("utf-8")
