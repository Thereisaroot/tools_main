import hashlib
import threading
import time
from collections import deque
from concurrent.futures import Future
from pathlib import Path

import pytest

from shooklink.files.service import (
    CHUNK_SIZE,
    WINDOW_SIZE,
    FileHashMismatch,
    FileOffer,
    FileProtocolError,
    FileTransferError,
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


class InlineExecutor:
    def submit(self, function, *args, **kwargs):
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


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


@pytest.mark.parametrize("mutation", ["remove", "truncate"])
def test_source_startup_failure_cleans_sender_and_reliably_cancels_peer(
    tmp_path,
    mutation,
):
    source = tmp_path / f"source-{mutation}.bin"
    source.write_bytes(b"source payload")
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    progress = []
    service.add_progress_listener(progress.append)
    transfer_id = service.send_file(source).result(timeout=2)
    if mutation == "remove":
        source.unlink()
    else:
        source.write_bytes(b"")
    bus.sent.clear()

    service.handle_message(
        Message(MessageType.FILE_ACCEPT, {"transfer_id": transfer_id})
    )

    assert progress[-1].state == "failed"
    assert [message.message_type for message, _secure, _priority in bus.sent] == [
        MessageType.FILE_CANCEL
    ]
    assert bus.sent[-1][0].metadata["transfer_id"] == transfer_id

    bus.sent.clear()
    service.poll(now=time.monotonic() + RETRANSMIT_TIMEOUT + 1)

    assert [message.message_type for message, _secure, _priority in bus.sent] == [
        MessageType.FILE_CANCEL
    ]

    service.handle_message(
        Message(
            MessageType.FILE_CANCEL,
            {"transfer_id": transfer_id, "reason": "ack"},
        )
    )
    bus.sent.clear()
    service.poll(now=time.monotonic() + RETRANSMIT_TIMEOUT * 3)

    assert bus.sent == []
    service.close()


def test_mid_transfer_source_failure_cleans_sender_and_reliably_cancels_peer(
    tmp_path,
):
    source = tmp_path / "source-mid-transfer.bin"
    source.write_bytes(b"x" * (CHUNK_SIZE * (WINDOW_SIZE + 1)))
    bus = QueueBus()
    service = FileService(
        bus,
        tmp_path / "downloads",
        max_outgoing_transfers=1,
    )
    progress = []
    service.add_progress_listener(progress.append)
    transfer_id = service.send_file(source).result(timeout=2)
    service.handle_message(
        Message(MessageType.FILE_ACCEPT, {"transfer_id": transfer_id})
    )
    source.write_bytes(b"")
    bus.sent.clear()

    service.handle_message(
        Message(
            MessageType.FILE_ACK,
            {
                "transfer_id": transfer_id,
                "base": WINDOW_SIZE,
                "span": 0,
                "bitmap": "0",
            },
        )
    )

    assert progress[-1].state == "failed"
    assert [message.message_type for message, _secure, _priority in bus.sent] == [
        MessageType.FILE_CANCEL
    ]
    with pytest.raises(FileTransferError, match="capacity"):
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(b"replacement")
        service.send_file(replacement)

    service.handle_message(
        Message(
            MessageType.FILE_CANCEL,
            {"transfer_id": transfer_id, "reason": "ack"},
        )
    )
    replacement = tmp_path / "replacement.bin"
    assert service.send_file(replacement).result(timeout=2)
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


def test_cancel_control_retries_until_peer_acknowledges(tmp_path):
    source = tmp_path / "cancel-retry.bin"
    source.write_bytes(b"cancel")
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    transfer_id = service.send_file(source).result(timeout=2)
    service.cancel(transfer_id)
    bus.sent.clear()

    service.poll(now=time.monotonic() + RETRANSMIT_TIMEOUT + 1)
    assert bus.sent[-1][0].message_type is MessageType.FILE_CANCEL
    service.handle_message(
        Message(
            MessageType.FILE_CANCEL,
            {"transfer_id": transfer_id, "reason": "ack"},
        )
    )
    bus.sent.clear()

    service.poll(now=time.monotonic() + RETRANSMIT_TIMEOUT * 3)
    assert bus.sent == []
    service.close()


def test_serial_incoming_cancels_wait_for_ack_before_admitting_more(tmp_path):
    bus = QueueBus()
    service = FileService(
        bus,
        tmp_path / "downloads",
        max_incoming_transfers=2,
        max_outgoing_transfers=1,
    )
    offers = [
        make_offer(
            f"cancel-{index}.bin",
            b"x",
            transfer_id=f"{index + 1:032x}",
            chunk_size=1,
        )
        for index in range(3)
    ]
    for offer in offers[:2]:
        service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))
        assert bus.sent[-1][0].message_type is MessageType.FILE_ACCEPT
        service.cancel(offer.transfer_id)

    bus.sent.clear()
    service.handle_message(Message(MessageType.FILE_OFFER, offers[2].to_metadata()))

    assert bus.sent[-1][0].message_type is MessageType.FILE_CANCEL
    assert bus.sent[-1][0].metadata["reason"] == "busy"
    assert not any((tmp_path / "downloads").glob("*.part"))

    bus.sent.clear()
    service.poll(now=time.monotonic() + RETRANSMIT_TIMEOUT + 1)
    retried_ids = {
        message.metadata["transfer_id"]
        for message, _secure, _priority in bus.sent
        if message.message_type is MessageType.FILE_CANCEL
    }
    assert retried_ids == {offer.transfer_id for offer in offers[:2]}

    service.handle_message(
        Message(
            MessageType.FILE_CANCEL,
            {"transfer_id": offers[0].transfer_id, "reason": "ack"},
        )
    )
    bus.sent.clear()
    service.handle_message(Message(MessageType.FILE_OFFER, offers[2].to_metadata()))

    assert bus.sent[-1][0].message_type is MessageType.FILE_ACCEPT
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


def test_cancel_tombstone_survives_bounded_detail_cache_eviction(tmp_path):
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    first_offer = None
    for index in range(130):
        offer = make_offer(
            f"cancel-{index}.bin",
            b"x",
            transfer_id=f"{index + 1:032x}",
            chunk_size=1,
        )
        first_offer = first_offer or offer
        service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))
        service.handle_message(
            Message(MessageType.FILE_CANCEL, {"transfer_id": offer.transfer_id})
        )
    bus.sent.clear()

    service.handle_message(Message(MessageType.FILE_OFFER, first_offer.to_metadata()))

    assert bus.sent[-1][0].message_type is MessageType.FILE_CANCEL
    service.close()


def test_incoming_offer_respects_byte_capacity(tmp_path):
    offer = FileOffer(
        "1" * 32,
        "too-large.bin",
        11,
        0,
        "0" * 64,
        8,
    )
    bus = QueueBus()
    service = FileService(
        bus,
        tmp_path / "downloads",
        max_incoming_bytes=10,
    )

    service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))

    assert bus.sent[-1][0].message_type is MessageType.FILE_CANCEL
    assert bus.sent[-1][0].metadata["reason"] == "capacity"
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

    for reserved in ("CONIN$.txt", "CONOUT$.txt", "COM¹.txt", "LPT³.bin"):
        assert sanitize_filename(reserved) != reserved


def test_progress_listener_can_close_service_from_completion_thread(tmp_path):
    source = tmp_path / "callback-close.bin"
    source.write_bytes(b"callback")
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    errors = []
    closed = threading.Event()

    def listener(progress):
        if progress.state == "complete":
            try:
                service.close()
            except BaseException as error:
                errors.append(error)
            finally:
                closed.set()

    service.add_progress_listener(listener)
    offer = make_offer(
        source.name,
        source.read_bytes(),
        transfer_id="a" * 32,
        chunk_size=len(source.read_bytes()),
    )
    service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))
    service.handle_message(
        Message(
            MessageType.FILE_CHUNK,
            {"transfer_id": offer.transfer_id, "index": 0},
            source.read_bytes(),
        )
    )
    service.handle_message(
        Message(
            MessageType.FILE_FINISH,
            {"transfer_id": offer.transfer_id, "sha256": offer.sha256},
        )
    )

    assert closed.wait(2)
    assert errors == []


def test_progress_listener_exception_does_not_abandon_offered_transfer(
    tmp_path,
    caplog,
):
    source = tmp_path / "listener-error.bin"
    source.write_bytes(b"listener")
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    observed = []

    def failing_listener(progress):
        if progress.state == "offered":
            raise RuntimeError("listener failed")

    service.add_progress_listener(failing_listener)
    service.add_progress_listener(observed.append)
    caplog.set_level("ERROR", logger="shooklink.files.service")

    try:
        transfer_id = service.send_file(source).result(timeout=2)
        offer = next(
            message
            for message, _secure, _priority in bus.sent
            if message.message_type is MessageType.FILE_OFFER
        )
        assert offer.metadata["transfer_id"] == transfer_id
        assert observed[-1].state == "offered"
        assert not any(
            message.message_type is MessageType.FILE_CANCEL
            for message, _secure, _priority in bus.sent
        )

        bus.sent.clear()
        service.handle_message(
            Message(MessageType.FILE_ACCEPT, {"transfer_id": transfer_id})
        )

        assert any(
            message.message_type is MessageType.FILE_CHUNK
            for message, _secure, _priority in bus.sent
        )
        assert "file progress listener failed" in caplog.text
    finally:
        service.close()


def test_closed_service_rejects_incoming_offer_without_creating_partial_file(tmp_path):
    offer = make_offer("closed.bin", b"closed", transfer_id="b" * 32)
    bus = QueueBus()
    download_dir = tmp_path / "downloads"
    service = FileService(bus, download_dir)
    service.close()
    bus.sent.clear()

    assert not service.handle_message(
        Message(MessageType.FILE_OFFER, offer.to_metadata())
    )
    assert bus.sent == []
    assert not list(download_dir.glob("*.part"))


def test_close_during_offer_decrypt_cannot_accept_or_create_partial_file(tmp_path):
    class BlockingDecryptBus(QueueBus):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def decrypt_secure(self, message):
            self.decrypt_calls.append(message)
            self.started.set()
            assert self.release.wait(2)
            return message.body

    offer = make_offer("decrypt-close.bin", b"closed", transfer_id="c" * 32)
    bus = BlockingDecryptBus()
    download_dir = tmp_path / "downloads"
    service = FileService(bus, download_dir)
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))
        )
    )
    worker.start()
    assert bus.started.wait(1)

    service.close()
    bus.release.set()
    worker.join(2)

    assert not worker.is_alive()
    assert results == [False]
    assert bus.sent == []
    assert not list(download_dir.glob("*.part"))


def test_close_after_cancel_decrypt_cannot_repopulate_terminal_state(
    tmp_path,
    monkeypatch,
):
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    entered_handler = threading.Event()
    release_handler = threading.Event()
    original_handler = service._handle_cancel

    def blocking_handler(message):
        entered_handler.set()
        assert release_handler.wait(2)
        original_handler(message)

    monkeypatch.setattr(service, "_handle_cancel", blocking_handler)
    worker = threading.Thread(
        target=service.handle_message,
        args=(
            Message(
                MessageType.FILE_CANCEL,
                {"transfer_id": "f" * 32, "reason": "cancelled"},
            ),
        ),
    )
    worker.start()
    assert entered_handler.wait(1)

    service.close()
    release_handler.set()
    worker.join(2)

    assert not worker.is_alive()
    assert bus.sent == []
    assert service._terminal_status == {}


def test_close_during_offer_policy_cannot_accept_or_leak_partial_file(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def blocking_policy(_offer):
        started.set()
        assert release.wait(2)
        return True

    offer = make_offer("policy-close.bin", b"closed", transfer_id="b" * 32)
    bus = QueueBus()
    download_dir = tmp_path / "downloads"
    service = FileService(bus, download_dir, accept_offer=blocking_policy)
    worker = threading.Thread(
        target=service.handle_message,
        args=(Message(MessageType.FILE_OFFER, offer.to_metadata()),),
    )
    worker.start()
    assert started.wait(1)

    service.close()
    release.set()
    worker.join(2)

    assert not worker.is_alive()
    assert bus.sent == []
    assert not list(download_dir.glob("*.part"))


def test_cancel_during_offer_policy_prevents_accept_and_partial_file(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def blocking_policy(_offer):
        started.set()
        assert release.wait(2)
        return True

    offer = make_offer("policy-cancel.bin", b"cancelled", transfer_id="a" * 32)
    bus = QueueBus()
    download_dir = tmp_path / "downloads"
    service = FileService(bus, download_dir, accept_offer=blocking_policy)
    worker = threading.Thread(
        target=service.handle_message,
        args=(Message(MessageType.FILE_OFFER, offer.to_metadata()),),
    )
    worker.start()
    assert started.wait(1)

    service.handle_message(
        Message(
            MessageType.FILE_CANCEL,
            {"transfer_id": offer.transfer_id, "reason": "cancelled"},
        )
    )
    release.set()
    worker.join(2)

    assert not worker.is_alive()
    assert not any(
        message.message_type is MessageType.FILE_ACCEPT
        for message, _secure, _priority in bus.sent
    )
    assert bus.sent[-1][0].message_type is MessageType.FILE_CANCEL
    assert bus.sent[-1][0].metadata["reason"] == "cancelled"
    assert not list(download_dir.glob("*.part"))
    service.close()


def test_unacknowledged_cancellation_counts_against_outgoing_capacity(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    service = FileService(
        QueueBus(),
        tmp_path / "downloads",
        max_outgoing_transfers=1,
    )
    transfer_id = service.send_file(first).result(timeout=2)
    service.cancel(transfer_id)

    with pytest.raises(Exception, match="capacity"):
        service.send_file(second)

    service.close()


def test_cancel_reserves_capacity_before_reentrant_progress_listener(tmp_path):
    first = tmp_path / "first-reentrant.bin"
    second = tmp_path / "second-reentrant.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    service = FileService(
        QueueBus(),
        tmp_path / "downloads",
        max_outgoing_transfers=1,
    )
    transfer_id = service.send_file(first).result(timeout=2)
    errors = []

    def listener(progress):
        if progress.transfer_id == transfer_id and progress.state == "cancelled":
            try:
                service.send_file(second)
            except Exception as error:
                errors.append(error)

    service.add_progress_listener(listener)
    service.cancel(transfer_id)

    assert len(errors) == 1
    assert "capacity" in str(errors[0])
    service.close()


def test_reentrant_close_cannot_be_followed_by_pending_cancel_insertion(tmp_path):
    source = tmp_path / "close-reentrant.bin"
    source.write_bytes(b"source")
    service = FileService(QueueBus(), tmp_path / "downloads")
    transfer_id = service.send_file(source).result(timeout=2)

    def listener(progress):
        if progress.transfer_id == transfer_id and progress.state == "cancelled":
            service.close()

    service.add_progress_listener(listener)
    service.cancel(transfer_id)

    assert service._closed is True
    assert service._pending_cancels == {}


def test_prepared_transfer_keeps_capacity_reserved_until_registration(
    tmp_path,
    monkeypatch,
):
    from shooklink.files import service as service_module

    entered_transition = threading.Event()
    release_transition = threading.Event()

    class BlockingTransitionFuture(Future):
        created = 0

        def __init__(self):
            super().__init__()
            self._blocks_transition = self.__class__.created == 0
            self.__class__.created += 1

        def set_running_or_notify_cancel(self):
            if self._blocks_transition:
                entered_transition.set()
                assert release_transition.wait(2)
            return super().set_running_or_notify_cancel()

    monkeypatch.setattr(service_module, "Future", BlockingTransitionFuture)
    first = tmp_path / "first-capacity.bin"
    second = tmp_path / "second-capacity.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    service = FileService(
        QueueBus(),
        tmp_path / "downloads",
        max_outgoing_transfers=1,
    )
    result = service.send_file(first)
    assert entered_transition.wait(1)

    try:
        with pytest.raises(FileTransferError, match="capacity"):
            service.send_file(second)
    finally:
        release_transition.set()
        result.result(timeout=2)
        service.close()


def test_exact_terminal_registry_refuses_new_ids_when_its_bound_is_reached(tmp_path):
    bus = QueueBus()
    service = FileService(
        bus,
        tmp_path / "downloads",
        max_terminal_transfers=2,
    )
    for transfer_id in ("1" * 32, "2" * 32):
        service.handle_message(
            Message(
                MessageType.FILE_CANCEL,
                {"transfer_id": transfer_id, "reason": "cancelled"},
            )
        )
    bus.sent.clear()
    offer = make_offer("new.bin", b"new", transfer_id="3" * 32)

    service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))

    assert bus.sent[-1][0].message_type is MessageType.FILE_CANCEL
    assert bus.sent[-1][0].metadata["reason"] == "capacity"
    service.close()


def test_terminal_registry_retains_offer_identity_after_detail_eviction(
    tmp_path,
    monkeypatch,
):
    from shooklink.files import service as service_module

    monkeypatch.setattr(service_module, "MAX_COMPLETED_TRANSFERS", 1)
    bus = QueueBus()
    service = FileService(
        bus,
        tmp_path / "downloads",
        executor=InlineExecutor(),
    )
    first = make_offer("first.bin", b"first", transfer_id="d" * 32)
    second = make_offer("second.bin", b"second", transfer_id="e" * 32)

    for offer, data in ((first, b"first"), (second, b"second")):
        service.handle_message(Message(MessageType.FILE_OFFER, offer.to_metadata()))
        service.handle_message(
            Message(
                MessageType.FILE_CHUNK,
                {"transfer_id": offer.transfer_id, "index": 0},
                data,
            )
        )
        service.handle_message(
            Message(
                MessageType.FILE_FINISH,
                {"transfer_id": offer.transfer_id, "sha256": offer.sha256},
            )
        )

    bus.sent.clear()
    service.handle_message(Message(MessageType.FILE_OFFER, first.to_metadata()))
    same_offer_reply = bus.sent[-1][0]

    conflicting = make_offer(
        "conflict.bin",
        b"different",
        transfer_id=first.transfer_id,
    )
    service.handle_message(Message(MessageType.FILE_OFFER, conflicting.to_metadata()))
    conflicting_reply = bus.sent[-1][0]

    assert same_offer_reply.message_type is MessageType.FILE_FINISH
    assert same_offer_reply.metadata["status"] == "ok"
    assert conflicting_reply.message_type is MessageType.FILE_CANCEL
    assert conflicting_reply.metadata["reason"] == "duplicate"
    service.close()


def test_disk_capacity_accounts_for_all_reserved_incoming_files(tmp_path, monkeypatch):
    from shooklink.files import service as service_module

    monkeypatch.setattr(service_module, "_available_disk_bytes", lambda _path: 5)
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads", max_incoming_bytes=100)
    first = make_offer("first.bin", b"123", transfer_id="4" * 32)
    second = make_offer("second.bin", b"456", transfer_id="5" * 32)

    service.handle_message(Message(MessageType.FILE_OFFER, first.to_metadata()))
    service.handle_message(Message(MessageType.FILE_OFFER, second.to_metadata()))

    assert [item[0].message_type for item in bus.sent] == [
        MessageType.FILE_ACCEPT,
        MessageType.FILE_CANCEL,
    ]
    assert bus.sent[-1][0].metadata["reason"] == "capacity"
    service.close()


def test_offered_progress_is_linearized_before_peer_cancel(tmp_path, monkeypatch):
    source = tmp_path / "linearized.bin"
    source.write_bytes(b"linearized")
    bus = QueueBus()
    service = FileService(bus, tmp_path / "downloads")
    progress = []
    service.add_progress_listener(progress.append)
    notify_entered = threading.Event()
    release_notify = threading.Event()
    original_notify = service._notify_outgoing

    def blocking_notify(transfer, state):
        if state == "offered":
            notify_entered.set()
            assert release_notify.wait(2)
        original_notify(transfer, state)

    monkeypatch.setattr(service, "_notify_outgoing", blocking_notify)
    future = service.send_file(source)
    assert notify_entered.wait(2)
    offer = next(
        item[0]
        for item in bus.sent
        if item[0].message_type is MessageType.FILE_OFFER
    )
    cancel_done = threading.Event()

    def cancel_from_peer():
        service.handle_message(
            Message(
                MessageType.FILE_CANCEL,
                {"transfer_id": offer.metadata["transfer_id"], "reason": "cancelled"},
            )
        )
        cancel_done.set()

    cancel_thread = threading.Thread(target=cancel_from_peer)
    cancel_thread.start()
    time.sleep(0.02)
    release_notify.set()
    transfer_id = future.result(timeout=2)
    cancel_thread.join(2)

    assert cancel_done.is_set()
    assert transfer_id == offer.metadata["transfer_id"]
    assert [item.state for item in progress[-2:]] == ["offered", "cancelled"]
    service.close()
