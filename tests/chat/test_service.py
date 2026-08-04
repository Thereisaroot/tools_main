import threading
import time

import pytest

from shooklink.chat.service import (
    ChatService,
    ChatTextTooLarge,
    PeerNotTrusted,
)
from shooklink.protocol.messages import Message, MessageType


class FakeBus:
    def __init__(
        self,
        trusted=False,
        decrypted_body=None,
        *,
        chat_ack_available=False,
        auto_write=False,
        chat_connection_id=1,
    ):
        self.trusted = trusted
        self.decrypted_body = decrypted_body
        self.chat_ack_available = chat_ack_available
        self.chat_connection_id = chat_connection_id
        self.auto_write = auto_write
        self.decrypt_calls = []
        self.sent = []
        self.on_written_callbacks = []

    def send(
        self,
        message,
        *,
        secure=False,
        on_written=None,
        expected_connection_id=None,
    ):
        if (
            expected_connection_id is not None
            and expected_connection_id != self.chat_connection_id
        ):
            raise RuntimeError("serial connection changed")
        self.sent.append((message, secure))
        self.on_written_callbacks.append(on_written)
        if self.auto_write and on_written is not None:
            on_written()

    def decrypt_secure(self, message):
        self.decrypt_calls.append(message)
        if not self.trusted or self.decrypted_body is None:
            raise RuntimeError("secure payload was not authenticated")
        return self.decrypted_body


def test_plain_chat_sends_utf8_without_transport_encryption():
    bus = FakeBus()
    service = ChatService(bus)

    service.send_plain("한글 message ./?=+-_0)")

    message, secure = bus.sent[-1]
    assert message == Message(
        MessageType.CHAT_PLAIN,
        {},
        "한글 message ./?=+-_0)".encode(),
    )
    assert secure is False


def test_plain_chat_forwards_transport_write_completion_callback():
    bus = FakeBus()
    service = ChatService(bus)
    completed = []

    service.send_plain("written", on_written=lambda: completed.append(True))

    callback = bus.on_written_callbacks[-1]
    assert callback is not None
    callback()
    assert completed == [True]


def test_ack_capable_chat_reports_delivery_only_after_peer_ack():
    bus = FakeBus(chat_ack_available=True, auto_write=True)
    service = ChatService(bus)
    delivered = []

    service.send_plain("reliable", on_delivered=lambda: delivered.append(True))

    message = bus.sent[-1][0]
    message_id = message.metadata["message_id"]
    assert delivered == []
    assert service.handle_message(
        Message(MessageType.CHAT_PLAIN, {"ack_id": message_id}, b"")
    )
    assert delivered == [True]


def test_ack_capable_chat_deduplicates_retries_and_acknowledges_each_copy():
    bus = FakeBus(chat_ack_available=True)
    service = ChatService(bus)
    received = []
    service.add_message_listener(received.append)
    message = Message(
        MessageType.CHAT_PLAIN,
        {"message_id": "a" * 32},
        b"only once",
    )

    assert service.handle_message(message)
    assert service.handle_message(message)

    assert [item.text for item in received] == ["only once"]
    acknowledgements = [item[0] for item in bus.sent]
    assert [item.metadata for item in acknowledgements] == [
        {"ack_id": "a" * 32},
        {"ack_id": "a" * 32},
    ]


def test_ack_capable_chat_reports_failure_after_bounded_retries():
    bus = FakeBus(chat_ack_available=True, auto_write=True)
    service = ChatService(bus, retry_interval=0.01, max_attempts=2)
    failed = []
    failure_reported = threading.Event()

    service.send_plain(
        "never acknowledged",
        on_failed=lambda reason: (failed.append(reason), failure_reported.set()),
    )

    assert failure_reported.wait(1)
    assert len(bus.sent) == 2
    assert failed == ["Delivery failed"]


def test_legacy_chat_reports_delivery_when_local_write_completes():
    bus = FakeBus(auto_write=True)
    service = ChatService(bus)
    delivered = []

    service.send_plain("legacy", on_delivered=lambda: delivered.append(True))

    assert bus.sent[-1][0].metadata == {}
    assert delivered == [True]


def test_legacy_chat_cannot_cross_into_a_replacement_connection():
    class ReconnectingBus(FakeBus):
        def send(self, message, **kwargs):
            self.chat_connection_id = 2
            return super().send(message, **kwargs)

    bus = ReconnectingBus()
    service = ChatService(bus)

    with pytest.raises(RuntimeError, match="serial connection changed"):
        service.send_plain("old peer")

    assert bus.sent == []


def test_disconnect_cancels_pending_chat_retries():
    bus = FakeBus(chat_ack_available=True, auto_write=True)
    service = ChatService(bus, retry_interval=0.01, max_attempts=10)
    failed = []
    service.send_plain("disconnect", on_failed=failed.append)
    deadline = time.monotonic() + 1
    while len(bus.sent) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)

    service.disconnect()
    sent_count = len(bus.sent)
    time.sleep(0.05)

    assert len(bus.sent) == sent_count
    assert failed == ["Disconnected"]


def test_retry_cannot_cross_into_a_replacement_connection():
    class BlockingRetryBus(FakeBus):
        def __init__(self):
            super().__init__(chat_ack_available=True, auto_write=True)
            self.retry_started = threading.Event()
            self.release_retry = threading.Event()

        def send(self, message, **kwargs):
            if self.sent:
                self.retry_started.set()
                assert self.release_retry.wait(1)
            return super().send(message, **kwargs)

    bus = BlockingRetryBus()
    service = ChatService(bus, retry_interval=0.01, max_attempts=3)
    failed = []
    service.send_plain("old peer", on_failed=failed.append)
    assert bus.retry_started.wait(1)

    service.disconnect()
    bus.chat_connection_id = 2
    bus.release_retry.set()
    time.sleep(0.05)

    assert len(bus.sent) == 1
    assert failed == ["Disconnected"]


@pytest.mark.parametrize("retry_interval", [float("nan"), float("inf")])
def test_chat_rejects_non_finite_retry_intervals(retry_interval):
    with pytest.raises(ValueError):
        ChatService(FakeBus(), retry_interval=retry_interval)


def test_secure_chat_requires_trust():
    with pytest.raises(PeerNotTrusted):
        ChatService(FakeBus()).send_secure("secret")


def test_secure_chat_uses_transport_encryption():
    bus = FakeBus(trusted=True)
    service = ChatService(bus)

    service.send_secure("secret")

    message, secure = bus.sent[-1]
    assert message == Message(MessageType.CHAT_SECURE, {}, b"secret")
    assert secure is True


def test_chat_rejects_text_that_exceeds_the_wire_budget():
    service = ChatService(FakeBus())

    with pytest.raises(ChatTextTooLarge):
        service.send_plain("한" * 30_000)


def test_received_chat_updates_last_text_and_notifies_listeners():
    bus = FakeBus(trusted=True, decrypted_body="받은 메시지".encode())
    service = ChatService(bus)
    received = []
    service.add_message_listener(received.append)

    assert service.handle_message(
        Message(MessageType.CHAT_SECURE, {}, b"authenticated ciphertext")
    )

    assert service.last_text == "받은 메시지"
    assert received[-1].text == "받은 메시지"
    assert received[-1].secure is True
    assert len(bus.decrypt_calls) == 1


def test_secure_chat_is_rejected_when_authentication_fails():
    service = ChatService(FakeBus())

    assert not service.handle_message(
        Message(MessageType.CHAT_SECURE, {}, b"unauthenticated")
    )
    assert service.last_text == ""


def test_non_chat_messages_are_not_consumed():
    service = ChatService(FakeBus())

    assert not service.handle_message(Message(MessageType.FILE_OFFER, {}, b""))
    assert service.last_text == ""


def test_malformed_utf8_chat_is_rejected_without_losing_last_text():
    service = ChatService(FakeBus())
    service.handle_message(Message(MessageType.CHAT_PLAIN, {}, b"valid"))

    assert not service.handle_message(Message(MessageType.CHAT_PLAIN, {}, b"\xff"))
    assert service.last_text == "valid"
