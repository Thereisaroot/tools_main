import pytest

from shooklink.chat.service import (
    ChatService,
    ChatTextTooLarge,
    PeerNotTrusted,
)
from shooklink.protocol.messages import Message, MessageType


class FakeBus:
    def __init__(self, trusted=False, decrypted_body=None):
        self.trusted = trusted
        self.decrypted_body = decrypted_body
        self.decrypt_calls = []
        self.sent = []
        self.on_written_callbacks = []

    def send(self, message, *, secure=False, on_written=None):
        self.sent.append((message, secure))
        self.on_written_callbacks.append(on_written)

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
