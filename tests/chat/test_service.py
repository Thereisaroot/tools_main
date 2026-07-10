import pytest

from shooklink.chat.service import (
    ChatService,
    ChatTextTooLarge,
    PeerNotTrusted,
)
from shooklink.protocol.messages import Message, MessageType


class FakeBus:
    def __init__(self, trusted=False):
        self.trusted = trusted
        self.sent = []

    def send(self, message, *, secure=False):
        self.sent.append((message, secure))


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
    service = ChatService(FakeBus())
    received = []
    service.add_message_listener(received.append)

    assert service.handle_message(
        Message(MessageType.CHAT_SECURE, {}, "받은 메시지".encode())
    )

    assert service.last_text == "받은 메시지"
    assert service.last_secure_text == "받은 메시지"
    assert received[-1].text == "받은 메시지"
    assert received[-1].secure is True


def test_non_chat_messages_are_not_consumed():
    service = ChatService(FakeBus())

    assert not service.handle_message(Message(MessageType.FILE_OFFER, {}, b""))
    assert service.last_text == ""


def test_malformed_utf8_chat_is_rejected_without_losing_last_text():
    service = ChatService(FakeBus())
    service.handle_message(Message(MessageType.CHAT_PLAIN, {}, b"valid"))

    assert not service.handle_message(Message(MessageType.CHAT_PLAIN, {}, b"\xff"))
    assert service.last_text == "valid"
