import json

import pytest

from shooklink.protocol.messages import (
    MAX_METADATA_SIZE,
    Message,
    MessageDecodeError,
    MessageType,
    decode_message,
    encode_message,
)


def test_message_metadata_and_binary_body_round_trip():
    original = Message(
        MessageType.FILE_CHUNK,
        {"transfer_id": "abc", "index": 2, "name": "한글.txt"},
        b"\x00data\xff",
    )

    assert decode_message(encode_message(original)) == original


def test_message_encoding_is_deterministic():
    first = Message(MessageType.CHAT_PLAIN, {"z": 1, "a": 2}, b"hello")
    second = Message(MessageType.CHAT_PLAIN, {"a": 2, "z": 1}, b"hello")

    assert encode_message(first) == encode_message(second)


def test_unknown_message_type_returns_none():
    metadata = json.dumps({"type": 999, "meta": {}}).encode()
    encoded = len(metadata).to_bytes(4, "big") + metadata

    assert decode_message(encoded) is None


@pytest.mark.parametrize(
    "encoded",
    [
        b"",
        b"\x00\x00\x00",
        b"\x00\x00\x00\x10{}",
        b"\x00\x00\x00\x01{",
    ],
)
def test_malformed_message_is_rejected(encoded):
    with pytest.raises(MessageDecodeError):
        decode_message(encoded)


def test_metadata_must_be_an_object_with_string_keys():
    with pytest.raises(TypeError):
        encode_message(Message(MessageType.CHAT_PLAIN, ["not", "object"], b""))

    with pytest.raises(TypeError):
        encode_message(Message(MessageType.CHAT_PLAIN, {1: "bad key"}, b""))


def test_body_must_be_bytes():
    with pytest.raises(TypeError):
        encode_message(Message(MessageType.CHAT_PLAIN, {}, "not bytes"))


def test_metadata_size_is_bounded():
    message = Message(MessageType.CHAT_PLAIN, {"value": "x" * MAX_METADATA_SIZE}, b"")

    with pytest.raises(ValueError, match="metadata"):
        encode_message(message)


def test_all_required_message_types_are_stable_and_unique():
    required = {
        "HELLO",
        "TRUST_DECISION",
        "CHAT_PLAIN",
        "CHAT_SECURE",
        "FILE_OFFER",
        "FILE_ACCEPT",
        "FILE_CHUNK",
        "FILE_ACK",
        "FILE_FINISH",
        "FILE_CANCEL",
        "SHELL_OPEN",
        "SHELL_ACCEPT",
        "SHELL_DENY",
        "SHELL_INPUT",
        "SHELL_OUTPUT",
        "SHELL_RESIZE",
        "SHELL_EXIT",
        "INPUT_REQUEST",
        "INPUT_ACCEPT",
        "INPUT_BUSY",
        "INPUT_ENTER",
        "INPUT_LEAVE",
        "INPUT_KEY",
        "INPUT_BUTTON",
        "INPUT_MOVE",
        "INPUT_WHEEL",
        "INPUT_POINTER_STATE",
        "INPUT_STOP",
        "INPUT_RELEASE_ALL",
    }

    assert required == {member.name for member in MessageType}
    assert len({member.value for member in MessageType}) == len(required)
