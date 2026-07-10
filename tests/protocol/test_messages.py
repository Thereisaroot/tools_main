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


@pytest.mark.parametrize(
    "metadata",
    [
        b'{"type":10,"type":11,"meta":{}}',
        b'{"type":10,"meta":{"value":1,"value":2}}',
    ],
)
def test_duplicate_json_keys_are_rejected(metadata):
    encoded = len(metadata).to_bytes(4, "big") + metadata

    with pytest.raises(MessageDecodeError, match="duplicate"):
        decode_message(encoded)


def test_pathological_json_errors_stay_inside_decode_boundary():
    deep = b'{"type":10,"meta":{"value":' + b"[" * 1_100 + b"0" + b"]" * 1_100 + b"}}"
    huge_integer = b'{"type":10,"meta":{"value":' + b"9" * 5_000 + b"}}"

    for metadata in (deep, huge_integer):
        encoded = len(metadata).to_bytes(4, "big") + metadata
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
        "HELLO": 1,
        "TRUST": 2,
        "CHAT_PLAIN": 10,
        "CHAT_SECURE": 11,
        "FILE_OFFER": 20,
        "FILE_ACCEPT": 21,
        "FILE_CHUNK": 22,
        "FILE_ACK": 23,
        "FILE_FINISH": 24,
        "FILE_CANCEL": 25,
        "SHELL_OPEN": 30,
        "SHELL_ACCEPT": 31,
        "SHELL_DENY": 32,
        "SHELL_INPUT": 33,
        "SHELL_OUTPUT": 34,
        "SHELL_RESIZE": 35,
        "SHELL_EXIT": 36,
        "INPUT_REQUEST": 40,
        "INPUT_ACCEPT": 41,
        "INPUT_BUSY": 42,
        "INPUT_ENTER": 43,
        "INPUT_LEAVE": 44,
        "INPUT_KEY": 45,
        "INPUT_BUTTON": 46,
        "INPUT_MOVE": 47,
        "INPUT_WHEEL": 48,
        "INPUT_POINTER_STATE": 49,
        "INPUT_STOP": 50,
        "INPUT_RELEASE_ALL": 51,
    }

    assert required == {member.name: member.value for member in MessageType}
