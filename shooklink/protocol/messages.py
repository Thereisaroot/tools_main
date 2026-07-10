"""Typed ShookLink feature messages carried inside binary frames."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from shooklink.protocol.framing import MAX_PAYLOAD_SIZE

METADATA_LENGTH = struct.Struct(">I")
MAX_METADATA_SIZE = 16 * 1024


class MessageDecodeError(ValueError):
    """Raised when a typed-message payload is malformed."""


class MessageType(IntEnum):
    HELLO = 1
    TRUST = 2

    CHAT_PLAIN = 10
    CHAT_SECURE = 11

    FILE_OFFER = 20
    FILE_ACCEPT = 21
    FILE_CHUNK = 22
    FILE_ACK = 23
    FILE_FINISH = 24
    FILE_CANCEL = 25

    SHELL_OPEN = 30
    SHELL_ACCEPT = 31
    SHELL_DENY = 32
    SHELL_INPUT = 33
    SHELL_OUTPUT = 34
    SHELL_RESIZE = 35
    SHELL_EXIT = 36

    INPUT_REQUEST = 40
    INPUT_ACCEPT = 41
    INPUT_BUSY = 42
    INPUT_ENTER = 43
    INPUT_LEAVE = 44
    INPUT_KEY = 45
    INPUT_BUTTON = 46
    INPUT_MOVE = 47
    INPUT_WHEEL = 48
    INPUT_POINTER_STATE = 49
    INPUT_STOP = 50
    INPUT_RELEASE_ALL = 51


@dataclass(frozen=True, slots=True)
class Message:
    message_type: MessageType
    metadata: dict[str, Any]
    body: bytes = b""


def _validate_json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("metadata floats must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("metadata object keys must be strings")
        for item in value.values():
            _validate_json_value(item)
        return
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def encode_message(message: Message) -> bytes:
    if not isinstance(message, Message):
        raise TypeError("message must be a Message")
    if not isinstance(message.message_type, MessageType):
        raise TypeError("message_type must be a MessageType")
    if not isinstance(message.metadata, dict):
        raise TypeError("metadata must be an object")
    _validate_json_value(message.metadata)
    if not isinstance(message.body, bytes):
        raise TypeError("message body must be bytes")

    envelope = {"meta": message.metadata, "type": int(message.message_type)}
    try:
        metadata = json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("metadata is not JSON serializable") from error

    if not metadata or len(metadata) > MAX_METADATA_SIZE:
        raise ValueError(f"metadata cannot exceed {MAX_METADATA_SIZE} bytes")
    encoded = METADATA_LENGTH.pack(len(metadata)) + metadata + message.body
    if len(encoded) > MAX_PAYLOAD_SIZE:
        raise ValueError(f"encoded message cannot exceed {MAX_PAYLOAD_SIZE} bytes")
    return encoded


def decode_message(encoded: bytes) -> Message | None:
    if not isinstance(encoded, bytes):
        raise TypeError("encoded message must be bytes")
    if len(encoded) < METADATA_LENGTH.size:
        raise MessageDecodeError("message is missing its metadata length")
    if len(encoded) > MAX_PAYLOAD_SIZE:
        raise MessageDecodeError("message exceeds the maximum size")

    metadata_length = METADATA_LENGTH.unpack(encoded[: METADATA_LENGTH.size])[0]
    if not 0 < metadata_length <= MAX_METADATA_SIZE:
        raise MessageDecodeError("invalid metadata length")
    metadata_end = METADATA_LENGTH.size + metadata_length
    if metadata_end > len(encoded):
        raise MessageDecodeError("truncated message metadata")

    try:
        envelope = json.loads(encoded[METADATA_LENGTH.size : metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MessageDecodeError("invalid message metadata") from error
    if not isinstance(envelope, dict) or set(envelope) != {"meta", "type"}:
        raise MessageDecodeError("invalid message envelope")
    if type(envelope["type"]) is not int:
        raise MessageDecodeError("message type must be an integer")
    if not isinstance(envelope["meta"], dict):
        raise MessageDecodeError("message metadata must be an object")
    try:
        _validate_json_value(envelope["meta"])
    except TypeError as error:
        raise MessageDecodeError(str(error)) from error

    try:
        message_type = MessageType(envelope["type"])
    except ValueError:
        return None
    return Message(message_type, envelope["meta"], encoded[metadata_end:])


__all__ = [
    "MAX_METADATA_SIZE",
    "Message",
    "MessageDecodeError",
    "MessageType",
    "decode_message",
    "encode_message",
]
