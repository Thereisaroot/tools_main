"""Binary framing with COBS delimiting and CRC32 integrity checks."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

MAGIC = b"SL"
VERSION = 1
SECURE_FLAG = 0x01
MAX_PAYLOAD_SIZE = 65_535

HEADER = struct.Struct(">2sBBBBIIII")
CRC = struct.Struct(">I")
MAX_DECODED_FRAME_SIZE = HEADER.size + MAX_PAYLOAD_SIZE + CRC.size
MAX_ENCODED_FRAME_SIZE = (
    MAX_DECODED_FRAME_SIZE + MAX_DECODED_FRAME_SIZE // 254 + 1
)


class FrameDecodeError(ValueError):
    """Raised when bytes do not contain a valid ShookLink frame."""


@dataclass(frozen=True, slots=True)
class Frame:
    message_type: int
    flags: int
    priority: int
    stream_id: int
    sequence: int
    acknowledgement: int
    payload: bytes


def cobs_encode(data: bytes) -> bytes:
    """Encode bytes using Consistent Overhead Byte Stuffing."""
    if not isinstance(data, bytes):
        raise TypeError("COBS input must be bytes")

    encoded = bytearray(b"\x00")
    code_index = 0
    code = 1

    for index, value in enumerate(data):
        if value == 0:
            encoded[code_index] = code
            code_index = len(encoded)
            encoded.append(0)
            code = 1
            continue

        encoded.append(value)
        code += 1
        if code == 0xFF:
            encoded[code_index] = code
            if index == len(data) - 1:
                return bytes(encoded)
            code_index = len(encoded)
            encoded.append(0)
            code = 1

    encoded[code_index] = code
    return bytes(encoded)


def cobs_decode(packet: bytes) -> bytes:
    """Decode one COBS packet that does not include its delimiter."""
    if not isinstance(packet, bytes):
        raise TypeError("COBS packet must be bytes")
    if not packet:
        raise FrameDecodeError("empty COBS packet")
    if 0 in packet:
        raise FrameDecodeError("COBS packet contains a zero byte")

    decoded = bytearray()
    index = 0
    packet_size = len(packet)

    while index < packet_size:
        code = packet[index]
        index += 1
        block_end = index + code - 1
        if block_end > packet_size:
            raise FrameDecodeError("truncated COBS block")

        decoded.extend(packet[index:block_end])
        index = block_end
        if code < 0xFF and index < packet_size:
            decoded.append(0)

    return bytes(decoded)


def _validate_uint(name: str, value: int, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 0 to {maximum}")


def pack_frame_header(
    *,
    message_type: int,
    flags: int,
    priority: int,
    stream_id: int,
    sequence: int,
    acknowledgement: int,
    payload_length: int,
) -> bytes:
    _validate_uint("message_type", message_type, 0xFF)
    _validate_uint("flags", flags, 0xFF)
    _validate_uint("priority", priority, 0xFF)
    _validate_uint("stream_id", stream_id, 0xFFFFFFFF)
    _validate_uint("sequence", sequence, 0xFFFFFFFF)
    _validate_uint("acknowledgement", acknowledgement, 0xFFFFFFFF)
    _validate_uint("payload_length", payload_length, MAX_PAYLOAD_SIZE)
    return HEADER.pack(
        MAGIC,
        VERSION,
        message_type,
        flags,
        priority,
        stream_id,
        sequence,
        acknowledgement,
        payload_length,
    )


def encode_frame(frame: Frame) -> bytes:
    """Encode a validated frame and append the stream delimiter."""
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a Frame")

    _validate_uint("message_type", frame.message_type, 0xFF)
    _validate_uint("flags", frame.flags, 0xFF)
    _validate_uint("priority", frame.priority, 0xFF)
    _validate_uint("stream_id", frame.stream_id, 0xFFFFFFFF)
    _validate_uint("sequence", frame.sequence, 0xFFFFFFFF)
    _validate_uint("acknowledgement", frame.acknowledgement, 0xFFFFFFFF)
    if not isinstance(frame.payload, bytes):
        raise TypeError("payload must be bytes")
    if len(frame.payload) > MAX_PAYLOAD_SIZE:
        raise ValueError(f"payload cannot exceed {MAX_PAYLOAD_SIZE} bytes")

    header = pack_frame_header(
        message_type=frame.message_type,
        flags=frame.flags,
        priority=frame.priority,
        stream_id=frame.stream_id,
        sequence=frame.sequence,
        acknowledgement=frame.acknowledgement,
        payload_length=len(frame.payload),
    )
    checked = header + frame.payload
    checksum = CRC.pack(zlib.crc32(checked) & 0xFFFFFFFF)
    return cobs_encode(checked + checksum) + b"\x00"


def decode_frame(packet: bytes) -> Frame:
    """Decode and validate one COBS packet without its delimiter."""
    if not isinstance(packet, bytes):
        raise TypeError("encoded frame must be bytes")
    if len(packet) > MAX_ENCODED_FRAME_SIZE:
        raise FrameDecodeError("encoded frame exceeds the maximum size")

    try:
        decoded = cobs_decode(packet)
    except FrameDecodeError:
        raise
    except (TypeError, ValueError) as error:
        raise FrameDecodeError(str(error)) from error

    if len(decoded) < HEADER.size + CRC.size:
        raise FrameDecodeError("frame is too short")
    if len(decoded) > MAX_DECODED_FRAME_SIZE:
        raise FrameDecodeError("frame exceeds the maximum size")

    try:
        (
            magic,
            version,
            message_type,
            flags,
            priority,
            stream_id,
            sequence,
            acknowledgement,
            payload_length,
        ) = HEADER.unpack(decoded[: HEADER.size])
    except struct.error as error:
        raise FrameDecodeError("invalid frame header") from error

    if magic != MAGIC:
        raise FrameDecodeError("invalid frame magic")
    if version != VERSION:
        raise FrameDecodeError("unsupported protocol version")
    if payload_length > MAX_PAYLOAD_SIZE:
        raise FrameDecodeError("payload exceeds the maximum size")

    expected_size = HEADER.size + payload_length + CRC.size
    if len(decoded) != expected_size:
        raise FrameDecodeError("frame payload length does not match the header")

    checked = decoded[:-CRC.size]
    expected_crc = CRC.unpack(decoded[-CRC.size :])[0]
    actual_crc = zlib.crc32(checked) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise FrameDecodeError("frame CRC mismatch")

    return Frame(
        message_type=message_type,
        flags=flags,
        priority=priority,
        stream_id=stream_id,
        sequence=sequence,
        acknowledgement=acknowledgement,
        payload=decoded[HEADER.size : -CRC.size],
    )


class FrameParser:
    """Incrementally parse a serial byte stream into validated frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._discarding = False
        self.malformed_frames = 0
        self.dropped_bytes = 0

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, data: bytes) -> list[Frame]:
        if not isinstance(data, bytes):
            raise TypeError("parser input must be bytes")

        frames: list[Frame] = []
        for value in data:
            if self._discarding:
                if value == 0:
                    self._discarding = False
                else:
                    self.dropped_bytes += 1
                continue

            if value == 0:
                if not self._buffer:
                    continue
                packet = bytes(self._buffer)
                self._buffer.clear()
                try:
                    frames.append(decode_frame(packet))
                except FrameDecodeError:
                    self.malformed_frames += 1
                    self.dropped_bytes += len(packet)
                continue

            if len(self._buffer) >= MAX_ENCODED_FRAME_SIZE:
                self.dropped_bytes += len(self._buffer) + 1
                self._buffer.clear()
                self._discarding = True
                self.malformed_frames += 1
                continue

            self._buffer.append(value)

        return frames


__all__ = [
    "HEADER",
    "MAGIC",
    "MAX_ENCODED_FRAME_SIZE",
    "MAX_PAYLOAD_SIZE",
    "SECURE_FLAG",
    "VERSION",
    "Frame",
    "FrameDecodeError",
    "FrameParser",
    "cobs_decode",
    "cobs_encode",
    "decode_frame",
    "encode_frame",
    "pack_frame_header",
]
