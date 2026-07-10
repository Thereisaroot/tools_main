import struct
import zlib

import pytest

from shooklink.protocol.framing import (
    HEADER,
    MAGIC,
    MAX_ENCODED_FRAME_SIZE,
    MAX_PAYLOAD_SIZE,
    VERSION,
    Frame,
    FrameDecodeError,
    FrameParser,
    cobs_decode,
    cobs_encode,
    decode_frame,
    encode_frame,
)


def _packet_from_decoded(decoded: bytes) -> bytes:
    return cobs_encode(decoded)


def _replace_crc(decoded_without_crc: bytes) -> bytes:
    crc = zlib.crc32(decoded_without_crc) & 0xFFFFFFFF
    return decoded_without_crc + struct.pack(">I", crc)


def test_frame_round_trip_with_zero_bytes_and_all_fields():
    frame = Frame(
        message_type=7,
        flags=3,
        priority=2,
        stream_id=0x10203040,
        sequence=0x50607080,
        acknowledgement=0x90A0B0C0,
        payload=b"a\x00b\x00",
    )

    encoded = encode_frame(frame)

    assert encoded.endswith(b"\x00")
    assert b"\x00" not in encoded[:-1]
    assert decode_frame(encoded[:-1]) == frame


def test_empty_payload_round_trip():
    frame = Frame(1, 0, 0, 0, 0, 0, b"")

    assert decode_frame(encode_frame(frame)[:-1]) == frame


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00",
        b"\x00\x00\x01\x00",
        bytes(range(1, 255)),
        bytes(range(1, 255)) * 2,
        b"prefix\x00" + bytes(range(1, 255)) + b"\x00suffix",
    ],
)
def test_cobs_round_trip(payload):
    encoded = cobs_encode(payload)

    assert encoded
    assert b"\x00" not in encoded
    assert cobs_decode(encoded) == payload


@pytest.mark.parametrize("packet", [b"", b"\x00", b"\x02", b"\x03a", b"\x01\x00"])
def test_invalid_cobs_packet_is_rejected(packet):
    with pytest.raises(FrameDecodeError):
        cobs_decode(packet)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("message_type", -1),
        ("message_type", 256),
        ("flags", -1),
        ("flags", 256),
        ("priority", -1),
        ("priority", 256),
        ("stream_id", -1),
        ("stream_id", 1 << 32),
        ("sequence", -1),
        ("sequence", 1 << 32),
        ("acknowledgement", -1),
        ("acknowledgement", 1 << 32),
        ("sequence", True),
    ],
)
def test_encode_rejects_invalid_integer_fields(field, value):
    values = {
        "message_type": 1,
        "flags": 0,
        "priority": 0,
        "stream_id": 0,
        "sequence": 0,
        "acknowledgement": 0,
        "payload": b"ok",
    }
    values[field] = value

    with pytest.raises(ValueError):
        encode_frame(Frame(**values))


def test_encode_rejects_oversized_or_non_bytes_payload():
    with pytest.raises(ValueError):
        encode_frame(Frame(1, 0, 0, 0, 0, 0, b"x" * (MAX_PAYLOAD_SIZE + 1)))

    with pytest.raises(TypeError):
        encode_frame(Frame(1, 0, 0, 0, 0, 0, "text"))


def test_decode_rejects_crc_corruption():
    decoded = bytearray(cobs_decode(encode_frame(Frame(1, 0, 0, 0, 0, 0, b"data"))[:-1]))
    decoded[HEADER.size] ^= 0x01

    with pytest.raises(FrameDecodeError, match="CRC"):
        decode_frame(_packet_from_decoded(bytes(decoded)))


@pytest.mark.parametrize("change", ["magic", "version", "length"])
def test_decode_rejects_bad_header(change):
    decoded = bytearray(cobs_decode(encode_frame(Frame(1, 0, 0, 0, 0, 0, b"data"))[:-1]))
    header = list(HEADER.unpack(decoded[: HEADER.size]))
    if change == "magic":
        header[0] = b"XX"
    elif change == "version":
        header[1] = VERSION + 1
    else:
        header[-1] += 1
    body = HEADER.pack(*header) + decoded[HEADER.size : -4]
    packet = _packet_from_decoded(_replace_crc(body))

    with pytest.raises(FrameDecodeError):
        decode_frame(packet)


def test_decode_rejects_truncated_frame():
    header = HEADER.pack(MAGIC, VERSION, 1, 0, 0, 0, 0, 0, 0)

    with pytest.raises(FrameDecodeError):
        decode_frame(_packet_from_decoded(header))


def test_parser_handles_fragmentation_and_multiple_frames():
    first = encode_frame(Frame(1, 0, 0, 1, 1, 0, b"first"))
    second = encode_frame(Frame(2, 0, 1, 2, 3, 1, b"second"))
    parser = FrameParser()

    frames = parser.feed(first[:3])
    frames += parser.feed(first[3:] + b"\x00" + second[:5])
    frames += parser.feed(second[5:])

    assert [frame.payload for frame in frames] == [b"first", b"second"]
    assert parser.malformed_frames == 0


def test_parser_resynchronizes_after_corrupt_frame():
    valid = encode_frame(Frame(1, 0, 0, 0, 1, 0, b"ok"))
    parser = FrameParser()

    frames = parser.feed(b"not-cobs\x00" + valid)

    assert [frame.payload for frame in frames] == [b"ok"]
    assert parser.malformed_frames == 1


def test_parser_bounds_unterminated_input_and_recovers():
    parser = FrameParser()
    oversized = b"x" * (MAX_ENCODED_FRAME_SIZE + 100)
    valid = encode_frame(Frame(1, 0, 0, 0, 1, 0, b"recovered"))

    assert parser.feed(oversized) == []
    assert parser.buffered_bytes <= MAX_ENCODED_FRAME_SIZE
    assert parser.dropped_bytes == len(oversized)
    assert parser.malformed_frames == 1

    frames = parser.feed(b"\x00" + valid)

    assert [frame.payload for frame in frames] == [b"recovered"]
