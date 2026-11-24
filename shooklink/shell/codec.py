"""Independent, bounded compression of shell output, before transport encryption."""

import zlib

from shooklink.protocol.messages import Message, MessageType

FEATURE = "shell_output_zlib_v1"
MAX_SHELL_IO_BYTES = 48 * 1024


def output_message(session_id: str, data: bytes, capable: bool) -> Message:
    fields = {"session_id": session_id}
    if len(data) > MAX_SHELL_IO_BYTES:
        raise ValueError("shell output exceeds the frame budget")
    if capable and len(data) >= 256:
        compressed = zlib.compress(data, level=1)
        if len(compressed) + 48 < len(data):
            return Message(MessageType.SHELL_OUTPUT, dict(fields, codec="zlib", raw_size=len(data)), compressed)
    return Message(MessageType.SHELL_OUTPUT, fields, data)


def decode_output(message: Message, capable: bool) -> bytes:
    fields, data = message.metadata, message.body
    if len(data) > MAX_SHELL_IO_BYTES:
        raise ValueError("shell output exceeds the frame budget")
    if set(fields) == {"session_id"}:
        return data
    size = fields.get("raw_size")
    if (not capable or set(fields) != {"session_id", "codec", "raw_size"}
            or fields["codec"] != "zlib" or type(size) is not int or not 1 <= size <= MAX_SHELL_IO_BYTES):
        raise ValueError("invalid shell output encoding")
    decoder = zlib.decompressobj()
    try:
        result = decoder.decompress(data, size + 1)
    except zlib.error as error:
        raise ValueError("invalid compressed shell output") from error
    if len(result) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("invalid compressed shell output length")
    return result
