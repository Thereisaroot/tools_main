import random
import zlib

import pytest

from shooklink.protocol.messages import MessageType
from shooklink.shell.codec import decode_output, output_message
from shooklink.shell.service import MAX_SHELL_IO_BYTES

SID = "a" * 32


def test_only_compress_large_usefully_compressible_output_for_capable_peers():
    log = b"build: all tests passed\r\n" * 400
    for data, capable in ((b"prompt> ", True), (log, False), (random.Random(7).randbytes(4096), True)):
        message = output_message(SID, data, capable)
        assert message.metadata == {"session_id": SID} and message.body == data
        assert decode_output(message, capable) == data
    compressed = output_message(SID, log, True)
    assert compressed.message_type is MessageType.SHELL_OUTPUT
    assert len(compressed.body) < len(log) // 5
    assert decode_output(compressed, True) == log
    with pytest.raises(ValueError):
        decode_output(compressed, False)


def test_compressed_output_never_exceeds_one_shell_chunk_or_accepts_trailing_data():
    from dataclasses import replace

    msg = output_message(SID, b"a" * MAX_SHELL_IO_BYTES, True)
    assert len(decode_output(msg, True)) == MAX_SHELL_IO_BYTES
    for body in (msg.body[:-1], msg.body + b"x", zlib.compress(b"x" * (MAX_SHELL_IO_BYTES + 1))):
        with pytest.raises(ValueError):
            decode_output(replace(msg, body=body), True)
