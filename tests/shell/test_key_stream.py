import threading

import pytest

from shooklink.protocol.messages import Message, MessageType
from shooklink.shell.key_stream import KeyReader
from tests.shell.test_service import FakeBus, FakeProcessFactory, ShellService, shell_message


class KeyBus(FakeBus):
    reliable_shell_input = True


def connected_services():
    a, b = KeyBus(), KeyBus()
    processes = FakeProcessFactory()
    viewer, host = ShellService(a), ShellService(b, processes)
    host.set_allow_remote_shell(True)
    sid = viewer.open_remote(columns=80, rows=24)
    host.handle_message(a.sent.pop()[0])
    viewer.handle_message(b.sent.pop()[0])
    return viewer, host, a, b, processes.processes[0], sid


def test_lost_input_and_lost_ack_recover_without_duplicate_enter():
    viewer, host, a, b, proc, sid = connected_services()
    try:
        arrived = threading.Event()
        def capture(msg, **kwargs):
            a.sent.append((msg, True, None))
            arrived.set()
        a.send = capture
        viewer.send_input(sid, b"a")
        assert arrived.wait(1)
        lost = a.sent.pop()[0]
        arrived.clear()
        viewer.send_input(sid, b"\r")
        # The first key was lost. The retry must cover it and the trailing Enter.
        assert arrived.wait(1)
        recovery = a.sent[-1][0]
        assert host.handle_message(recovery)
        arrived.clear()
        assert arrived.wait(2)  # Idle retry even if this was the very last key.
        assert host.handle_message(lost)
        assert host.handle_message(recovery)  # previous ACK was lost
        assert b"".join(proc.writes) == b"a\r"
        assert viewer.handle_message(b.sent[-1][0])
    finally:
        viewer.close()
        host.close()


def test_legacy_peer_gets_one_plain_shell_input_without_retries():
    bus = FakeBus()
    viewer = ShellService(bus)
    sid = viewer.open_remote(columns=80, rows=24)
    viewer.handle_message(shell_message(MessageType.SHELL_ACCEPT, sid))
    viewer.send_input(sid, b"ab\r")
    assert bus.sent[-1][0] == Message(MessageType.SHELL_INPUT, {"session_id": sid}, b"ab\r")
    viewer.close()


def test_gap_overlap_invalid_range_and_ambiguous_write():
    reader = KeyReader()
    writes = []
    assert reader.receive(1, b"b", writes.append) == 0 and not writes
    assert reader.receive(0, b"ab", writes.append) == 2
    assert reader.receive(1, b"b\r", writes.append) == 3
    assert writes == [b"ab", b"\r"]
    for offset in (-1, True, 1 << 54):
        with pytest.raises(ValueError):
            reader.receive(offset, b"x", writes.append)
    def partial(_):
        raise OSError("write interrupted")
    with pytest.raises(OSError):
        reader.receive(3, b"more", partial)
    with pytest.raises(OSError):
        reader.receive(3, b"more", writes.append)
    assert writes == [b"ab", b"\r"]


def test_lost_acks_eventually_close_rather_than_replaying_into_next_session(monkeypatch):
    from shooklink.shell import key_stream
    monkeypatch.setattr(key_stream, "MIN_STALL_SECONDS", 0)
    viewer, host, a, _, proc, sid = connected_services()
    failed = threading.Event()
    viewer.add_state_listener(lambda s: failed.set() if s.reason == "input_unconfirmed" else None)
    try:
        viewer.send_input(sid, b"\r")
        assert failed.wait(3)
        assert viewer.active_session_id is None
        keys = [msg for msg, _, _ in a.sent if msg.message_type is MessageType.SHELL_KEYS]
        assert len(keys) > 1
        assert host.handle_message(a.sent[-1][0])
        for msg in keys:
            assert not host.handle_message(msg)
        assert not proc.writes
    finally:
        viewer.close()
        host.close()


def test_receiver_does_not_accept_unsequenced_input_in_reliable_session():
    viewer, host, _, _, proc, sid = connected_services()
    try:
        assert not host.handle_message(shell_message(MessageType.SHELL_INPUT, sid, b"x"))
        assert not proc.writes
    finally:
        viewer.close()
        host.close()
