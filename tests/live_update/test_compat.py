import queue

import pytest

from shooklink import __version__
from shooklink.core import CoreState
from shooklink.live_update.release import FEATURE
from shooklink.live_update.wire import Unavailable
from shooklink.protocol.messages import Message, MessageType
from shooklink.transport.multiplexer import Priority
from tests.integration.test_peer_pair import approve_pair, build_core, endpoint_pair, wait_for


def test_legacy_peer_is_detected_before_live_update_while_plain_files_remain_available(tmp_path):
    sender, *_ = build_core(tmp_path, "sender")
    old, *_ = build_core(tmp_path, "legacy")
    features = old._local_features() - {FEATURE}
    old._local_features = lambda: features
    try:
        first, second = endpoint_pair()
        sender.connect_endpoint(first)
        old.connect_endpoint(second)
        assert wait_for(lambda: sender.snapshot.state is old.snapshot.state is CoreState.UNTRUSTED)
        approve_pair(sender, old)
        assert sender.snapshot.peer_version == __version__
        assert sender.snapshot.peer_revision is None
        with pytest.raises(Unavailable, match="bootstrap"):
            sender.updates.exchange(MessageType.UPDATE_PROBE, {"job": ""})
        assert "files" in sender.snapshot.features and sender.trusted
    finally:
        sender.close()
        old.close()


def test_ready_does_not_wait_for_build_and_only_displays_the_requested_running_revision(tmp_path):
    sender, *_ = build_core(tmp_path, "viewer")
    peer, *_ = build_core(tmp_path, "peer")
    probes = queue.Queue()
    handle = peer.updates.handle_message

    def hold_probe(message):
        if message.message_type is MessageType.UPDATE_PROBE:
            probes.put(message.metadata["nonce"])
        else:
            handle(message)

    peer.updates.handle_message = hold_probe
    try:
        a, b = endpoint_pair()
        sender.connect_endpoint(a)
        peer.connect_endpoint(b)
        assert wait_for(lambda: sender.snapshot.state is peer.snapshot.state is CoreState.UNTRUSTED)
        approve_pair(sender, peer)
        nonce = probes.get(timeout=2)
        assert sender.snapshot.peer_version == __version__
        assert sender.snapshot.peer_revision is None

        for reply_nonce, revision in (("0" * 32, "b" * 40), (nonce, "invalid"), (nonce, "a" * 40)):
            peer.send(
                Message(MessageType.UPDATE_REPORT, {
                    "nonce": reply_nonce, "ok": True,
                    "result": {"revision": revision, "target": "c" * 40},
                }),
                secure=True, priority=Priority.INTERACTIVE,
            )
        assert wait_for(lambda: sender.snapshot.peer_revision == "a" * 40)
    finally:
        sender.close()
        peer.close()
