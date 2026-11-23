import pytest

from shooklink.core import CoreState
from shooklink.live_update.release import FEATURE
from shooklink.live_update.wire import Unavailable
from shooklink.protocol.messages import MessageType
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
        with pytest.raises(Unavailable, match="bootstrap"):
            sender.updates.exchange(MessageType.UPDATE_PROBE, {"job": ""})
        assert "files" in sender.snapshot.features and sender.trusted
    finally:
        sender.close()
        old.close()
