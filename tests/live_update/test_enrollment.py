from dataclasses import replace
from zipfile import ZipFile

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PySide6.QtCore import Qt

from shooklink.core import CoreError, CoreState
from shooklink.input.service import InputSessionState
from shooklink.live_update import enrollment
from shooklink.live_update.manager import ManagedReceiver
from shooklink.live_update.release import FEATURE, Release
from shooklink.ui.main_window import MainWindow
from tests.integration.test_peer_pair import approve_pair, build_core, endpoint_pair, wait_for


def test_local_click_enrolls_only_the_connected_trusted_publisher(tmp_path, qtbot):
    signer = enrollment.local_signer(tmp_path / "mac-data")
    pub = signer.public_bytes.hex()
    assert enrollment.local_signer(tmp_path / "mac-data").public_bytes == signer.public_bytes
    mac, *_ = build_core(tmp_path, "mac", update_publisher=pub)
    home = tmp_path / "windows-data"
    home.mkdir()
    receiver = ManagedReceiver(home)
    win, *_ = build_core(tmp_path, "windows", update_receiver=receiver)
    window = MainWindow(win.chat)
    qtbot.addWidget(window)
    window.allow_updates_requested.connect(win.allow_peer_updates)
    window.show()
    try:
        a, b = endpoint_pair()
        mac.connect_endpoint(a)
        win.connect_endpoint(b)
        assert wait_for(lambda: win.snapshot.state is mac.snapshot.state is CoreState.UNTRUSTED)
        window.apply_core_snapshot(win.snapshot)
        assert window.allow_updates_button.isHidden() and not window.allow_updates_button.isEnabled()
        with pytest.raises(CoreError):
            win.allow_peer_updates(win.snapshot)
        approve_pair(win, mac)
        ready = win.snapshot
        assert ready.peer_publisher == pub and mac.identity.public_bytes != signer.public_bytes
        window.apply_core_snapshot(replace(ready, features=frozenset({FEATURE})))
        assert window.allow_updates_button.text() == "Update Peer App First"
        assert not window.allow_updates_button.isEnabled()
        assert not receiver.report()["managed"] and not (home / "publisher").exists()
        with pytest.raises(CoreError):
            win.allow_peer_updates(replace(ready, connection_id=ready.connection_id + 1))
        with pytest.raises(CoreError):
            win.allow_peer_updates(replace(ready, fingerprint="another peer"))
        with pytest.raises(CoreError):
            win.allow_peer_updates(replace(ready, features=frozenset({FEATURE})))
        win.input.set_allow_remote_input(True)
        mac.input.request_control()
        assert wait_for(lambda: win.input.state is InputSessionState.BEING_CONTROLLED)
        with pytest.raises(CoreError, match="input session"):
            win.allow_peer_updates(ready)
        mac.input.stop_control(reason="manual")
        assert wait_for(lambda: win.input.state is InputSessionState.IDLE)
        assert not (home / "publisher").exists()
        window.apply_core_snapshot(ready)
        assert window.allow_updates_button.isEnabled()
        qtbot.mouseClick(window.allow_updates_button, Qt.MouseButton.LeftButton)
        window.apply_core_snapshot(win.snapshot)
        assert not window.allow_updates_button.isEnabled()
        assert "allowed" in window.allow_updates_button.text().lower()
        restarted = ManagedReceiver(home)
        assert restarted.report()["publisher"] == pub
        assert restarted.report(mac.identity.fingerprint)["managed"]
        assert not restarted.report("another peer")["managed"]
        archive = tmp_path / "release.zip"
        with ZipFile(archive, "w") as package:
            package.comment = ("a" * 40).encode()
        release = Release.sign(archive, 1, signer)
        with pytest.raises(ValueError, match="peer"):
            restarted.apply("f" * 32, release, archive, "another peer", "c" * 32)
        wrong = Release.sign(archive, 1, Ed25519PrivateKey.generate())
        with pytest.raises(ValueError, match="signature"):
            restarted.apply("f" * 32, wrong, archive, mac.identity.fingerprint, "c" * 32)
        with pytest.raises(ValueError, match="already"):
            restarted.enroll("d" * 64, mac.identity.fingerprint)
    finally:
        win.close()
        mac.close()


def test_ambiguous_malformed_and_legacy_publisher_offers_fail_closed():
    first = enrollment.OFFER_PREFIX + "1" * 64
    for features in (set(), {FEATURE}, {first}, {FEATURE, enrollment.OFFER_PREFIX + "bad"},
                     {FEATURE, first, enrollment.OFFER_PREFIX + "2" * 64}):
        assert enrollment.offered_publisher(features) is None


def test_unenrolled_receiver_refuses_install_and_preserves_invalid_pin(tmp_path):
    receiver = ManagedReceiver(tmp_path)
    with pytest.raises(ValueError, match="peer"):
        receiver.enroll("c" * 64, None)
    assert not (tmp_path / "publisher").exists()
    draft = Release("a" * 40, "b" * 64, 1, "0" * 128)
    with pytest.raises(ValueError, match="not allowed"):
        receiver.apply("a" * 32, draft, tmp_path / "no.zip", "another peer", "b" * 32)
    pin = tmp_path / "publisher"
    pin.write_text("broken")
    with pytest.raises(ValueError, match="publisher"):
        ManagedReceiver(tmp_path)
    assert pin.read_text() == "broken"
