import json
import os
import stat

import pytest

from shooklink.protocol.crypto import (
    HandshakeError,
    IdentityStore,
    ReplayError,
    SecureSession,
    SecureSessionError,
    TrustStatus,
    TrustStore,
)


def _connected_sessions(tmp_path):
    alice_identity = IdentityStore(tmp_path / "alice.key").load_or_create()
    bob_identity = IdentityStore(tmp_path / "bob.key").load_or_create()
    alice = SecureSession.initiator(alice_identity)
    bob = SecureSession.responder(bob_identity)
    bob.receive_hello(alice.create_hello())
    alice.receive_hello(bob.create_hello())
    return alice, bob


def test_identity_is_persistent_and_private(tmp_path):
    path = tmp_path / "identity.key"
    store = IdentityStore(path)

    first = store.load_or_create()
    second = store.load_or_create()

    assert first.public_bytes == second.public_bytes
    assert first.fingerprint == second.fingerprint
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_corrupt_identity_file_is_rejected(tmp_path):
    path = tmp_path / "identity.key"
    path.write_bytes(b"short")

    with pytest.raises(SecureSessionError, match="identity"):
        IdentityStore(path).load_or_create()


def test_two_signed_handshakes_derive_matching_directional_keys(tmp_path):
    alice, bob = _connected_sessions(tmp_path)

    assert alice.is_ready and bob.is_ready
    assert alice.remote_fingerprint == bob.local_fingerprint
    assert bob.remote_fingerprint == alice.local_fingerprint
    sealed = alice.encrypt(4, 7, b"secret", associated_data=b"chat")

    assert sealed != b"secret"
    assert bob.decrypt(4, 7, sealed, associated_data=b"chat") == b"secret"


def test_bidirectional_encryption_uses_independent_keys(tmp_path):
    alice, bob = _connected_sessions(tmp_path)

    from_alice = alice.encrypt(1, 1, b"alice")
    from_bob = bob.encrypt(1, 1, b"bob")

    assert from_alice != from_bob
    assert bob.decrypt(1, 1, from_alice) == b"alice"
    assert alice.decrypt(1, 1, from_bob) == b"bob"


def test_tampered_handshake_signature_is_rejected(tmp_path):
    alice_identity = IdentityStore(tmp_path / "alice.key").load_or_create()
    bob_identity = IdentityStore(tmp_path / "bob.key").load_or_create()
    hello = bytearray(SecureSession.initiator(alice_identity).create_hello())
    hello[-1] ^= 1

    with pytest.raises(HandshakeError, match="signature"):
        SecureSession.responder(bob_identity).receive_hello(bytes(hello))


@pytest.mark.parametrize("hello", [b"", b"wrong", b"SLH1" + b"x" * 10])
def test_malformed_handshake_is_rejected(tmp_path, hello):
    identity = IdentityStore(tmp_path / "identity.key").load_or_create()

    with pytest.raises(HandshakeError):
        SecureSession.responder(identity).receive_hello(hello)


def test_ciphertext_and_associated_data_tampering_is_rejected(tmp_path):
    alice, bob = _connected_sessions(tmp_path)
    sealed = bytearray(alice.encrypt(3, 9, b"secret", associated_data=b"shell"))
    sealed[-1] ^= 1

    with pytest.raises(SecureSessionError, match="authentication"):
        bob.decrypt(3, 9, bytes(sealed), associated_data=b"shell")


def test_nonce_reuse_and_replay_are_rejected(tmp_path):
    alice, bob = _connected_sessions(tmp_path)
    sealed = alice.encrypt(2, 10, b"first")

    with pytest.raises(ReplayError):
        alice.encrypt(2, 10, b"second")

    assert bob.decrypt(2, 10, sealed) == b"first"
    with pytest.raises(ReplayError):
        bob.decrypt(2, 10, sealed)


def test_unseen_out_of_order_ciphertext_is_accepted_within_replay_window(tmp_path):
    alice, bob = _connected_sessions(tmp_path)
    earlier = alice.encrypt(8, 5, b"earlier")
    later = alice.encrypt(8, 7, b"later")

    assert bob.decrypt(8, 7, later) == b"later"
    assert bob.decrypt(8, 5, earlier) == b"earlier"


def test_ciphertext_older_than_replay_window_is_rejected(tmp_path):
    alice, bob = _connected_sessions(tmp_path)
    old = alice.encrypt(9, 1, b"old")
    newest = alice.encrypt(9, 300, b"new")

    assert bob.decrypt(9, 300, newest) == b"new"
    with pytest.raises(ReplayError, match="stale"):
        bob.decrypt(9, 1, old)


def test_failed_authentication_does_not_consume_sequence(tmp_path):
    alice, bob = _connected_sessions(tmp_path)
    sealed = alice.encrypt(10, 4, b"payload", associated_data=b"right")

    with pytest.raises(SecureSessionError, match="authentication"):
        bob.decrypt(10, 4, sealed, associated_data=b"wrong")

    assert bob.decrypt(10, 4, sealed, associated_data=b"right") == b"payload"


def test_encryption_requires_completed_handshake(tmp_path):
    identity = IdentityStore(tmp_path / "identity.key").load_or_create()
    session = SecureSession.initiator(identity)

    with pytest.raises(SecureSessionError, match="not ready"):
        session.encrypt(1, 1, b"payload")


def test_sequence_and_stream_ranges_are_validated(tmp_path):
    alice, _bob = _connected_sessions(tmp_path)

    with pytest.raises(ValueError):
        alice.encrypt(-1, 1, b"bad")
    with pytest.raises(ValueError):
        alice.encrypt(1, 1 << 32, b"bad")


def test_trust_store_reports_unknown_trusted_and_changed(tmp_path):
    trust = TrustStore(tmp_path / "trusted.json")

    assert trust.status("peer", "fingerprint-a") is TrustStatus.UNKNOWN
    trust.accept("peer", "fingerprint-a")
    assert trust.status("peer", "fingerprint-a") is TrustStatus.TRUSTED
    assert trust.check("peer", "fingerprint-a") is True
    assert trust.status("peer", "fingerprint-b") is TrustStatus.CHANGED
    assert trust.check("peer", "fingerprint-b") is False


def test_trust_store_is_atomic_and_tolerates_corrupt_data(tmp_path):
    path = tmp_path / "trusted.json"
    path.write_text("not json", encoding="utf-8")
    trust = TrustStore(path)

    assert trust.status("peer", "fingerprint") is TrustStatus.UNKNOWN
    trust.accept("peer", "fingerprint")

    assert json.loads(path.read_text(encoding="utf-8")) == {"peer": "fingerprint"}
    assert not list(tmp_path.glob("*.tmp"))
