import json
import multiprocessing
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

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
import shooklink.protocol.crypto as crypto


def _connected_sessions(tmp_path):
    alice_identity = IdentityStore(tmp_path / "alice.key").load_or_create()
    bob_identity = IdentityStore(tmp_path / "bob.key").load_or_create()
    alice = SecureSession.initiator(alice_identity)
    bob = SecureSession.responder(bob_identity)
    bob.receive_hello(alice.create_hello())
    alice.receive_hello(bob.create_hello())
    return alice, bob


def _coordinate_atomic_write(barrier):
    real_atomic_write = crypto._atomic_write

    def coordinated_write(*args, **kwargs):
        try:
            barrier.wait(timeout=0.3)
        except threading.BrokenBarrierError:
            pass
        return real_atomic_write(*args, **kwargs)

    crypto._atomic_write = coordinated_write


def _create_identity_process(path, barrier, result_queue):
    _coordinate_atomic_write(barrier)
    identity = IdentityStore(path).load_or_create()
    result_queue.put(identity.public_bytes)


def _accept_trust_process(path, peer_id, fingerprint, barrier):
    _coordinate_atomic_write(barrier)
    TrustStore(path).accept(peer_id, fingerprint)


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

    from_alice = alice.encrypt(1, 1, b"same plaintext")
    from_bob = bob.encrypt(1, 1, b"same plaintext")

    assert from_alice != from_bob
    assert bob.decrypt(1, 1, from_alice) == b"same plaintext"
    assert alice.decrypt(1, 1, from_bob) == b"same plaintext"


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


def test_signed_low_order_ephemeral_key_is_rejected_as_handshake_error(tmp_path):
    alice = IdentityStore(tmp_path / "alice.key").load_or_create()
    bob = IdentityStore(tmp_path / "bob.key").load_or_create()
    body = (
        crypto.HELLO_MAGIC
        + alice.public_bytes
        + bytes(crypto.HELLO_EPHEMERAL_SIZE)
        + os.urandom(crypto.HELLO_NONCE_SIZE)
    )
    hello = body + alice.sign(crypto.HELLO_DOMAIN + body)

    with pytest.raises(HandshakeError, match="ephemeral"):
        SecureSession.responder(bob).receive_hello(hello)


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


def test_invalid_ciphertexts_do_not_allocate_replay_streams(tmp_path):
    _alice, bob = _connected_sessions(tmp_path)

    for stream_id in range(100):
        with pytest.raises(SecureSessionError):
            bob.decrypt(stream_id, 1, b"x" * 16)

    assert len(bob._received_windows) == 0


def test_concurrent_encrypt_rejects_duplicate_nonce(tmp_path, monkeypatch):
    alice, _bob = _connected_sessions(tmp_path)
    real_cipher = alice._send_cipher
    first_entered = threading.Event()
    second_entered = threading.Event()
    calls_lock = threading.Lock()
    calls = 0

    class CoordinatedCipher:
        def encrypt(self, nonce, plaintext, aad):
            nonlocal calls
            with calls_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                first_entered.set()
                second_entered.wait(0.2)
            else:
                second_entered.set()
            return real_cipher.encrypt(nonce, plaintext, aad)

    monkeypatch.setattr(alice, "_send_cipher", CoordinatedCipher())

    def encrypt_once():
        try:
            return alice.encrypt(11, 5, b"same")
        except ReplayError:
            return "replay"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: encrypt_once(), range(2)))

    assert sum(result == "replay" for result in results) == 1
    assert sum(isinstance(result, bytes) for result in results) == 1


def test_concurrent_decrypt_rejects_duplicate_ciphertext(tmp_path, monkeypatch):
    alice, bob = _connected_sessions(tmp_path)
    sealed = alice.encrypt(12, 6, b"same")
    real_cipher = bob._receive_cipher
    first_entered = threading.Event()
    second_entered = threading.Event()
    calls_lock = threading.Lock()
    calls = 0

    class CoordinatedCipher:
        def decrypt(self, nonce, ciphertext, aad):
            nonlocal calls
            with calls_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                first_entered.set()
                second_entered.wait(0.2)
            else:
                second_entered.set()
            return real_cipher.decrypt(nonce, ciphertext, aad)

    monkeypatch.setattr(bob, "_receive_cipher", CoordinatedCipher())

    def decrypt_once():
        try:
            return bob.decrypt(12, 6, sealed)
        except ReplayError:
            return "replay"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: decrypt_once(), range(2)))

    assert sorted(results, key=str) == [b"same", "replay"]


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


def test_concurrent_identity_creation_returns_persisted_winner(tmp_path, monkeypatch):
    path = tmp_path / "identity.key"
    real_atomic_write = crypto._atomic_write
    first_entered = threading.Event()
    second_entered = threading.Event()
    call_lock = threading.Lock()
    calls = 0

    def coordinated_write(*args, **kwargs):
        nonlocal calls
        with call_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_entered.set()
            second_entered.wait(0.2)
        else:
            second_entered.set()
        return real_atomic_write(*args, **kwargs)

    monkeypatch.setattr(crypto, "_atomic_write", coordinated_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        identities = list(
            executor.map(lambda _index: IdentityStore(path).load_or_create(), range(2))
        )

    persisted = IdentityStore(path).load_or_create()
    assert {identity.public_bytes for identity in identities} == {persisted.public_bytes}


def test_cross_process_identity_creation_returns_persisted_winner(tmp_path):
    context = multiprocessing.get_context("spawn")
    path = tmp_path / "identity-process.key"
    process_count = 3
    barrier = context.Barrier(process_count)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_create_identity_process,
            args=(path, barrier, result_queue),
        )
        for _index in range(process_count)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    results = [result_queue.get(timeout=2) for _index in range(process_count)]
    result_queue.close()
    result_queue.join_thread()

    persisted = IdentityStore(path).load_or_create()
    assert set(results) == {persisted.public_bytes}


def test_concurrent_trust_updates_preserve_both_peers(tmp_path, monkeypatch):
    path = tmp_path / "trusted.json"
    real_atomic_write = crypto._atomic_write
    first_entered = threading.Event()
    second_entered = threading.Event()
    call_lock = threading.Lock()
    calls = 0

    def coordinated_write(*args, **kwargs):
        nonlocal calls
        with call_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            first_entered.set()
            second_entered.wait(0.2)
        else:
            second_entered.set()
        return real_atomic_write(*args, **kwargs)

    monkeypatch.setattr(crypto, "_atomic_write", coordinated_write)
    trust = TrustStore(path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda pair: trust.accept(*pair),
                [("alice", "fingerprint-a"), ("bob", "fingerprint-b")],
            )
        )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "alice": "fingerprint-a",
        "bob": "fingerprint-b",
    }


def test_cross_process_trust_updates_preserve_every_peer(tmp_path):
    context = multiprocessing.get_context("spawn")
    path = tmp_path / "trusted-process.json"
    peers = [(f"peer-{index}", f"fingerprint-{index}") for index in range(3)]
    barrier = context.Barrier(len(peers))
    processes = [
        context.Process(
            target=_accept_trust_process,
            args=(path, peer_id, fingerprint, barrier),
        )
        for peer_id, fingerprint in peers
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert json.loads(path.read_text(encoding="utf-8")) == dict(peers)
