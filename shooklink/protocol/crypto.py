"""Peer identity, trust-on-first-use, and authenticated session encryption."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import tempfile
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

HELLO_MAGIC = b"SLH1"
HELLO_DOMAIN = b"ShookLink signed hello v1\x00"
HELLO_IDENTITY_SIZE = 32
HELLO_EPHEMERAL_SIZE = 32
HELLO_NONCE_SIZE = 16
HELLO_SIGNATURE_SIZE = 64
HELLO_BODY_SIZE = (
    len(HELLO_MAGIC)
    + HELLO_IDENTITY_SIZE
    + HELLO_EPHEMERAL_SIZE
    + HELLO_NONCE_SIZE
)
HELLO_SIZE = HELLO_BODY_SIZE + HELLO_SIGNATURE_SIZE
KDF_INFO = b"ShookLink secure session v1"
AEAD_DOMAIN = b"ShookLink AEAD v1\x00"
LOW_TO_HIGH_NONCE = b"L2H1"
HIGH_TO_LOW_NONCE = b"H2L1"
UINT32_MAX = (1 << 32) - 1
REPLAY_WINDOW_SIZE = 256
REPLAY_WINDOW_MASK = (1 << REPLAY_WINDOW_SIZE) - 1
MAX_TRACKED_STREAMS = 1_024
MAX_TRUST_STORE_SIZE = 1 << 20
_STORE_LOCK = threading.RLock()


class SecureSessionError(RuntimeError):
    """Base class for identity and secure-session failures."""


class HandshakeError(SecureSessionError):
    """Raised when a signed peer hello is malformed or invalid."""


class ReplayError(SecureSessionError):
    """Raised when a stream sequence would reuse an AEAD nonce."""


class TrustStatus(Enum):
    UNKNOWN = "unknown"
    TRUSTED = "trusted"
    CHANGED = "changed"


@dataclass(slots=True)
class _ReplayWindow:
    highest: int = -1
    bitmap: int = 0

    def check(self, sequence: int) -> None:
        if self.highest < 0 or sequence > self.highest:
            return
        distance = self.highest - sequence
        if distance >= REPLAY_WINDOW_SIZE:
            raise ReplayError("ciphertext sequence is stale")
        if self.bitmap & (1 << distance):
            raise ReplayError("ciphertext sequence was already received")

    def mark(self, sequence: int) -> None:
        if self.highest < 0:
            self.highest = sequence
            self.bitmap = 1
            return
        if sequence > self.highest:
            distance = sequence - self.highest
            if distance >= REPLAY_WINDOW_SIZE:
                self.bitmap = 1
            else:
                self.bitmap = ((self.bitmap << distance) | 1) & REPLAY_WINDOW_MASK
            self.highest = sequence
            return
        self.bitmap |= 1 << (self.highest - sequence)


def _public_bytes(public_key: Ed25519PublicKey | X25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _fingerprint(public_bytes: bytes) -> str:
    digest = hashlib.sha256(public_bytes).digest()
    encoded = base64.b64encode(digest).decode("ascii").rstrip("=")
    return f"SHA256:{encoded}"


def _atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        if os.name == "posix":
            os.fchmod(descriptor, mode)
        temporary_file = os.fdopen(descriptor, "wb")
        descriptor = None
        with temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class Identity:
    _private_key: Ed25519PrivateKey
    public_bytes: bytes
    fingerprint: str

    @classmethod
    def from_private_key(cls, private_key: Ed25519PrivateKey) -> Identity:
        public = _public_bytes(private_key.public_key())
        return cls(private_key, public, _fingerprint(public))

    def sign(self, data: bytes) -> bytes:
        return self._private_key.sign(data)


class IdentityStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_or_create(self) -> Identity:
        with _STORE_LOCK:
            if self.path.exists():
                try:
                    with self.path.open("rb") as identity_file:
                        private_bytes = identity_file.read(33)
                    if len(private_bytes) != 32:
                        raise ValueError("wrong key size")
                    private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
                except (OSError, ValueError) as error:
                    raise SecureSessionError("invalid identity file") from error
                if os.name == "posix":
                    self.path.chmod(0o600)
                return Identity.from_private_key(private_key)

            private_key = Ed25519PrivateKey.generate()
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            try:
                _atomic_write(self.path, private_bytes)
            except OSError as error:
                raise SecureSessionError("could not store identity") from error
            return Identity.from_private_key(private_key)


class TrustStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def _validate(peer_id: str, fingerprint: str) -> None:
        if not isinstance(peer_id, str) or not peer_id.strip():
            raise ValueError("peer_id must be a non-empty string")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("fingerprint must be a non-empty string")

    def _load(self) -> dict[str, str]:
        try:
            if self.path.stat().st_size > MAX_TRUST_STORE_SIZE:
                return {}
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            key: value
            for key, value in raw.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    def status(self, peer_id: str, fingerprint: str) -> TrustStatus:
        self._validate(peer_id, fingerprint)
        with _STORE_LOCK:
            stored = self._load().get(peer_id)
            if stored is None:
                return TrustStatus.UNKNOWN
            if stored == fingerprint:
                return TrustStatus.TRUSTED
            return TrustStatus.CHANGED

    def check(self, peer_id: str, fingerprint: str) -> bool:
        return self.status(peer_id, fingerprint) is TrustStatus.TRUSTED

    def accept(self, peer_id: str, fingerprint: str) -> None:
        self._validate(peer_id, fingerprint)
        with _STORE_LOCK:
            trusted = self._load()
            trusted[peer_id] = fingerprint
            encoded = json.dumps(
                trusted,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            _atomic_write(self.path, encoded)


class SecureSession:
    def __init__(self, identity: Identity) -> None:
        if not isinstance(identity, Identity):
            raise TypeError("identity must be an Identity")
        self._crypto_lock = threading.RLock()
        self._identity = identity
        self._ephemeral_private = X25519PrivateKey.generate()
        self._ephemeral_public = _public_bytes(self._ephemeral_private.public_key())
        self._hello_nonce = os.urandom(HELLO_NONCE_SIZE)
        hello_body = (
            HELLO_MAGIC
            + self._identity.public_bytes
            + self._ephemeral_public
            + self._hello_nonce
        )
        self._hello = hello_body + self._identity.sign(HELLO_DOMAIN + hello_body)
        self._remote_hello: bytes | None = None
        self._remote_fingerprint: str | None = None
        self._send_cipher: ChaCha20Poly1305 | None = None
        self._receive_cipher: ChaCha20Poly1305 | None = None
        self._send_nonce_prefix: bytes | None = None
        self._receive_nonce_prefix: bytes | None = None
        self._session_context: bytes | None = None
        self._sent_sequences: dict[int, int] = {}
        self._received_windows: dict[int, _ReplayWindow] = {}

    @classmethod
    def initiator(cls, identity: Identity) -> SecureSession:
        return cls(identity)

    @classmethod
    def responder(cls, identity: Identity) -> SecureSession:
        return cls(identity)

    @property
    def is_ready(self) -> bool:
        return self._send_cipher is not None and self._receive_cipher is not None

    @property
    def local_fingerprint(self) -> str:
        return self._identity.fingerprint

    @property
    def remote_fingerprint(self) -> str | None:
        return self._remote_fingerprint

    def create_hello(self) -> bytes:
        return self._hello

    def receive_hello(self, hello: bytes) -> None:
        with self._crypto_lock:
            self._receive_hello(hello)

    def _receive_hello(self, hello: bytes) -> None:
        if not isinstance(hello, bytes) or len(hello) != HELLO_SIZE:
            raise HandshakeError("invalid handshake length")
        if hello[: len(HELLO_MAGIC)] != HELLO_MAGIC:
            raise HandshakeError("invalid handshake magic")
        if self._remote_hello is not None:
            if hello == self._remote_hello:
                return
            raise HandshakeError("peer changed its handshake")

        identity_start = len(HELLO_MAGIC)
        ephemeral_start = identity_start + HELLO_IDENTITY_SIZE
        nonce_start = ephemeral_start + HELLO_EPHEMERAL_SIZE
        signature_start = nonce_start + HELLO_NONCE_SIZE
        remote_identity_bytes = hello[identity_start:ephemeral_start]
        remote_ephemeral_bytes = hello[ephemeral_start:nonce_start]
        body = hello[:signature_start]
        signature = hello[signature_start:]
        if remote_identity_bytes == self._identity.public_bytes:
            raise HandshakeError("peer identity matches local identity")

        try:
            remote_identity = Ed25519PublicKey.from_public_bytes(remote_identity_bytes)
            remote_identity.verify(signature, HELLO_DOMAIN + body)
            remote_ephemeral = X25519PublicKey.from_public_bytes(remote_ephemeral_bytes)
        except InvalidSignature as error:
            raise HandshakeError("invalid handshake signature") from error
        except ValueError as error:
            raise HandshakeError("invalid handshake public key") from error

        try:
            shared_secret = self._ephemeral_private.exchange(remote_ephemeral)
        except ValueError as error:
            raise HandshakeError("invalid handshake ephemeral key") from error
        ordered_hellos = sorted(
            (self._hello, hello),
            key=lambda item: item[identity_start:ephemeral_start],
        )
        transcript = hashlib.sha256(b"".join(ordered_hellos)).digest()
        key_material = HKDF(
            algorithm=hashes.SHA256(),
            length=64,
            salt=transcript,
            info=KDF_INFO,
        ).derive(shared_secret)

        local_is_lower = self._identity.public_bytes < remote_identity_bytes
        low_to_high_key, high_to_low_key = key_material[:32], key_material[32:]
        if local_is_lower:
            send_key, receive_key = low_to_high_key, high_to_low_key
            send_prefix, receive_prefix = LOW_TO_HIGH_NONCE, HIGH_TO_LOW_NONCE
        else:
            send_key, receive_key = high_to_low_key, low_to_high_key
            send_prefix, receive_prefix = HIGH_TO_LOW_NONCE, LOW_TO_HIGH_NONCE

        self._remote_hello = hello
        self._remote_fingerprint = _fingerprint(remote_identity_bytes)
        self._send_cipher = ChaCha20Poly1305(send_key)
        self._receive_cipher = ChaCha20Poly1305(receive_key)
        self._send_nonce_prefix = send_prefix
        self._receive_nonce_prefix = receive_prefix
        self._session_context = transcript

    @staticmethod
    def _validate_position(stream_id: int, sequence: int) -> None:
        if type(stream_id) is not int or not 0 <= stream_id <= UINT32_MAX:
            raise ValueError("stream_id must be an unsigned 32-bit integer")
        if type(sequence) is not int or not 0 <= sequence <= UINT32_MAX:
            raise ValueError("sequence must be an unsigned 32-bit integer")

    def _aead_parameters(
        self,
        prefix: bytes | None,
        stream_id: int,
        sequence: int,
        associated_data: bytes,
    ) -> tuple[bytes, bytes]:
        self._validate_position(stream_id, sequence)
        if not isinstance(associated_data, bytes):
            raise TypeError("associated_data must be bytes")
        if prefix is None or self._session_context is None:
            raise SecureSessionError("secure session is not ready")
        position = struct.pack(">II", stream_id, sequence)
        nonce = prefix + position
        aad = AEAD_DOMAIN + self._session_context + position + associated_data
        return nonce, aad

    def encrypt(
        self,
        stream_id: int,
        sequence: int,
        plaintext: bytes,
        *,
        associated_data: bytes = b"",
    ) -> bytes:
        with self._crypto_lock:
            if not isinstance(plaintext, bytes):
                raise TypeError("plaintext must be bytes")
            nonce, aad = self._aead_parameters(
                self._send_nonce_prefix,
                stream_id,
                sequence,
                associated_data,
            )
            last_sequence = self._sent_sequences.get(stream_id, -1)
            if sequence <= last_sequence:
                raise ReplayError("send sequence would reuse a nonce")
            if (
                stream_id not in self._sent_sequences
                and len(self._sent_sequences) >= MAX_TRACKED_STREAMS
            ):
                raise SecureSessionError("too many active encrypted streams")
            if self._send_cipher is None:
                raise SecureSessionError("secure session is not ready")
            ciphertext = self._send_cipher.encrypt(nonce, plaintext, aad)
            self._sent_sequences[stream_id] = sequence
            return ciphertext

    def decrypt(
        self,
        stream_id: int,
        sequence: int,
        ciphertext: bytes,
        *,
        associated_data: bytes = b"",
    ) -> bytes:
        with self._crypto_lock:
            if not isinstance(ciphertext, bytes):
                raise TypeError("ciphertext must be bytes")
            nonce, aad = self._aead_parameters(
                self._receive_nonce_prefix,
                stream_id,
                sequence,
                associated_data,
            )
            replay_window = self._received_windows.get(stream_id)
            if replay_window is not None:
                replay_window.check(sequence)
            if self._receive_cipher is None:
                raise SecureSessionError("secure session is not ready")
            try:
                plaintext = self._receive_cipher.decrypt(nonce, ciphertext, aad)
            except InvalidTag as error:
                raise SecureSessionError("ciphertext authentication failed") from error
            if replay_window is None:
                if len(self._received_windows) >= MAX_TRACKED_STREAMS:
                    raise SecureSessionError("too many active encrypted streams")
                replay_window = _ReplayWindow()
                self._received_windows[stream_id] = replay_window
            replay_window.mark(sequence)
            return plaintext


__all__ = [
    "HandshakeError",
    "Identity",
    "IdentityStore",
    "ReplayError",
    "SecureSession",
    "SecureSessionError",
    "TrustStatus",
    "TrustStore",
]
