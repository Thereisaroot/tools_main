"""Connection, trust, encryption, and feature routing for one serial peer."""

from __future__ import annotations

import logging
import sys
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from shooklink import __version__
from shooklink.chat.service import ChatService
from shooklink.files.service import FileService
from shooklink.input.backend import BaseInputBackend
from shooklink.input.service import InputService
from shooklink.input.topology import Side
from shooklink.protocol.crypto import (
    Identity,
    SecureSession,
    TrustStatus,
    TrustStore,
)
from shooklink.protocol.framing import (
    MAX_PAYLOAD_SIZE,
    SECURE_FLAG,
    Frame,
    pack_frame_header,
)
from shooklink.protocol.messages import (
    Message,
    MessageType,
    decode_message,
    encode_message,
)
from shooklink.shell.service import ShellService
from shooklink.transport.multiplexer import OutboundItem, Priority
from shooklink.transport.serial_link import (
    LinkClosedError,
    SerialEndpoint,
    SerialLink,
)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
AEAD_TAG_SIZE = 16
LOCAL_FEATURES = frozenset({"chat", "files", "shell", "input"})
MAX_FEATURES = 32
MAX_PEER_ID_BYTES = 256

_authenticated_message: ContextVar[Message | None] = ContextVar(
    "shooklink_authenticated_message",
    default=None,
)


class CoreError(RuntimeError):
    """Raised when a connection or trust operation cannot proceed."""


class CoreState(str, Enum):
    DISCONNECTED = "disconnected"
    DISCONNECTING = "disconnecting"
    HANDSHAKING = "handshaking"
    UNTRUSTED = "untrusted"
    CHANGED = "changed"
    READY = "ready"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CoreSnapshot:
    connection_id: int
    state: CoreState
    peer_id: str | None = None
    fingerprint: str | None = None
    trust_status: TrustStatus = TrustStatus.UNKNOWN
    local_approved: bool = False
    remote_approved: bool = False
    features: frozenset[str] = frozenset()
    error: str | None = None


@dataclass(slots=True)
class _Connection:
    connection_id: int
    link: SerialLink
    secure_session: SecureSession
    remote_peer_id: str | None = None
    remote_fingerprint: str | None = None
    trust_status: TrustStatus = TrustStatus.UNKNOWN
    remote_approved: bool = False
    remote_features: frozenset[str] = frozenset()
    last_accepts_fingerprint: str | None = None
    trust_sent: bool = False
    input_bound: bool = False
    local_hello_sent: bool = False
    remote_hello_received: bool = False
    plain_receive_sequences: dict[int, int] = field(default_factory=dict)


_FEATURE_BY_TYPE = {
    MessageType.CHAT_PLAIN: "chat",
    MessageType.CHAT_SECURE: "chat",
    MessageType.FILE_OFFER: "files",
    MessageType.FILE_ACCEPT: "files",
    MessageType.FILE_CHUNK: "files",
    MessageType.FILE_ACK: "files",
    MessageType.FILE_FINISH: "files",
    MessageType.FILE_CANCEL: "files",
    MessageType.SHELL_OPEN: "shell",
    MessageType.SHELL_ACCEPT: "shell",
    MessageType.SHELL_DENY: "shell",
    MessageType.SHELL_INPUT: "shell",
    MessageType.SHELL_OUTPUT: "shell",
    MessageType.SHELL_RESIZE: "shell",
    MessageType.SHELL_EXIT: "shell",
    MessageType.INPUT_REQUEST: "input",
    MessageType.INPUT_ACCEPT: "input",
    MessageType.INPUT_BUSY: "input",
    MessageType.INPUT_ENTER: "input",
    MessageType.INPUT_LEAVE: "input",
    MessageType.INPUT_KEY: "input",
    MessageType.INPUT_BUTTON: "input",
    MessageType.INPUT_MOVE: "input",
    MessageType.INPUT_WHEEL: "input",
    MessageType.INPUT_POINTER_STATE: "input",
    MessageType.INPUT_STOP: "input",
    MessageType.INPUT_RELEASE_ALL: "input",
}

_PLAIN_TYPES = frozenset({MessageType.HELLO, MessageType.CHAT_PLAIN})

_ALLOWED_PRIORITIES = {
    MessageType.HELLO: frozenset({Priority.INTERACTIVE}),
    MessageType.TRUST: frozenset({Priority.INTERACTIVE}),
    MessageType.CHAT_PLAIN: frozenset({Priority.NORMAL}),
    MessageType.CHAT_SECURE: frozenset({Priority.NORMAL}),
    MessageType.FILE_OFFER: frozenset({Priority.NORMAL}),
    MessageType.FILE_ACCEPT: frozenset({Priority.NORMAL}),
    MessageType.FILE_CHUNK: frozenset({Priority.FILE}),
    MessageType.FILE_ACK: frozenset({Priority.NORMAL}),
    MessageType.FILE_FINISH: frozenset({Priority.NORMAL}),
    MessageType.FILE_CANCEL: frozenset(
        {Priority.INTERACTIVE, Priority.NORMAL}
    ),
    MessageType.SHELL_OPEN: frozenset({Priority.INTERACTIVE}),
    MessageType.SHELL_ACCEPT: frozenset({Priority.INTERACTIVE}),
    MessageType.SHELL_DENY: frozenset({Priority.INTERACTIVE}),
    MessageType.SHELL_INPUT: frozenset({Priority.INTERACTIVE}),
    MessageType.SHELL_OUTPUT: frozenset({Priority.NORMAL}),
    MessageType.SHELL_RESIZE: frozenset({Priority.INTERACTIVE}),
    MessageType.SHELL_EXIT: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_REQUEST: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_ACCEPT: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_BUSY: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_ENTER: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_LEAVE: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_KEY: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_BUTTON: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_MOVE: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_WHEEL: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_POINTER_STATE: frozenset({Priority.MOTION}),
    MessageType.INPUT_STOP: frozenset({Priority.INTERACTIVE}),
    MessageType.INPUT_RELEASE_ALL: frozenset({Priority.INTERACTIVE}),
}


class ShookLinkCore:
    def __init__(
        self,
        *,
        identity: Identity,
        trust_store: TrustStore,
        download_dir: str | Path,
        local_peer_id: str,
        input_backend: BaseInputBackend | None = None,
        process_factory=None,
        peer_side: Side = Side.RIGHT,
        auto_edge_enabled: bool = False,
        debug: bool = False,
    ) -> None:
        _validate_peer_id(local_peer_id)
        if not isinstance(identity, Identity):
            raise TypeError("identity must be an Identity")
        if not isinstance(trust_store, TrustStore):
            raise TypeError("trust_store must be a TrustStore")
        self.identity = identity
        self.trust_store = trust_store
        self.local_peer_id = local_peer_id
        self.debug = bool(debug)
        self._lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._connection: _Connection | None = None
        self._next_connection_id = 1
        self._last_connection_id = 0
        self._last_state = CoreState.DISCONNECTED
        self._last_error: str | None = None
        self._closed = False
        self._disconnect_requested = threading.Event()
        self._listeners = []
        self._links: list[SerialLink] = []

        self.chat = ChatService(self)
        self.files = FileService(self, download_dir)
        self.files.disconnect()
        self.shell = (
            ShellService(self)
            if process_factory is None
            else ShellService(self, process_factory)
        )
        if input_backend is None:
            self.input = None
        else:
            pending_peer = (
                "pending-remote-peer"
                if local_peer_id != "pending-remote-peer"
                else "pending-remote-peer-2"
            )
            self.input = InputService(
                self,
                input_backend,
                local_peer_id=local_peer_id,
                peer_id=pending_peer,
                peer_side=peer_side,
                auto_edge_enabled=auto_edge_enabled,
                connected=False,
            )

    @property
    def trusted(self) -> bool:
        with self._lock:
            return self._trusted_locked()

    @property
    def snapshot(self) -> CoreSnapshot:
        with self._lock:
            return self._snapshot_locked()

    @property
    def threads_alive(self) -> bool:
        with self._lock:
            links = tuple(self._links)
        return any(link.threads_alive for link in links)

    def add_state_listener(self, listener) -> None:
        if not callable(listener):
            raise TypeError("state listener must be callable")
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_state_listener(self, listener) -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    def connect_endpoint(self, endpoint: SerialEndpoint) -> None:
        if endpoint is None:
            raise TypeError("endpoint is required")
        self._connect(
            lambda connection_id: SerialLink(
                endpoint,
                lambda frame: self._on_frame(connection_id, frame),
                lambda error: self._on_disconnect(connection_id, error),
            )
        )

    def connect_port(self, port: str, baud_rate: int) -> None:
        if not isinstance(port, str) or not port.strip():
            raise ValueError("serial port must be a non-empty string")
        if type(baud_rate) is not int or baud_rate <= 0:
            raise ValueError("baud rate must be positive")
        self._connect(
            lambda connection_id: SerialLink.open_port(
                port,
                baud_rate,
                lambda frame: self._on_frame(connection_id, frame),
                lambda error: self._on_disconnect(connection_id, error),
            )
        )

    def _connect(self, link_factory) -> None:
        with self._lock:
            if self._closed:
                raise CoreError("core is closed")
            if self._connection is not None:
                raise CoreError("a serial peer is already connected")
            connection_id = self._next_connection_id
            self._next_connection_id += 1
            link = link_factory(connection_id)
            connection = _Connection(
                connection_id,
                link,
                SecureSession(self.identity),
            )
            self._connection = connection
            self._last_connection_id = connection_id
            self._last_state = CoreState.HANDSHAKING
            self._last_error = None
            self._links = [item for item in self._links if item.threads_alive]
            self._links.append(link)
        self.files.connection_changed(True)
        self._publish_current()
        try:
            link.start()
            self._send_internal(
                connection_id,
                Message(
                    MessageType.HELLO,
                    {"protocol": PROTOCOL_VERSION},
                    connection.secure_session.create_hello(),
                ),
                secure=False,
                priority=Priority.INTERACTIVE,
                allow_untrusted=True,
            )
            with self._lock:
                current = self._connection
                send_trust = bool(
                    current is not None
                    and current.connection_id == connection_id
                    and current.remote_hello_received
                )
                if current is not None and current.connection_id == connection_id:
                    current.local_hello_sent = True
            if send_trust:
                self._send_trust(connection_id)
        except BaseException:
            try:
                link.close()
            except BaseException:
                pass
            raise

    def approve_peer(self, connection_id: int, fingerprint: str) -> None:
        with self._lock:
            connection = self._connection
            if (
                connection is None
                or connection.connection_id != connection_id
                or connection.remote_peer_id is None
                or connection.remote_fingerprint != fingerprint
            ):
                raise CoreError("peer approval is stale")
            if connection.trust_status is TrustStatus.CHANGED:
                raise CoreError("peer identity changed; remove the old trust entry first")
            peer_id = connection.remote_peer_id
            # Keep approval and disconnect ordered so persisted trust cannot
            # disagree with the result reported to the UI.
            self.trust_store.accept(peer_id, fingerprint)
            connection.trust_status = TrustStatus.TRUSTED
            link_closed = connection.link.closed
        if not link_closed:
            try:
                self._send_trust(connection_id)
            except LinkClosedError:
                pass
        self._refresh_input_if_ready(connection_id)
        self._publish_current()

    def send(
        self,
        message: Message,
        *,
        secure: bool = False,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        if message.message_type in {MessageType.HELLO, MessageType.TRUST}:
            raise CoreError("handshake messages are managed by the core")
        self._send_user(message, secure=secure, priority=priority)

    def decrypt_secure(self, message: Message) -> bytes:
        if _authenticated_message.get() is not message:
            raise CoreError("message was not authenticated by the active dispatch")
        return message.body

    def disconnect(self) -> None:
        # Record intent before waiting for the core lock. A transport worker may
        # already be reporting the peer's concurrent close through this lock.
        self._disconnect_requested.set()
        with self._lock:
            connection = self._connection
            if connection is None:
                self._last_state = CoreState.DISCONNECTED
                self._last_error = None
                self._disconnect_requested.clear()
        if connection is None:
            self._publish_current()
            return
        self._publish_current()
        connection.link.close()

    def close(self) -> None:
        self._disconnect_requested.set()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            connection = self._connection
        if connection is not None:
            try:
                connection.link.close()
            except BaseException:
                pass
        self.files.close()
        self.shell.close()
        if self.input is not None:
            self.input.close()

    def _send_user(
        self,
        message: Message,
        *,
        secure: bool,
        priority: Priority,
    ) -> None:
        if not isinstance(message, Message):
            raise TypeError("message must be a Message")
        if not isinstance(priority, Priority):
            raise TypeError("priority must be a Priority")
        expected_secure = message.message_type not in _PLAIN_TYPES
        if secure != expected_secure:
            raise CoreError("message security does not match its protocol type")
        with self._lock:
            connection = self._connection
            if connection is None:
                raise CoreError("connect a serial peer first")
            if priority not in _ALLOWED_PRIORITIES[message.message_type]:
                raise CoreError("message priority is invalid for its protocol type")
            feature = _FEATURE_BY_TYPE[message.message_type]
            if feature not in connection.remote_features:
                raise CoreError(f"peer does not support {feature}")
            if secure and not self._trusted_locked():
                raise CoreError("both peers must approve trust before protected traffic")
            connection_id = connection.connection_id
        self._send_internal(
            connection_id,
            message,
            secure=secure,
            priority=priority,
            allow_untrusted=False,
        )

    def _send_internal(
        self,
        connection_id: int,
        message: Message,
        *,
        secure: bool,
        priority: Priority,
        allow_untrusted: bool,
    ) -> None:
        with self._send_lock:
            self._send_internal_serialized(
                connection_id,
                message,
                secure=secure,
                priority=priority,
                allow_untrusted=allow_untrusted,
            )

    def _send_internal_serialized(
        self,
        connection_id: int,
        message: Message,
        *,
        secure: bool,
        priority: Priority,
        allow_untrusted: bool,
    ) -> None:
        with self._lock:
            connection = self._connection
            if connection is None or connection.connection_id != connection_id:
                raise CoreError("serial connection changed")
            if secure and not allow_untrusted and not self._trusted_locked():
                raise CoreError("peer is not trusted")
            link = connection.link
            secure_session = connection.secure_session
        encoded = encode_message(message)
        stream_id = _stream_id(message.message_type, priority)
        flags = SECURE_FLAG if secure else 0
        sequence = None
        payload = encoded
        if secure:
            if not secure_session.is_ready:
                raise CoreError("secure handshake is not ready")
            sequence = link.reserve_sequence(stream_id)
            ciphertext_length = len(encoded) + AEAD_TAG_SIZE
            if ciphertext_length > MAX_PAYLOAD_SIZE:
                try:
                    link.cancel_sequence(stream_id, sequence)
                except Exception:
                    pass
                raise CoreError("encrypted message exceeds the frame budget")
            associated_data = pack_frame_header(
                message_type=int(message.message_type),
                flags=flags,
                priority=int(priority),
                stream_id=stream_id,
                sequence=sequence,
                acknowledgement=0,
                payload_length=ciphertext_length,
            )
            try:
                payload = secure_session.encrypt(
                    stream_id,
                    sequence,
                    encoded,
                    associated_data=associated_data,
                )
            except BaseException:
                try:
                    link.cancel_sequence(stream_id, sequence)
                except Exception:
                    pass
                raise
        try:
            if message.message_type is MessageType.INPUT_MOVE:
                link.send_pointer(
                    stream_id,
                    payload,
                    priority=priority,
                    message_type=int(message.message_type),
                    flags=flags,
                    sequence=sequence,
                )
            else:
                link.send(
                    OutboundItem(
                        priority,
                        stream_id,
                        payload,
                        message_type=int(message.message_type),
                        flags=flags,
                        sequence=sequence,
                    )
                )
        except BaseException:
            if sequence is not None:
                try:
                    link.cancel_sequence(stream_id, sequence)
                except Exception:
                    pass
            raise

    def _send_trust(self, connection_id: int) -> None:
        with self._lock:
            connection = self._connection
            if connection is None or connection.connection_id != connection_id:
                return
            fingerprint = connection.remote_fingerprint
            accepted = (
                fingerprint
                if connection.trust_status is TrustStatus.TRUSTED
                else None
            )
            if (
                connection.trust_sent
                and connection.last_accepts_fingerprint == accepted
            ):
                return
            message = Message(
                MessageType.TRUST,
                {
                    "protocol": PROTOCOL_VERSION,
                    "app_version": __version__,
                    "platform": sys.platform,
                    "peer_id": self.local_peer_id,
                    "features": sorted(self._local_features()),
                    "max_frame_size": MAX_PAYLOAD_SIZE,
                    "accepts_fingerprint": accepted,
                },
            )
        try:
            self._send_internal(
                connection_id,
                message,
                secure=True,
                priority=Priority.INTERACTIVE,
                allow_untrusted=True,
            )
        except (CoreError, LinkClosedError):
            with self._lock:
                current = self._connection
                interrupted = bool(
                    current is None
                    or current.connection_id != connection_id
                    or current.link.closed
                    or self._disconnect_requested.is_set()
                )
            if interrupted:
                return
            raise
        with self._lock:
            current = self._connection
            if current is not None and current.connection_id == connection_id:
                current.last_accepts_fingerprint = accepted
                current.trust_sent = True

    def _on_frame(self, connection_id: int, frame: Frame) -> None:
        try:
            self._process_frame(connection_id, frame)
        except Exception:
            logger.debug("discarded invalid serial frame", exc_info=self.debug)

    def _process_frame(self, connection_id: int, frame: Frame) -> None:
        try:
            message_type = MessageType(frame.message_type)
            priority = Priority(frame.priority)
        except ValueError:
            return
        if frame.flags not in {0, SECURE_FLAG}:
            return
        if priority not in _ALLOWED_PRIORITIES.get(message_type, frozenset()):
            return
        if frame.stream_id != _stream_id(message_type, priority):
            return
        secure = bool(frame.flags & SECURE_FLAG)
        if secure != (message_type not in _PLAIN_TYPES):
            return
        with self._lock:
            connection = self._connection
            if connection is None or connection.connection_id != connection_id:
                return
            secure_session = connection.secure_session
            ready = self._trusted_locked()
        if secure and message_type is not MessageType.TRUST and not ready:
            return
        payload = frame.payload
        if secure:
            associated_data = pack_frame_header(
                message_type=frame.message_type,
                flags=frame.flags,
                priority=frame.priority,
                stream_id=frame.stream_id,
                sequence=frame.sequence,
                acknowledgement=frame.acknowledgement,
                payload_length=len(frame.payload),
            )
            payload = secure_session.decrypt(
                frame.stream_id,
                frame.sequence,
                frame.payload,
                associated_data=associated_data,
            )
        message = decode_message(payload)
        if message is None or message.message_type is not message_type:
            return
        if not secure:
            with self._lock:
                connection = self._connection
                if connection is None or connection.connection_id != connection_id:
                    return
                last_sequence = connection.plain_receive_sequences.get(
                    frame.stream_id,
                    -1,
                )
                if frame.sequence <= last_sequence:
                    return
                connection.plain_receive_sequences[frame.stream_id] = frame.sequence
        if self.debug and message_type is not MessageType.INPUT_MOVE:
            logger.debug(
                "received %s frame (%d bytes%s)",
                message_type.name,
                len(frame.payload),
                ", secure" if secure else "",
            )
        if message_type is MessageType.HELLO:
            self._handle_hello(connection_id, message)
            return
        if message_type is MessageType.TRUST:
            self._handle_trust(connection_id, message)
            return
        feature = _FEATURE_BY_TYPE.get(message_type)
        with self._lock:
            connection = self._connection
            if connection is None or connection.connection_id != connection_id:
                return
            if feature not in connection.remote_features:
                return
        self._dispatch(message, feature)

    def _handle_hello(self, connection_id: int, message: Message) -> None:
        if message.metadata != {"protocol": PROTOCOL_VERSION}:
            return
        with self._lock:
            connection = self._connection
            if connection is None or connection.connection_id != connection_id:
                return
            secure_session = connection.secure_session
        secure_session.receive_hello(message.body)
        with self._lock:
            connection = self._connection
            if connection is None or connection.connection_id != connection_id:
                return
            connection.remote_hello_received = True
            send_trust = connection.local_hello_sent
        if send_trust:
            self._send_trust(connection_id)

    def _handle_trust(self, connection_id: int, message: Message) -> None:
        metadata = _validate_trust_metadata(message.metadata)
        with self._lock:
            connection = self._connection
            if connection is None or connection.connection_id != connection_id:
                return
            fingerprint = connection.secure_session.remote_fingerprint
            if fingerprint is None:
                return
        peer_id = metadata["peer_id"]
        if peer_id == self.local_peer_id:
            raise CoreError("peer ID matches the local installation")
        trust_status = self.trust_store.status(peer_id, fingerprint)
        bind_input = False
        with self._lock:
            connection = self._connection
            if connection is None or connection.connection_id != connection_id:
                return
            if (
                connection.remote_peer_id is not None
                and connection.remote_peer_id != peer_id
            ):
                raise CoreError("peer changed identity metadata")
            connection.remote_peer_id = peer_id
            connection.remote_fingerprint = fingerprint
            connection.trust_status = trust_status
            connection.remote_features = frozenset(metadata["features"])
            connection.remote_approved = (
                metadata["accepts_fingerprint"] == self.identity.fingerprint
            )
            bind_input = self.input is not None and not connection.input_bound
        if bind_input and self.input is not None:
            self.input.set_peer_id(peer_id)
            self.input.connection_changed(True)
            with self._lock:
                current = self._connection
                if current is not None and current.connection_id == connection_id:
                    current.input_bound = True
        self._send_trust(connection_id)
        self._refresh_input_if_ready(connection_id)
        self._publish_current()

    def _dispatch(self, message: Message, feature: str | None) -> None:
        if feature == "chat":
            service = self.chat
        elif feature == "files":
            service = self.files
        elif feature == "shell":
            service = self.shell
        elif feature == "input" and self.input is not None:
            service = self.input
        else:
            return
        token = _authenticated_message.set(
            message if message.message_type not in _PLAIN_TYPES else None
        )
        try:
            service.handle_message(message)
        finally:
            _authenticated_message.reset(token)

    def _on_disconnect(
        self,
        connection_id: int,
        error: BaseException | None,
    ) -> None:
        with self._lock:
            connection = self._connection
            if connection is None or connection.connection_id != connection_id:
                return
            disconnect_requested = self._disconnect_requested.is_set()
            self._connection = None
            self._last_connection_id = connection_id
            self._last_state = (
                CoreState.DISCONNECTED
                if disconnect_requested or error is None
                else CoreState.ERROR
            )
            self._last_error = (
                None if disconnect_requested or error is None else str(error)
            )
            self._disconnect_requested.clear()
        self.files.disconnect()
        self.shell.close()
        if self.input is not None:
            try:
                self.input.disconnect()
            except Exception:
                logger.exception("input cleanup failed during disconnect")
        self._publish_current()

    def _trusted_locked(self) -> bool:
        connection = self._connection
        return bool(
            connection is not None
            and not self._disconnect_requested.is_set()
            and connection.remote_peer_id is not None
            and connection.remote_fingerprint is not None
            and connection.trust_status is TrustStatus.TRUSTED
            and connection.remote_approved
        )

    def _snapshot_locked(self) -> CoreSnapshot:
        connection = self._connection
        if connection is None:
            return CoreSnapshot(
                self._last_connection_id,
                self._last_state,
                error=self._last_error,
            )
        if self._disconnect_requested.is_set():
            state = CoreState.DISCONNECTING
        elif connection.remote_peer_id is None:
            state = CoreState.HANDSHAKING
        elif connection.trust_status is TrustStatus.CHANGED:
            state = CoreState.CHANGED
        elif self._trusted_locked():
            state = CoreState.READY
        else:
            state = CoreState.UNTRUSTED
        return CoreSnapshot(
            connection.connection_id,
            state,
            connection.remote_peer_id,
            connection.remote_fingerprint,
            connection.trust_status,
            connection.trust_status is TrustStatus.TRUSTED,
            connection.remote_approved,
            connection.remote_features,
        )

    def _publish_current(self) -> None:
        with self._lock:
            snapshot = self._snapshot_locked()
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception:
                logger.exception("core state listener failed")

    def _refresh_input_if_ready(self, connection_id: int) -> None:
        with self._lock:
            connection = self._connection
            refresh = bool(
                self.input is not None
                and connection is not None
                and connection.connection_id == connection_id
                and connection.input_bound
                and self._trusted_locked()
            )
        if not refresh or self.input is None:
            return
        try:
            self.input.connection_changed(True)
        except Exception:
            logger.exception("input capture could not refresh after trust")

    def _local_features(self) -> frozenset[str]:
        if self.input is None:
            return LOCAL_FEATURES - {"input"}
        return LOCAL_FEATURES


def _stream_id(message_type: MessageType, priority: Priority) -> int:
    return int(message_type) * 4 + int(priority)


def _validate_peer_id(peer_id: Any) -> None:
    if not isinstance(peer_id, str) or not peer_id:
        raise ValueError("peer ID must be a non-empty string")
    try:
        encoded = peer_id.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("peer ID must be valid UTF-8") from error
    if len(encoded) > MAX_PEER_ID_BYTES or any(ord(char) < 32 for char in peer_id):
        raise ValueError("peer ID is invalid")


def _validate_trust_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    required = {
        "protocol",
        "app_version",
        "platform",
        "peer_id",
        "features",
        "max_frame_size",
        "accepts_fingerprint",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise CoreError("trust metadata fields are invalid")
    if metadata["protocol"] != PROTOCOL_VERSION:
        raise CoreError("peer protocol version is incompatible")
    if not isinstance(metadata["app_version"], str) or not metadata["app_version"]:
        raise CoreError("peer application version is invalid")
    if not isinstance(metadata["platform"], str) or not metadata["platform"]:
        raise CoreError("peer platform is invalid")
    _validate_peer_id(metadata["peer_id"])
    features = metadata["features"]
    if (
        not isinstance(features, list)
        or len(features) > MAX_FEATURES
        or any(not isinstance(item, str) or not item for item in features)
        or len(set(features)) != len(features)
    ):
        raise CoreError("peer feature list is invalid")
    max_frame_size = metadata["max_frame_size"]
    if type(max_frame_size) is not int or not 1024 <= max_frame_size <= MAX_PAYLOAD_SIZE:
        raise CoreError("peer frame size is invalid")
    accepted = metadata["accepts_fingerprint"]
    if accepted is not None and (
        not isinstance(accepted, str) or not accepted.startswith("SHA256:")
    ):
        raise CoreError("peer trust acceptance is invalid")
    return metadata


__all__ = [
    "CoreError",
    "CoreSnapshot",
    "CoreState",
    "ShookLinkCore",
]
