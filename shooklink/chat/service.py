"""Plain and authenticated text-message workflows."""

from __future__ import annotations

import logging
import math
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from shooklink.protocol.messages import Message, MessageType

MAX_CHAT_TEXT_BYTES = 48 * 1024
CHAT_ACK_FEATURE = "chat_ack_v1"
DEFAULT_RETRY_INTERVAL = 0.35
DEFAULT_MAX_ATTEMPTS = 8
MAX_RECEIVED_MESSAGE_IDS = 2_048

logger = logging.getLogger(__name__)


class ChatError(RuntimeError):
    """Base class for chat workflow errors."""


class PeerNotTrusted(ChatError):
    """Raised when secure text is requested before peer trust is established."""


class ChatTextTooLarge(ChatError):
    """Raised when a UTF-8 message cannot fit inside one chat frame."""


class ChatBus(Protocol):
    trusted: bool
    chat_ack_available: bool
    chat_connection_id: int | None

    def send(
        self,
        message: Message,
        *,
        secure: bool = False,
        on_written: Callable[[], None] | None = None,
        expected_connection_id: int | None = None,
    ) -> None:
        """Queue a typed message for transport."""

    def decrypt_secure(self, message: Message) -> bytes:
        """Authenticate and decrypt an incoming secure message body."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    text: str
    secure: bool


@dataclass(slots=True)
class _PendingChat:
    message: Message
    secure: bool
    connection_id: int
    attempts: int
    on_delivered: Callable[[], None] | None = None
    on_failed: Callable[[str], None] | None = None
    timer: threading.Timer | None = field(default=None, repr=False)


class ChatService:
    """Validate, send, and retain only user-visible chat content."""

    def __init__(
        self,
        bus: ChatBus,
        *,
        retry_interval: float = DEFAULT_RETRY_INTERVAL,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if (
            isinstance(retry_interval, bool)
            or not isinstance(retry_interval, (int, float))
            or not math.isfinite(retry_interval)
            or retry_interval <= 0
        ):
            raise ValueError("retry_interval must be positive")
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self._bus = bus
        self._retry_interval = float(retry_interval)
        self._max_attempts = max_attempts
        self._lock = threading.RLock()
        self._listeners: list[Callable[[ChatMessage], None]] = []
        self._last_text = ""
        self._pending: dict[str, _PendingChat] = {}
        self._received_ids: OrderedDict[str, None] = OrderedDict()

    @property
    def secure_available(self) -> bool:
        return bool(self._bus.trusted)

    @property
    def last_text(self) -> str:
        with self._lock:
            return self._last_text

    def add_message_listener(self, listener: Callable[[ChatMessage], None]) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_message_listener(self, listener: Callable[[ChatMessage], None]) -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    def send_plain(
        self,
        text: str,
        *,
        on_written: Callable[[], None] | None = None,
        on_delivered: Callable[[], None] | None = None,
        on_failed: Callable[[str], None] | None = None,
    ) -> None:
        body = self._encode_text(text)
        self._send_chat(
            MessageType.CHAT_PLAIN,
            body,
            secure=False,
            on_written=on_written,
            on_delivered=on_delivered,
            on_failed=on_failed,
        )

    def send_secure(
        self,
        text: str,
        *,
        on_written: Callable[[], None] | None = None,
        on_delivered: Callable[[], None] | None = None,
        on_failed: Callable[[str], None] | None = None,
    ) -> None:
        if not self.secure_available:
            raise PeerNotTrusted("trust the connected peer before sending secure text")
        body = self._encode_text(text)
        self._send_chat(
            MessageType.CHAT_SECURE,
            body,
            secure=True,
            on_written=on_written,
            on_delivered=on_delivered,
            on_failed=on_failed,
        )

    def handle_message(self, message: Message) -> bool:
        if message.message_type not in (
            MessageType.CHAT_PLAIN,
            MessageType.CHAT_SECURE,
        ):
            return False
        body = message.body
        secure = message.message_type is MessageType.CHAT_SECURE
        if secure:
            if not self.secure_available:
                return False
            try:
                body = self._bus.decrypt_secure(message)
            except Exception:
                return False
            if not isinstance(body, bytes):
                return False
        if set(message.metadata) == {"ack_id"}:
            message_id = message.metadata["ack_id"]
            if body or not self._valid_message_id(message_id):
                return False
            self._acknowledge_delivery(message_id, secure=secure)
            return True
        if message.metadata == {}:
            message_id = None
        elif set(message.metadata) == {"message_id"}:
            message_id = message.metadata["message_id"]
            if not self._valid_message_id(message_id):
                return False
        else:
            return False
        if len(body) > MAX_CHAT_TEXT_BYTES:
            return False
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            return False

        duplicate = False
        with self._lock:
            if message_id is not None and message_id in self._received_ids:
                duplicate = True
                self._received_ids.move_to_end(message_id)
            else:
                if message_id is not None:
                    self._received_ids[message_id] = None
                    while len(self._received_ids) > MAX_RECEIVED_MESSAGE_IDS:
                        self._received_ids.popitem(last=False)
                self._last_text = text
                listeners = tuple(self._listeners)
        if message_id is not None:
            self._send_ack(message_id, secure=secure)
        if duplicate:
            return True
        received = ChatMessage(text=text, secure=secure)
        for listener in listeners:
            listener(received)
        return True

    def disconnect(self) -> None:
        with self._lock:
            pending = tuple(self._pending.values())
            self._pending.clear()
            self._received_ids.clear()
        for item in pending:
            if item.timer is not None:
                item.timer.cancel()
            self._invoke_failed(item.on_failed, "Disconnected")

    def _send_chat(
        self,
        message_type: MessageType,
        body: bytes,
        *,
        secure: bool,
        on_written: Callable[[], None] | None,
        on_delivered: Callable[[], None] | None,
        on_failed: Callable[[str], None] | None,
    ) -> None:
        self._validate_callback("on_written", on_written)
        self._validate_callback("on_delivered", on_delivered)
        self._validate_callback("on_failed", on_failed)
        connection_id = getattr(self._bus, "chat_connection_id", None)
        expected_connection_id = (
            connection_id
            if type(connection_id) is int and connection_id > 0
            else None
        )
        if not bool(getattr(self._bus, "chat_ack_available", False)):
            callback = None
            if on_written is not None or on_delivered is not None:
                def legacy_written() -> None:
                    self._invoke(on_written)
                    self._invoke(on_delivered)

                callback = legacy_written
            self._bus.send(
                Message(message_type, {}, body),
                secure=secure,
                on_written=callback,
                expected_connection_id=expected_connection_id,
            )
            return

        if type(connection_id) is not int or connection_id <= 0:
            raise ChatError("connect a serial peer first")
        message_id = uuid.uuid4().hex
        message = Message(message_type, {"message_id": message_id}, body)
        pending = _PendingChat(
            message,
            secure,
            connection_id,
            attempts=1,
            on_delivered=on_delivered,
            on_failed=on_failed,
        )
        with self._lock:
            self._pending[message_id] = pending

        def first_written() -> None:
            self._attempt_written(message_id)
            self._invoke(on_written)

        try:
            self._bus.send(
                message,
                secure=secure,
                on_written=first_written,
                expected_connection_id=connection_id,
            )
        except BaseException:
            self._discard_pending(message_id)
            raise

    def _attempt_written(self, message_id: str) -> None:
        with self._lock:
            pending = self._pending.get(message_id)
            if pending is None or pending.timer is not None:
                return
            timer = threading.Timer(
                self._retry_interval,
                self._retry_pending,
                args=(message_id,),
            )
            timer.daemon = True
            pending.timer = timer
        try:
            timer.start()
        except BaseException:
            failed = self._discard_pending(message_id)
            if failed is not None:
                self._invoke_failed(failed.on_failed, "Delivery failed")

    def _retry_pending(self, message_id: str) -> None:
        with self._lock:
            pending = self._pending.get(message_id)
            if pending is None:
                return
            pending.timer = None
            if pending.attempts >= self._max_attempts:
                del self._pending[message_id]
                failed_callback = pending.on_failed
                message = None
                secure = False
            else:
                pending.attempts += 1
                failed_callback = None
                message = pending.message
                secure = pending.secure
                connection_id = pending.connection_id
        if message is None:
            self._invoke_failed(failed_callback, "Delivery failed")
            return
        try:
            self._bus.send(
                message,
                secure=secure,
                on_written=lambda: self._attempt_written(message_id),
                expected_connection_id=connection_id,
            )
        except Exception:
            failed = self._discard_pending(message_id)
            if failed is not None:
                self._invoke_failed(failed.on_failed, "Delivery failed")

    def _acknowledge_delivery(self, message_id: str, *, secure: bool) -> None:
        with self._lock:
            pending = self._pending.get(message_id)
            if pending is None or pending.secure != secure:
                return
            del self._pending[message_id]
        if pending.timer is not None:
            pending.timer.cancel()
        self._invoke(pending.on_delivered)

    def _send_ack(self, message_id: str, *, secure: bool) -> None:
        message_type = MessageType.CHAT_SECURE if secure else MessageType.CHAT_PLAIN
        connection_id = getattr(self._bus, "chat_connection_id", None)
        if type(connection_id) is not int or connection_id <= 0:
            return
        try:
            self._bus.send(
                Message(message_type, {"ack_id": message_id}, b""),
                secure=secure,
                expected_connection_id=connection_id,
            )
        except Exception:
            logger.debug("chat delivery ACK could not be queued", exc_info=True)

    def _discard_pending(self, message_id: str) -> _PendingChat | None:
        with self._lock:
            pending = self._pending.pop(message_id, None)
        if pending is not None and pending.timer is not None:
            pending.timer.cancel()
        return pending

    @staticmethod
    def _valid_message_id(value) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) == 32
            and all(char in "0123456789abcdef" for char in value)
        )

    @staticmethod
    def _validate_callback(name: str, callback) -> None:
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable")

    @staticmethod
    def _invoke(callback: Callable[[], None] | None) -> None:
        if callback is None:
            return
        try:
            callback()
        except BaseException:
            logger.exception("chat callback failed")

    @staticmethod
    def _invoke_failed(
        callback: Callable[[str], None] | None,
        reason: str,
    ) -> None:
        if callback is None:
            return
        try:
            callback(reason)
        except BaseException:
            logger.exception("chat failure callback failed")

    @staticmethod
    def _encode_text(text: str) -> bytes:
        if not isinstance(text, str):
            raise TypeError("chat text must be a string")
        body = text.encode("utf-8")
        if len(body) > MAX_CHAT_TEXT_BYTES:
            raise ChatTextTooLarge(
                f"chat text cannot exceed {MAX_CHAT_TEXT_BYTES} UTF-8 bytes"
            )
        return body


__all__ = [
    "CHAT_ACK_FEATURE",
    "MAX_CHAT_TEXT_BYTES",
    "ChatError",
    "ChatMessage",
    "ChatService",
    "ChatTextTooLarge",
    "PeerNotTrusted",
]
