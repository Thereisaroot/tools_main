"""Plain and authenticated text-message workflows."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from shooklink.protocol.messages import Message, MessageType

MAX_CHAT_TEXT_BYTES = 48 * 1024


class ChatError(RuntimeError):
    """Base class for chat workflow errors."""


class PeerNotTrusted(ChatError):
    """Raised when secure text is requested before peer trust is established."""


class ChatTextTooLarge(ChatError):
    """Raised when a UTF-8 message cannot fit inside one chat frame."""


class ChatBus(Protocol):
    trusted: bool

    def send(self, message: Message, *, secure: bool = False) -> None:
        """Queue a typed message for transport."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    text: str
    secure: bool


class ChatService:
    """Validate, send, and retain only user-visible chat content."""

    def __init__(self, bus: ChatBus) -> None:
        self._bus = bus
        self._lock = threading.RLock()
        self._listeners: list[Callable[[ChatMessage], None]] = []
        self._last_text = ""
        self._last_secure_text = ""

    @property
    def secure_available(self) -> bool:
        return bool(self._bus.trusted)

    @property
    def last_text(self) -> str:
        with self._lock:
            return self._last_text

    @property
    def last_secure_text(self) -> str:
        with self._lock:
            return self._last_secure_text

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

    def send_plain(self, text: str) -> None:
        body = self._encode_text(text)
        self._bus.send(Message(MessageType.CHAT_PLAIN, {}, body), secure=False)

    def send_secure(self, text: str) -> None:
        if not self.secure_available:
            raise PeerNotTrusted("trust the connected peer before sending secure text")
        body = self._encode_text(text)
        self._bus.send(Message(MessageType.CHAT_SECURE, {}, body), secure=True)

    def handle_message(self, message: Message) -> bool:
        if message.message_type not in (
            MessageType.CHAT_PLAIN,
            MessageType.CHAT_SECURE,
        ):
            return False
        if len(message.body) > MAX_CHAT_TEXT_BYTES:
            return False
        try:
            text = message.body.decode("utf-8")
        except UnicodeDecodeError:
            return False

        received = ChatMessage(
            text=text,
            secure=message.message_type is MessageType.CHAT_SECURE,
        )
        with self._lock:
            self._last_text = text
            if received.secure:
                self._last_secure_text = text
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(received)
        return True

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
    "MAX_CHAT_TEXT_BYTES",
    "ChatError",
    "ChatMessage",
    "ChatService",
    "ChatTextTooLarge",
    "PeerNotTrusted",
]
