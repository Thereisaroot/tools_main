"""Selective-repeat file transfer with final SHA-256 verification."""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from shooklink.protocol.messages import Message, MessageType
from shooklink.transport.multiplexer import Priority

CHUNK_SIZE = 16 * 1024
WINDOW_SIZE = 16
ACK_BITMAP_BITS = 64
RETRANSMIT_TIMEOUT = 0.75
MAX_FILE_SIZE = 1 << 42
MAX_FILENAME_BYTES = 240
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRANSFER_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class FileTransferError(RuntimeError):
    """Base class for file-transfer failures."""


class FilePeerNotTrusted(FileTransferError):
    """Raised when a protected transfer is requested before trust."""


class FileProtocolError(FileTransferError):
    """Raised when peer file metadata is malformed or inconsistent."""


class FileHashMismatch(FileTransferError):
    """Raised when a completed partial file has the wrong SHA-256."""


class FileTransferCancelled(FileTransferError):
    """Raised when work is attempted after a transfer is cancelled."""


class FileBus(Protocol):
    trusted: bool

    def send(
        self,
        message: Message,
        *,
        secure: bool = True,
        priority: Priority = Priority.NORMAL,
    ) -> None: ...

    def decrypt_secure(self, message: Message) -> bytes: ...


@dataclass(frozen=True, slots=True)
class FileOffer:
    transfer_id: str
    name: str
    size: int
    mtime_ns: int
    sha256: str
    chunk_size: int = CHUNK_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.transfer_id, str) or not _TRANSFER_ID_RE.fullmatch(
            self.transfer_id
        ):
            raise ValueError("transfer_id must be 32 lowercase hexadecimal characters")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("file name must be a non-empty string")
        if type(self.size) is not int or not 0 <= self.size <= MAX_FILE_SIZE:
            raise ValueError(f"file size must be from 0 to {MAX_FILE_SIZE}")
        if type(self.mtime_ns) is not int or self.mtime_ns < 0:
            raise ValueError("mtime_ns must be a non-negative integer")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if type(self.chunk_size) is not int or not 0 < self.chunk_size <= CHUNK_SIZE:
            raise ValueError(f"chunk_size must be from 1 to {CHUNK_SIZE}")

    @property
    def chunk_count(self) -> int:
        return math.ceil(self.size / self.chunk_size)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "name": self.name,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> FileOffer:
        if not isinstance(metadata, Mapping):
            raise FileProtocolError("file offer metadata must be an object")
        required = {
            "transfer_id",
            "name",
            "size",
            "mtime_ns",
            "sha256",
            "chunk_size",
        }
        if set(metadata) != required:
            raise FileProtocolError("file offer fields are invalid")
        try:
            return cls(
                transfer_id=metadata["transfer_id"],
                name=metadata["name"],
                size=metadata["size"],
                mtime_ns=metadata["mtime_ns"],
                sha256=metadata["sha256"],
                chunk_size=metadata["chunk_size"],
            )
        except (TypeError, ValueError) as error:
            raise FileProtocolError(str(error)) from error


@dataclass(frozen=True, slots=True)
class FileProgress:
    transfer_id: str
    name: str
    direction: str
    transferred: int
    total: int
    state: str
    path: Path | None = None
    throughput_bps: float = 0.0


@dataclass(slots=True)
class _PendingChunk:
    index: int
    body: bytes
    sent_at: float
    gap_retried: bool = False


def sanitize_filename(untrusted_name: str) -> str:
    if not isinstance(untrusted_name, str):
        raise TypeError("file name must be a string")
    basename = untrusted_name.replace("\\", "/").rsplit("/", 1)[-1]
    basename = "".join(character for character in basename if ord(character) >= 32)
    basename = re.sub(r'[<>:"/\\|?*]', "_", basename).strip(" .")
    if not basename or basename in {".", ".."}:
        basename = "download"
    stem = basename.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED:
        basename = f"_{basename}"
    while len(basename.encode("utf-8")) > MAX_FILENAME_BYTES:
        basename = basename[:-1]
    return basename or "download"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_ack_metadata(received_indexes: Iterable[int]) -> dict[str, Any]:
    received = set(received_indexes)
    if any(type(index) is not int or index < 0 for index in received):
        raise ValueError("received indexes must be non-negative integers")
    base = 0
    while base in received:
        base += 1
    if not received or max(received) < base:
        span = 0
    else:
        span = min(ACK_BITMAP_BITS, max(received) - base + 1)
    bitmap = 0
    for offset in range(span):
        if base + offset in received:
            bitmap |= 1 << offset
    return {"base": base, "span": span, "bitmap": format(bitmap, "x")}


def _parse_ack_metadata(metadata: Mapping[str, Any]) -> tuple[int, int, int]:
    try:
        base = metadata["base"]
        span = metadata["span"]
        encoded_bitmap = metadata["bitmap"]
    except (KeyError, TypeError) as error:
        raise FileProtocolError("file ACK fields are missing") from error
    if type(base) is not int or base < 0:
        raise FileProtocolError("file ACK base is invalid")
    if type(span) is not int or not 0 <= span <= ACK_BITMAP_BITS:
        raise FileProtocolError("file ACK span is invalid")
    if not isinstance(encoded_bitmap, str) or not encoded_bitmap:
        raise FileProtocolError("file ACK bitmap is invalid")
    try:
        bitmap = int(encoded_bitmap, 16)
    except ValueError as error:
        raise FileProtocolError("file ACK bitmap is invalid") from error
    if bitmap < 0:
        raise FileProtocolError("file ACK bitmap is invalid")
    if (span == 0 and bitmap != 0) or (span > 0 and bitmap >= 1 << span):
        raise FileProtocolError("file ACK bitmap exceeds its span")
    return base, span, bitmap


class IncomingTransfer:
    def __init__(
        self,
        offer: FileOffer,
        final_path: Path,
        partial_path: Path,
        partial_file: BinaryIO,
    ) -> None:
        self.offer = offer
        self.final_path = final_path
        self.partial_path = partial_path
        self._file = partial_file
        self._received: set[int] = set()
        self._received_bytes = 0
        self._closed = False
        self._cancelled = False

    @classmethod
    def create(cls, offer: FileOffer, download_dir: str | Path) -> IncomingTransfer:
        directory = Path(download_dir)
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(offer.name)
        original = Path(safe_name)
        stem = original.stem or "download"
        suffix = original.suffix
        counter = 0
        while True:
            name = safe_name if counter == 0 else f"{stem} ({counter}){suffix}"
            final_path = directory / name
            partial_path = directory / f"{name}.part"
            if final_path.exists():
                counter += 1
                continue
            try:
                partial_file = partial_path.open("x+b")
            except FileExistsError:
                counter += 1
                continue
            return cls(offer, final_path, partial_path, partial_file)

    @property
    def chunk_count(self) -> int:
        return self.offer.chunk_count

    @property
    def received_bytes(self) -> int:
        return self._received_bytes

    @property
    def complete(self) -> bool:
        return len(self._received) == self.chunk_count

    def write_chunk(self, index: int, body: bytes) -> bool:
        self._ensure_active()
        if type(index) is not int or not 0 <= index < self.chunk_count:
            raise FileProtocolError("file chunk index is out of range")
        if not isinstance(body, bytes):
            raise TypeError("file chunk body must be bytes")
        offset = index * self.offer.chunk_size
        expected_size = min(self.offer.chunk_size, self.offer.size - offset)
        if len(body) != expected_size:
            raise FileProtocolError("file chunk has the wrong size")
        if index in self._received:
            return False
        self._file.seek(offset)
        written = self._file.write(body)
        if written != len(body):
            raise OSError("partial file write made no progress")
        self._received.add(index)
        self._received_bytes += len(body)
        return True

    def ack_metadata(self) -> dict[str, Any]:
        metadata = build_ack_metadata(self._received)
        metadata["transfer_id"] = self.offer.transfer_id
        return metadata

    def finalize(self) -> Path:
        self._ensure_active()
        if not self.complete:
            raise FileProtocolError("cannot finalize an incomplete file")
        try:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._closed = True
            actual_hash = _sha256_path(self.partial_path)
            if actual_hash != self.offer.sha256:
                self.partial_path.unlink(missing_ok=True)
                raise FileHashMismatch("completed file SHA-256 did not match")
            os.replace(self.partial_path, self.final_path)
            try:
                os.utime(
                    self.final_path,
                    ns=(self.offer.mtime_ns, self.offer.mtime_ns),
                )
            except OSError:
                pass
            return self.final_path
        except BaseException:
            if not self._closed:
                try:
                    self._file.close()
                finally:
                    self._closed = True
            if not self.final_path.exists():
                self.partial_path.unlink(missing_ok=True)
            raise

    def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        if not self._closed:
            try:
                self._file.close()
            finally:
                self._closed = True
        self.partial_path.unlink(missing_ok=True)

    def _ensure_active(self) -> None:
        if self._cancelled:
            raise FileTransferCancelled("file transfer was cancelled")
        if self._closed:
            raise FileTransferError("file transfer is closed")


class OutgoingTransfer:
    def __init__(
        self,
        path: Path,
        offer: FileOffer,
        *,
        window_size: int = WINDOW_SIZE,
        retransmit_timeout: float = RETRANSMIT_TIMEOUT,
    ) -> None:
        if type(window_size) is not int or window_size <= 0:
            raise ValueError("window_size must be positive")
        if retransmit_timeout <= 0:
            raise ValueError("retransmit_timeout must be positive")
        self.path = path
        self.offer = offer
        self.window_size = window_size
        self.retransmit_timeout = retransmit_timeout
        self._file: BinaryIO | None = None
        self._pending: dict[int, _PendingChunk] = {}
        self._next_index = 0
        self._accepted = False
        self._cancelled = False
        self._finish_sent = False

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        transfer_id: str | None = None,
        chunk_size: int = CHUNK_SIZE,
        window_size: int = WINDOW_SIZE,
        retransmit_timeout: float = RETRANSMIT_TIMEOUT,
    ) -> OutgoingTransfer:
        source = Path(path)
        stat = source.stat()
        if not source.is_file():
            raise FileTransferError("selected path is not a regular file")
        offer = FileOffer(
            transfer_id=transfer_id or uuid.uuid4().hex,
            name=source.name,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            sha256=_sha256_path(source),
            chunk_size=chunk_size,
        )
        return cls(
            source,
            offer,
            window_size=window_size,
            retransmit_timeout=retransmit_timeout,
        )

    @property
    def in_flight_count(self) -> int:
        return len(self._pending)

    @property
    def acknowledged_bytes(self) -> int:
        pending_bytes = sum(len(chunk.body) for chunk in self._pending.values())
        queued_bytes = min(self._next_index * self.offer.chunk_size, self.offer.size)
        return queued_bytes - pending_bytes

    def offer_message(self) -> Message:
        return Message(MessageType.FILE_OFFER, self.offer.to_metadata())

    def accept(self) -> None:
        if self._cancelled:
            raise FileTransferCancelled("file transfer was cancelled")
        if self._accepted:
            return
        self._file = self.path.open("rb")
        self._accepted = True

    def next_messages(self, *, now: float | None = None) -> list[Message]:
        self._ensure_active()
        timestamp = time.monotonic() if now is None else now
        messages: list[Message] = []
        while (
            len(self._pending) < self.window_size
            and self._next_index < self.offer.chunk_count
        ):
            messages.append(self._load_next_chunk(timestamp))
        if not self._pending and self._next_index == self.offer.chunk_count:
            finish = self._finish_message()
            if finish is not None:
                messages.append(finish)
        return messages

    def acknowledge(
        self,
        metadata: Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> list[Message]:
        self._ensure_active()
        base, span, bitmap = _parse_ack_metadata(metadata)
        if base > self.offer.chunk_count or base + span > self.offer.chunk_count:
            raise FileProtocolError("file ACK exceeds the transfer")
        timestamp = time.monotonic() if now is None else now
        for index in tuple(self._pending):
            if index < base:
                del self._pending[index]
                continue
            offset = index - base
            if 0 <= offset < span and bitmap & (1 << offset):
                del self._pending[index]

        messages: list[Message] = []
        for index in sorted(self._pending):
            offset = index - base
            pending = self._pending[index]
            if 0 <= offset < span and not bitmap & (1 << offset):
                if not pending.gap_retried:
                    pending.gap_retried = True
                    pending.sent_at = timestamp
                    messages.append(self._chunk_message(pending))
        messages.extend(self.next_messages(now=timestamp))
        return messages

    def retransmit_expired(self, *, now: float | None = None) -> list[Message]:
        self._ensure_active()
        timestamp = time.monotonic() if now is None else now
        messages = []
        for pending in sorted(self._pending.values(), key=lambda item: item.index):
            if timestamp - pending.sent_at >= self.retransmit_timeout:
                pending.sent_at = timestamp
                messages.append(self._chunk_message(pending))
        return messages

    def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self._pending.clear()
        self._close_file()

    def _ensure_active(self) -> None:
        if self._cancelled:
            raise FileTransferCancelled("file transfer was cancelled")
        if not self._accepted:
            raise FileTransferError("file offer has not been accepted")

    def _load_next_chunk(self, timestamp: float) -> Message:
        if self._file is None:
            raise FileTransferError("source file is not open")
        index = self._next_index
        self._file.seek(index * self.offer.chunk_size)
        body = self._file.read(self.offer.chunk_size)
        expected = min(
            self.offer.chunk_size,
            self.offer.size - index * self.offer.chunk_size,
        )
        if len(body) != expected:
            raise FileTransferError("source file changed during transfer")
        pending = _PendingChunk(index, body, timestamp)
        self._pending[index] = pending
        self._next_index += 1
        return self._chunk_message(pending)

    def _chunk_message(self, pending: _PendingChunk) -> Message:
        return Message(
            MessageType.FILE_CHUNK,
            {
                "transfer_id": self.offer.transfer_id,
                "index": pending.index,
            },
            pending.body,
        )

    def _finish_message(self) -> Message | None:
        if self._finish_sent:
            return None
        self._finish_sent = True
        self._close_file()
        return Message(
            MessageType.FILE_FINISH,
            {
                "transfer_id": self.offer.transfer_id,
                "sha256": self.offer.sha256,
            },
        )

    def _close_file(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class FileService:
    """Route typed file messages and keep transfers isolated by session ID."""

    def __init__(
        self,
        bus: FileBus,
        download_dir: str | Path,
        *,
        executor: Executor | None = None,
    ) -> None:
        self._bus = bus
        self.download_dir = Path(download_dir)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="shooklink-file",
        )
        self._owns_executor = executor is None
        self._lock = threading.RLock()
        self._outgoing: dict[str, OutgoingTransfer] = {}
        self._incoming: dict[str, IncomingTransfer] = {}
        self._listeners: list[Callable[[FileProgress], None]] = []
        self._progress_started: dict[str, float] = {}
        self._closed = False

    def add_progress_listener(self, listener: Callable[[FileProgress], None]) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_progress_listener(self, listener: Callable[[FileProgress], None]) -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    def send_file(self, path: str | Path) -> Future[str]:
        self._ensure_trusted()
        with self._lock:
            if self._closed:
                raise FileTransferError("file service is closed")
        future = self._executor.submit(OutgoingTransfer.from_path, path)
        result: Future[str] = Future()

        def prepared(preparation: Future[OutgoingTransfer]) -> None:
            try:
                transfer = preparation.result()
                with self._lock:
                    if self._closed:
                        transfer.cancel()
                        raise FileTransferError("file service is closed")
                    self._outgoing[transfer.offer.transfer_id] = transfer
                self._send(transfer.offer_message(), Priority.NORMAL)
                self._notify_outgoing(transfer, "offered")
                result.set_result(transfer.offer.transfer_id)
            except BaseException as error:
                result.set_exception(error)

        future.add_done_callback(prepared)
        return result

    def handle_message(self, message: Message) -> bool:
        handlers = {
            MessageType.FILE_OFFER: self._handle_offer,
            MessageType.FILE_ACCEPT: self._handle_accept,
            MessageType.FILE_CHUNK: self._handle_chunk,
            MessageType.FILE_ACK: self._handle_ack,
            MessageType.FILE_FINISH: self._handle_finish,
            MessageType.FILE_CANCEL: self._handle_cancel,
        }
        handler = handlers.get(message.message_type)
        if handler is None:
            return False
        self._ensure_trusted()
        try:
            body = self._bus.decrypt_secure(message)
        except Exception as error:
            raise FileProtocolError("file message authentication failed") from error
        if not isinstance(body, bytes):
            raise FileProtocolError("file message decryption returned invalid data")
        handler(Message(message.message_type, message.metadata, body))
        return True

    def poll(self, *, now: float | None = None) -> None:
        with self._lock:
            transfers = tuple(self._outgoing.values())
        for transfer in transfers:
            try:
                messages = transfer.retransmit_expired(now=now)
            except FileTransferError:
                continue
            self._send_many(messages)

    def cancel(self, transfer_id: str) -> None:
        transfer = None
        with self._lock:
            transfer = self._outgoing.pop(transfer_id, None)
            if transfer is None:
                transfer = self._incoming.pop(transfer_id, None)
        if transfer is None:
            return
        transfer.cancel()
        self._send(
            Message(MessageType.FILE_CANCEL, {"transfer_id": transfer_id}),
            Priority.INTERACTIVE,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            transfers = tuple(self._outgoing.values()) + tuple(self._incoming.values())
            self._outgoing.clear()
            self._incoming.clear()
        for transfer in transfers:
            transfer.cancel()
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _handle_offer(self, message: Message) -> None:
        self._ensure_trusted()
        if message.body:
            raise FileProtocolError("file offer cannot contain a body")
        offer = FileOffer.from_metadata(message.metadata)
        with self._lock:
            if offer.transfer_id in self._incoming or offer.transfer_id in self._outgoing:
                raise FileProtocolError("duplicate file transfer ID")
            transfer = IncomingTransfer.create(offer, self.download_dir)
            self._incoming[offer.transfer_id] = transfer
        self._send(
            Message(MessageType.FILE_ACCEPT, {"transfer_id": offer.transfer_id}),
            Priority.NORMAL,
        )
        self._notify_incoming(transfer, "receiving")

    def _handle_accept(self, message: Message) -> None:
        transfer = self._get_outgoing(message.metadata)
        transfer.accept()
        self._send_many(transfer.next_messages())
        self._notify_outgoing(transfer, "sending")

    def _handle_chunk(self, message: Message) -> None:
        transfer = self._get_incoming(message.metadata)
        index = message.metadata.get("index")
        transfer.write_chunk(index, message.body)
        self._send(Message(MessageType.FILE_ACK, transfer.ack_metadata()), Priority.NORMAL)
        self._notify_incoming(transfer, "receiving")

    def _handle_ack(self, message: Message) -> None:
        transfer = self._get_outgoing(message.metadata)
        self._send_many(transfer.acknowledge(message.metadata))
        self._notify_outgoing(transfer, "sending")

    def _handle_finish(self, message: Message) -> None:
        transfer_id = _metadata_transfer_id(message.metadata)
        status = message.metadata.get("status")
        if status is not None:
            if status not in {"ok", "error"}:
                raise FileProtocolError("file finish status is invalid")
            with self._lock:
                transfer = self._outgoing.pop(transfer_id, None)
            if transfer is not None:
                self._notify_outgoing(transfer, "complete" if status == "ok" else "failed")
            return

        with self._lock:
            transfer = self._incoming.get(transfer_id)
        if transfer is None:
            raise FileProtocolError("unknown incoming transfer")
        if message.metadata.get("sha256") != transfer.offer.sha256:
            raise FileProtocolError("file finish hash differs from offer")
        future = self._executor.submit(transfer.finalize)

        def finalized(completion: Future[Path]) -> None:
            try:
                path = completion.result()
            except BaseException:
                transfer.cancel()
                self._send(
                    Message(
                        MessageType.FILE_FINISH,
                        {"transfer_id": transfer_id, "status": "error"},
                    ),
                    Priority.NORMAL,
                )
                self._notify_incoming(transfer, "failed")
            else:
                self._send(
                    Message(
                        MessageType.FILE_FINISH,
                        {"transfer_id": transfer_id, "status": "ok"},
                    ),
                    Priority.NORMAL,
                )
                self._notify(
                    FileProgress(
                        transfer_id,
                        transfer.offer.name,
                        "incoming",
                        transfer.offer.size,
                        transfer.offer.size,
                        "complete",
                        path,
                    )
                )
            finally:
                with self._lock:
                    self._incoming.pop(transfer_id, None)

        future.add_done_callback(finalized)

    def _handle_cancel(self, message: Message) -> None:
        transfer_id = _metadata_transfer_id(message.metadata)
        with self._lock:
            transfer = self._outgoing.pop(transfer_id, None)
            if transfer is None:
                transfer = self._incoming.pop(transfer_id, None)
        if transfer is not None:
            transfer.cancel()

    def _get_outgoing(self, metadata: Mapping[str, Any]) -> OutgoingTransfer:
        transfer_id = _metadata_transfer_id(metadata)
        with self._lock:
            transfer = self._outgoing.get(transfer_id)
        if transfer is None:
            raise FileProtocolError("unknown outgoing transfer")
        return transfer

    def _get_incoming(self, metadata: Mapping[str, Any]) -> IncomingTransfer:
        transfer_id = _metadata_transfer_id(metadata)
        with self._lock:
            transfer = self._incoming.get(transfer_id)
        if transfer is None:
            raise FileProtocolError("unknown incoming transfer")
        return transfer

    def _send_many(self, messages: Iterable[Message]) -> None:
        for message in messages:
            priority = (
                Priority.FILE
                if message.message_type is MessageType.FILE_CHUNK
                else Priority.NORMAL
            )
            self._send(message, priority)

    def _send(self, message: Message, priority: Priority) -> None:
        self._bus.send(message, secure=True, priority=priority)

    def _ensure_trusted(self) -> None:
        if not self._bus.trusted:
            raise FilePeerNotTrusted("trust the connected peer before transferring files")

    def _notify_outgoing(self, transfer: OutgoingTransfer, state: str) -> None:
        self._notify(
            FileProgress(
                transfer.offer.transfer_id,
                transfer.offer.name,
                "outgoing",
                transfer.acknowledged_bytes,
                transfer.offer.size,
                state,
                transfer.path,
            )
        )

    def _notify_incoming(self, transfer: IncomingTransfer, state: str) -> None:
        self._notify(
            FileProgress(
                transfer.offer.transfer_id,
                transfer.offer.name,
                "incoming",
                transfer.received_bytes,
                transfer.offer.size,
                state,
                transfer.partial_path,
            )
        )

    def _notify(self, progress: FileProgress) -> None:
        now = time.monotonic()
        with self._lock:
            started = self._progress_started.setdefault(progress.transfer_id, now)
            elapsed = max(0.001, now - started)
            progress = replace(
                progress,
                throughput_bps=(progress.transferred / elapsed),
            )
            if progress.state in {"complete", "failed", "cancelled"}:
                self._progress_started.pop(progress.transfer_id, None)
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(progress)


def _metadata_transfer_id(metadata: Mapping[str, Any]) -> str:
    try:
        transfer_id = metadata["transfer_id"]
    except (KeyError, TypeError) as error:
        raise FileProtocolError("transfer_id is missing") from error
    if not isinstance(transfer_id, str) or not _TRANSFER_ID_RE.fullmatch(transfer_id):
        raise FileProtocolError("transfer_id is invalid")
    return transfer_id


__all__ = [
    "ACK_BITMAP_BITS",
    "CHUNK_SIZE",
    "MAX_FILE_SIZE",
    "RETRANSMIT_TIMEOUT",
    "WINDOW_SIZE",
    "FileHashMismatch",
    "FileOffer",
    "FilePeerNotTrusted",
    "FileProgress",
    "FileProtocolError",
    "FileService",
    "FileTransferCancelled",
    "FileTransferError",
    "IncomingTransfer",
    "OutgoingTransfer",
    "build_ack_metadata",
    "sanitize_filename",
]
