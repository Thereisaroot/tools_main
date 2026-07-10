"""Selective-repeat file transfer with final SHA-256 verification."""

from __future__ import annotations

import hashlib
import logging
import math
import ntpath
import os
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
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
MAX_MTIME_NS = (1 << 63) - 1
MAX_FILENAME_BYTES = 240
DEFAULT_MAX_INCOMING_TRANSFERS = 8
DEFAULT_MAX_OUTGOING_TRANSFERS = 8
DEFAULT_MAX_INCOMING_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_TERMINAL_TRANSFERS = 4_096
MAX_COMPLETED_TRANSFERS = 128
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

logger = logging.getLogger(__name__)


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
        if type(self.mtime_ns) is not int or not 0 <= self.mtime_ns <= MAX_MTIME_NS:
            raise ValueError(f"mtime_ns must be from 0 to {MAX_MTIME_NS}")
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


@dataclass(slots=True)
class _PendingCancel:
    last_sent_at: float
    direction: str


@dataclass(frozen=True, slots=True)
class _TerminalTransfer:
    status: str
    offer: FileOffer | None


def sanitize_filename(untrusted_name: str) -> str:
    if not isinstance(untrusted_name, str):
        raise TypeError("file name must be a string")
    basename = untrusted_name.replace("\\", "/").rsplit("/", 1)[-1]
    basename = "".join(
        "_" if 0xD800 <= ord(character) <= 0xDFFF else character
        for character in basename
        if ord(character) >= 32 and ord(character) != 127
    )
    basename = re.sub(r'[<>:"/\\|?*]', "_", basename).strip(" .")
    if not basename or basename in {".", ".."}:
        basename = "download"
    stem = basename.split(".", 1)[0].rstrip(" .").upper()
    if stem in _WINDOWS_RESERVED or _is_windows_reserved(basename):
        basename = f"_{basename}"
    while len(basename.encode("utf-8")) > MAX_FILENAME_BYTES:
        basename = basename[:-1]
    return basename or "download"


def _is_windows_reserved(name: str) -> bool:
    checker = getattr(ntpath, "isreserved", None)
    if checker is not None and checker(name):
        return True
    stem = name.split(".", 1)[0].rstrip(" .").upper()
    return bool(
        stem in {"CONIN$", "CONOUT$"}
        or re.fullmatch(r"(?:COM|LPT)[1-9¹²³]", stem)
    )


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
        self._lock = threading.RLock()
        self._state = "active"
        self._cancel_requested = False

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
        with self._lock:
            return self._received_bytes

    @property
    def complete(self) -> bool:
        with self._lock:
            return len(self._received) == self.chunk_count

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def write_chunk(self, index: int, body: bytes) -> bool:
        with self._lock:
            self._ensure_active_locked()
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
        with self._lock:
            metadata = build_ack_metadata(self._received)
            metadata["transfer_id"] = self.offer.transfer_id
            return metadata

    def finalize(self) -> Path:
        with self._lock:
            self._ensure_active_locked()
            if len(self._received) != self.chunk_count:
                raise FileProtocolError("cannot finalize an incomplete file")
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._state = "finalizing"
        try:
            actual_hash = _sha256_path(self.partial_path)
            with self._lock:
                if self._cancel_requested:
                    self._state = "cancelled"
                    self.partial_path.unlink(missing_ok=True)
                    raise FileTransferCancelled("file transfer was cancelled")
                if actual_hash != self.offer.sha256:
                    self._state = "failed"
                    self.partial_path.unlink(missing_ok=True)
                    raise FileHashMismatch("completed file SHA-256 did not match")
                self.final_path = self._commit_without_overwrite()
                self._state = "completed"
            try:
                os.utime(
                    self.final_path,
                    ns=(self.offer.mtime_ns, self.offer.mtime_ns),
                )
            except (OSError, OverflowError, ValueError):
                pass
            return self.final_path
        except BaseException:
            with self._lock:
                if self._state == "finalizing":
                    self._state = "failed"
                self.partial_path.unlink(missing_ok=True)
            raise

    def cancel(self) -> bool:
        with self._lock:
            if self._state in {"cancelled", "failed", "completed"}:
                return False
            if self._state == "finalizing":
                self._cancel_requested = True
                return True
            self._state = "cancelled"
            try:
                self._file.close()
            finally:
                self.partial_path.unlink(missing_ok=True)
            return True

    def _ensure_active_locked(self) -> None:
        if self._state == "cancelled":
            raise FileTransferCancelled("file transfer was cancelled")
        if self._state != "active":
            raise FileTransferError("file transfer is closed")

    def _commit_without_overwrite(self) -> Path:
        safe_name = self.final_path.name
        original = Path(safe_name)
        stem = original.stem or "download"
        suffix = original.suffix
        counter = 0
        while True:
            name = safe_name if counter == 0 else f"{stem} ({counter}){suffix}"
            candidate = self.final_path.parent / name
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                counter += 1
                continue
            os.close(descriptor)
            try:
                os.replace(self.partial_path, candidate)
            except BaseException:
                candidate.unlink(missing_ok=True)
                raise
            return candidate


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
        self._completed = False
        self._offer_sent_at: float | None = None
        self._finish_sent_at: float | None = None

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

    @property
    def accepted(self) -> bool:
        return self._accepted

    def offer_message(self, *, now: float | None = None) -> Message:
        self._offer_sent_at = time.monotonic() if now is None else now
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
            finish = self._finish_message(timestamp)
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

    def retry_control(self, *, now: float | None = None) -> list[Message]:
        if self._cancelled or self._completed:
            return []
        timestamp = time.monotonic() if now is None else now
        if not self._accepted:
            if (
                self._offer_sent_at is None
                or timestamp - self._offer_sent_at >= self.retransmit_timeout
            ):
                return [self.offer_message(now=timestamp)]
            return []
        if (
            self._finish_sent
            and self._finish_sent_at is not None
            and timestamp - self._finish_sent_at >= self.retransmit_timeout
        ):
            self._finish_sent_at = timestamp
            return [self._new_finish_message()]
        return []

    def mark_completed(self) -> None:
        self._completed = True

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

    def _finish_message(self, timestamp: float) -> Message | None:
        if self._finish_sent:
            return None
        self._finish_sent = True
        self._finish_sent_at = timestamp
        self._close_file()
        return self._new_finish_message()

    def _new_finish_message(self) -> Message:
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
        accept_offer: Callable[[FileOffer], bool] | None = None,
        max_incoming_transfers: int = DEFAULT_MAX_INCOMING_TRANSFERS,
        max_outgoing_transfers: int = DEFAULT_MAX_OUTGOING_TRANSFERS,
        max_incoming_bytes: int = DEFAULT_MAX_INCOMING_BYTES,
        max_terminal_transfers: int = DEFAULT_MAX_TERMINAL_TRANSFERS,
    ) -> None:
        if type(max_incoming_transfers) is not int or max_incoming_transfers <= 0:
            raise ValueError("max_incoming_transfers must be positive")
        if type(max_outgoing_transfers) is not int or max_outgoing_transfers <= 0:
            raise ValueError("max_outgoing_transfers must be positive")
        if type(max_incoming_bytes) is not int or max_incoming_bytes <= 0:
            raise ValueError("max_incoming_bytes must be positive")
        if type(max_terminal_transfers) is not int or max_terminal_transfers <= 0:
            raise ValueError("max_terminal_transfers must be positive")
        self._bus = bus
        self.download_dir = Path(download_dir)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="shooklink-file",
        )
        self._owns_executor = executor is None
        self._accept_offer = accept_offer or (lambda _offer: True)
        self._max_incoming_transfers = max_incoming_transfers
        self._max_outgoing_transfers = max_outgoing_transfers
        self._max_incoming_bytes = max_incoming_bytes
        self._max_terminal_transfers = max_terminal_transfers
        self._lock = threading.RLock()
        self._outgoing: dict[str, OutgoingTransfer] = {}
        self._incoming: dict[str, IncomingTransfer] = {}
        self._finalizing: dict[str, Future[Path]] = {}
        self._completed_incoming: OrderedDict[
            str, tuple[FileOffer, str, Path | None]
        ] = OrderedDict()
        self._cancelled_incoming: OrderedDict[str, FileOffer | None] = OrderedDict()
        self._terminal_status: dict[str, _TerminalTransfer] = {}
        self._pending_cancels: dict[str, _PendingCancel] = {}
        self._preparing_count = 0
        self._listeners: list[Callable[[FileProgress], None]] = []
        self._progress_started: dict[str, float] = {}
        self._closed = False
        self._timer_stop = threading.Event()
        self._timer_thread = threading.Thread(
            target=self._timer_loop,
            name="shooklink-file-timer",
            daemon=True,
        )
        self._timer_thread.start()

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
            if (
                len(self._outgoing)
                + self._preparing_count
                + len(self._pending_cancels)
                >= self._max_outgoing_transfers
            ):
                raise FileTransferError("outgoing file transfer capacity is full")
            self._preparing_count += 1
        try:
            future = self._executor.submit(OutgoingTransfer.from_path, path)
        except BaseException:
            with self._lock:
                self._preparing_count -= 1
            raise
        result: Future[str] = Future()

        def prepared(preparation: Future[OutgoingTransfer]) -> None:
            transfer: OutgoingTransfer | None = None
            slot_reserved = True
            if not result.set_running_or_notify_cancel():
                try:
                    preparation.result().cancel()
                except BaseException:
                    pass
                finally:
                    with self._lock:
                        self._preparing_count -= 1
                return
            try:
                transfer = preparation.result()
                with self._lock:
                    if self._closed:
                        transfer.cancel()
                        raise FileTransferError("file service is closed")
                    self._outgoing[transfer.offer.transfer_id] = transfer
                    self._preparing_count -= 1
                    slot_reserved = False
                    self._send(transfer.offer_message(), Priority.NORMAL)
                    self._notify_outgoing(transfer, "offered")
                    if self._outgoing.get(transfer.offer.transfer_id) is not transfer:
                        raise FileTransferCancelled("file transfer was cancelled while preparing")
                    result.set_result(transfer.offer.transfer_id)
            except BaseException as error:
                with self._lock:
                    if slot_reserved:
                        self._preparing_count -= 1
                    if transfer is not None:
                        self._outgoing.pop(transfer.offer.transfer_id, None)
                if transfer is not None:
                    transfer.cancel()
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
        with self._lock:
            if self._closed:
                return False
        self._ensure_trusted()
        try:
            body = self._bus.decrypt_secure(message)
        except Exception as error:
            raise FileProtocolError("file message authentication failed") from error
        with self._lock:
            if self._closed:
                return False
        if not isinstance(body, bytes):
            raise FileProtocolError("file message decryption returned invalid data")
        handler(Message(message.message_type, message.metadata, body))
        return True

    def poll(self, *, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            if self._closed:
                return
            for transfer in self._outgoing.values():
                pending_messages = transfer.retry_control(now=now)
                try:
                    if transfer.accepted:
                        pending_messages.extend(transfer.retransmit_expired(now=now))
                except FileTransferError:
                    pass
                self._send_many(pending_messages)
            for transfer_id, pending in self._pending_cancels.items():
                if timestamp - pending.last_sent_at >= RETRANSMIT_TIMEOUT:
                    pending.last_sent_at = timestamp
                    self._send(
                        Message(
                            MessageType.FILE_CANCEL,
                            {"transfer_id": transfer_id, "reason": "cancelled"},
                        ),
                        Priority.INTERACTIVE,
                    )

    def cancel(self, transfer_id: str) -> None:
        with self._lock:
            transfer = self._outgoing.get(transfer_id)
            if transfer is not None:
                direction = "outgoing"
                self._outgoing.pop(transfer_id, None)
                transfer.cancel()
                self._queue_cancel_locked(transfer_id, direction)
                self._notify_outgoing(transfer, "cancelled")
            else:
                transfer = self._incoming.get(transfer_id)
                if transfer is None or not transfer.cancel():
                    return
                direction = "incoming"
                self._incoming.pop(transfer_id, None)
                self._remember_cancelled_locked(transfer.offer.transfer_id, transfer.offer)
                self._queue_cancel_locked(transfer_id, direction)
                self._notify_incoming(transfer, "cancelled")

    def close(self) -> None:
        self._timer_stop.set()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            transfers = tuple(self._outgoing.values()) + tuple(self._incoming.values())
            self._outgoing.clear()
            self._incoming.clear()
            self._completed_incoming.clear()
            self._cancelled_incoming.clear()
            self._pending_cancels.clear()
            self._terminal_status.clear()
        for transfer in transfers:
            transfer.cancel()
        if self._timer_thread is not threading.current_thread():
            self._timer_thread.join(2.0)
        if self._owns_executor:
            if threading.current_thread().name.startswith("shooklink-file"):
                threading.Thread(
                    target=self._executor.shutdown,
                    kwargs={"wait": True, "cancel_futures": True},
                    name="shooklink-file-shutdown",
                    daemon=True,
                ).start()
            else:
                self._executor.shutdown(wait=True, cancel_futures=True)

    def _timer_loop(self) -> None:
        interval = min(0.25, RETRANSMIT_TIMEOUT / 2)
        while not self._timer_stop.wait(interval):
            try:
                self.poll()
            except BaseException:
                logger.exception("file retransmission timer failed")

    def _handle_offer(self, message: Message) -> None:
        self._ensure_trusted()
        if message.body:
            raise FileProtocolError("file offer cannot contain a body")
        offer = FileOffer.from_metadata(message.metadata)
        with self._lock:
            if self._offer_is_unavailable_locked(offer):
                return
        try:
            accepted = bool(self._accept_offer(offer))
        except BaseException:
            logger.exception("incoming file policy failed")
            accepted = False
        with self._lock:
            if self._offer_is_unavailable_locked(offer):
                return
            if not accepted:
                self._send_cancel(offer.transfer_id, "rejected")
                return
            try:
                transfer = IncomingTransfer.create(offer, self.download_dir)
            except OSError:
                self._send_cancel(offer.transfer_id, "unavailable")
                return
            self._incoming[offer.transfer_id] = transfer
            self._send_accept(offer.transfer_id)
        self._notify_incoming(transfer, "receiving")

    def _offer_is_unavailable_locked(self, offer: FileOffer) -> bool:
        if self._closed:
            return True
        cancelled = self._cancelled_incoming.get(offer.transfer_id)
        if offer.transfer_id in self._cancelled_incoming:
            self._cancelled_incoming.move_to_end(offer.transfer_id)
            reason = "cancelled" if cancelled is None or cancelled == offer else "duplicate"
            self._send_cancel(offer.transfer_id, reason)
            return True
        terminal = self._terminal_status.get(offer.transfer_id)
        if terminal is not None and terminal.status == "cancelled":
            reason = (
                "cancelled"
                if terminal.offer is None or terminal.offer == offer
                else "duplicate"
            )
            self._send_cancel(offer.transfer_id, reason)
            return True
        completed = self._completed_incoming.get(offer.transfer_id)
        if completed is not None:
            self._completed_incoming.move_to_end(offer.transfer_id)
            if completed[0] == offer:
                self._send_finish_status(offer.transfer_id, completed[1])
            else:
                self._send_cancel(offer.transfer_id, "duplicate")
            return True
        if terminal is not None and terminal.status in {"ok", "error"}:
            if terminal.offer == offer:
                self._send_finish_status(offer.transfer_id, terminal.status)
            else:
                self._send_cancel(offer.transfer_id, "duplicate")
            return True
        existing = self._incoming.get(offer.transfer_id)
        if existing is not None:
            if existing.offer == offer:
                self._send_accept(offer.transfer_id)
            else:
                self._send_cancel(offer.transfer_id, "duplicate")
            return True
        if offer.transfer_id in self._outgoing:
            self._send_cancel(offer.transfer_id, "duplicate")
            return True
        pending_incoming_cancels = sum(
            pending.direction == "incoming"
            for pending in self._pending_cancels.values()
        )
        if (
            len(self._incoming) + pending_incoming_cancels
            >= self._max_incoming_transfers
        ):
            self._send_cancel(offer.transfer_id, "busy")
            return True
        if (
            len(self._terminal_status) + len(self._incoming)
            >= self._max_terminal_transfers
        ):
            self._send_cancel(offer.transfer_id, "capacity")
            return True
        incoming_bytes = sum(item.offer.size for item in self._incoming.values())
        if (
            incoming_bytes + offer.size > self._max_incoming_bytes
            or incoming_bytes + offer.size > _available_disk_bytes(self.download_dir)
        ):
            self._send_cancel(offer.transfer_id, "capacity")
            return True
        return False

    def _handle_accept(self, message: Message) -> None:
        if message.body or set(message.metadata) != {"transfer_id"}:
            raise FileProtocolError("file accept fields are invalid")
        with self._lock:
            transfer_id = _metadata_transfer_id(message.metadata)
            transfer = self._outgoing.get(transfer_id)
            if transfer is None:
                return
            try:
                transfer.accept()
                messages = transfer.next_messages()
            except (OSError, FileTransferError):
                self._outgoing.pop(transfer_id, None)
                transfer.cancel()
                self._queue_cancel_locked(transfer_id, "outgoing")
                self._notify_outgoing(transfer, "failed")
                return
            self._send_many(messages)
            self._notify_outgoing(transfer, "sending")

    def _handle_chunk(self, message: Message) -> None:
        if set(message.metadata) != {"transfer_id", "index"}:
            raise FileProtocolError("file chunk fields are invalid")
        transfer_id = _metadata_transfer_id(message.metadata)
        with self._lock:
            if (
                transfer_id in self._cancelled_incoming
                or (
                    (terminal := self._terminal_status.get(transfer_id)) is not None
                    and terminal.status == "cancelled"
                )
            ):
                self._send_cancel(transfer_id, "ack")
                return
            completed = self._completed_incoming.get(transfer_id)
            if completed is not None:
                self._completed_incoming.move_to_end(transfer_id)
                self._send_finish_status(transfer_id, completed[1])
                return
            terminal = self._terminal_status.get(transfer_id)
            if terminal is not None and terminal.status in {"ok", "error"}:
                self._send_finish_status(transfer_id, terminal.status)
                return
            transfer = self._incoming.get(transfer_id)
            if transfer is None:
                return
            if transfer_id not in self._finalizing:
                try:
                    transfer.write_chunk(message.metadata["index"], message.body)
                except OSError:
                    self._incoming.pop(transfer_id, None)
                    transfer.cancel()
                    self._remember_completed_locked(transfer, None, "error")
                    self._send_cancel(transfer_id, "storage")
                    self._notify_incoming(transfer, "failed")
                    return
            self._send(
                Message(MessageType.FILE_ACK, transfer.ack_metadata()),
                Priority.NORMAL,
            )
            self._notify_incoming(transfer, "receiving")

    def _handle_ack(self, message: Message) -> None:
        if message.body or set(message.metadata) != {
            "transfer_id",
            "base",
            "span",
            "bitmap",
        }:
            raise FileProtocolError("file ACK fields are invalid")
        with self._lock:
            transfer_id = _metadata_transfer_id(message.metadata)
            transfer = self._outgoing.get(transfer_id)
            if transfer is None:
                return
            try:
                messages = transfer.acknowledge(message.metadata)
            except (OSError, FileTransferError):
                self._outgoing.pop(transfer_id, None)
                transfer.cancel()
                self._queue_cancel_locked(transfer_id, "outgoing")
                self._notify_outgoing(transfer, "failed")
                return
            self._send_many(messages)
            if self._outgoing.get(transfer_id) is transfer:
                self._notify_outgoing(transfer, "sending")

    def _handle_finish(self, message: Message) -> None:
        if message.body:
            raise FileProtocolError("file finish cannot contain a body")
        transfer_id = _metadata_transfer_id(message.metadata)
        status = message.metadata.get("status")
        if status is not None:
            if set(message.metadata) != {"transfer_id", "status"}:
                raise FileProtocolError("file finish status fields are invalid")
            if status not in {"ok", "error"}:
                raise FileProtocolError("file finish status is invalid")
            with self._lock:
                self._pending_cancels.pop(transfer_id, None)
                transfer = self._outgoing.pop(transfer_id, None)
                if transfer is not None:
                    transfer.mark_completed()
            if transfer is not None:
                self._notify_outgoing(transfer, "complete" if status == "ok" else "failed")
            return

        with self._lock:
            if set(message.metadata) != {"transfer_id", "sha256"}:
                raise FileProtocolError("file finish fields are invalid")
            if (
                transfer_id in self._cancelled_incoming
                or (
                    (terminal := self._terminal_status.get(transfer_id)) is not None
                    and terminal.status == "cancelled"
                )
            ):
                self._send_cancel(transfer_id, "ack")
                return
            if transfer_id in self._completed_incoming:
                self._completed_incoming.move_to_end(transfer_id)
                self._send_finish_status(
                    transfer_id,
                    self._completed_incoming[transfer_id][1],
                )
                return
            terminal = self._terminal_status.get(transfer_id)
            if terminal is not None and terminal.status in {"ok", "error"}:
                self._send_finish_status(transfer_id, terminal.status)
                return
            transfer = self._incoming.get(transfer_id)
            if transfer is None:
                return
            if transfer_id in self._finalizing:
                return
            if message.metadata["sha256"] != transfer.offer.sha256:
                self._incoming.pop(transfer_id, None)
                transfer.cancel()
                self._remember_completed_locked(transfer, None, "error")
                self._send_finish_status(transfer_id, "error")
                self._notify_incoming(transfer, "failed")
                return
            future = self._executor.submit(transfer.finalize)
            self._finalizing[transfer_id] = future

        def finalized(completion: Future[Path]) -> None:
            notify: tuple[IncomingTransfer, str] | None = None
            try:
                path = completion.result()
            except FileTransferCancelled:
                with self._lock:
                    self._finalizing.pop(transfer_id, None)
                    self._incoming.pop(transfer_id, None)
            except BaseException:
                transfer.cancel()
                with self._lock:
                    self._finalizing.pop(transfer_id, None)
                    active = self._incoming.pop(transfer_id, None) is transfer
                    if active and not self._closed:
                        self._remember_completed_locked(transfer, None, "error")
                        self._send_finish_status(transfer_id, "error")
                        notify = (transfer, "failed")
            else:
                with self._lock:
                    self._finalizing.pop(transfer_id, None)
                    active = self._incoming.pop(transfer_id, None) is transfer
                    if active and not self._closed:
                        self._remember_completed_locked(transfer, path, "ok")
                        self._send_finish_status(transfer_id, "ok")
                        notify = (transfer, "complete")
            if notify is not None:
                if notify[1] == "complete":
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
                else:
                    self._notify_incoming(*notify)

        future.add_done_callback(finalized)

    def _handle_cancel(self, message: Message) -> None:
        if message.body or not set(message.metadata).issubset({"transfer_id", "reason"}):
            raise FileProtocolError("file cancel fields are invalid")
        transfer_id = _metadata_transfer_id(message.metadata)
        reason = message.metadata.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise FileProtocolError("file cancel reason is invalid")
        with self._lock:
            if self._closed:
                return
            if reason == "ack":
                self._pending_cancels.pop(transfer_id, None)
                return
            self._pending_cancels.pop(transfer_id, None)
            transfer = self._outgoing.pop(transfer_id, None)
            if transfer is not None:
                self._notify_outgoing(transfer, "cancelled")
                transfer.cancel()
            else:
                transfer = self._incoming.get(transfer_id)
                if transfer is not None:
                    if transfer.cancel():
                        self._incoming.pop(transfer_id, None)
                        self._remember_cancelled_locked(transfer.offer.transfer_id, transfer.offer)
                        self._notify_incoming(transfer, "cancelled")
                    elif transfer.state == "completed":
                        self._send_finish_status(transfer_id, "ok")
                        return
                elif (
                    (terminal := self._terminal_status.get(transfer_id)) is not None
                    and terminal.status in {"ok", "error"}
                ):
                    self._send_finish_status(
                        transfer_id,
                        terminal.status,
                    )
                    return
                else:
                    self._remember_cancelled_locked(transfer_id, None)
            self._send_cancel(transfer_id, "ack")

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

    def _send_accept(self, transfer_id: str) -> None:
        self._send(
            Message(MessageType.FILE_ACCEPT, {"transfer_id": transfer_id}),
            Priority.NORMAL,
        )

    def _send_cancel(self, transfer_id: str, reason: str) -> None:
        self._send(
            Message(
                MessageType.FILE_CANCEL,
                {"transfer_id": transfer_id, "reason": reason},
            ),
            Priority.NORMAL,
        )

    def _queue_cancel_locked(self, transfer_id: str, direction: str) -> None:
        timestamp = time.monotonic()
        self._pending_cancels[transfer_id] = _PendingCancel(timestamp, direction)
        self._send(
            Message(
                MessageType.FILE_CANCEL,
                {"transfer_id": transfer_id, "reason": "cancelled"},
            ),
            Priority.INTERACTIVE,
        )

    def _send_finish_status(self, transfer_id: str, status: str) -> None:
        self._send(
            Message(
                MessageType.FILE_FINISH,
                {"transfer_id": transfer_id, "status": status},
            ),
            Priority.NORMAL,
        )

    def _remember_completed_locked(
        self,
        transfer: IncomingTransfer,
        path: Path | None,
        status: str,
    ) -> None:
        self._completed_incoming[transfer.offer.transfer_id] = (
            transfer.offer,
            status,
            path,
        )
        self._completed_incoming.move_to_end(transfer.offer.transfer_id)
        self._terminal_status[transfer.offer.transfer_id] = _TerminalTransfer(
            status,
            transfer.offer,
        )
        while len(self._completed_incoming) > MAX_COMPLETED_TRANSFERS:
            self._completed_incoming.popitem(last=False)

    def _remember_cancelled_locked(
        self,
        transfer_id: str,
        offer: FileOffer | None,
    ) -> None:
        self._cancelled_incoming[transfer_id] = offer
        self._cancelled_incoming.move_to_end(transfer_id)
        if (
            transfer_id in self._terminal_status
            or len(self._terminal_status) + len(self._incoming)
            < self._max_terminal_transfers
        ):
            self._terminal_status[transfer_id] = _TerminalTransfer(
                "cancelled",
                offer,
            )
        while len(self._cancelled_incoming) > MAX_COMPLETED_TRANSFERS:
            self._cancelled_incoming.popitem(last=False)
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
            try:
                listener(progress)
            except Exception:
                logger.exception("file progress listener failed")


def _available_disk_bytes(path: Path) -> int:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        return shutil.disk_usage(candidate).free
    except OSError:
        return 0


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
