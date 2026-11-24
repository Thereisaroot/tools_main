"""An offline publisher signs a specific ShookLink archive and release sequence."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)

FEATURE = "live_update_v1"
MAX_ZIP = 16 * 1024 * 1024
MAX_EXPANDED = 32 * 1024 * 1024
TERMINAL_ASSET_DIR = "shooklink/ui/terminal_assets"
TERMINAL_ASSETS = frozenset(f"{TERMINAL_ASSET_DIR}/{name}" for name in (
    "index.html", "terminal.js", "terminal.css", "xterm.js", "xterm.css",
    "addon-fit.js", "addon-search.js", "LICENSE-xterm.txt", "LICENSE-fit.txt", "LICENSE-search.txt",
))
_ROOT_FILES = {"README.md", "requirements.txt", "pyproject.toml",
               "run_serial_text_chat.bat", "run_serial_text_chat.sh"}
_REQUIRED = {"shooklink/__init__.py", "shooklink/__main__.py", "shooklink/app.py",
             "shooklink/live_update/supervisor.py", "run_serial_text_chat.bat",
             "requirements.txt"}


def hex_id(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and bool(re.fullmatch("[0-9a-f]+", value))


def read_bounded(path: Path, maximum: int) -> bytes:
    if not path.is_file():
        raise ValueError(f"expected a regular file: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"{path.name} exceeds the size limit")
    return data


def atomic_json(path: Path, document: dict) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        # A Windows reader may briefly hold the journal without FILE_SHARE_DELETE.
        deadline = time.monotonic() + 1
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if os.name != "nt" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
    finally:
        temporary.unlink(missing_ok=True)


class Signer(Protocol):
    def sign(self, data: bytes, /) -> bytes: ...


def generate_key(path: Path) -> str:
    key = Ed25519PrivateKey.generate()
    # O_EXCL prevents replacing a publisher key which is already enrolled.
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_EXCL | os.O_CREAT, 0o600), "wb") as stream:
        stream.write(key.private_bytes_raw())
    return key.public_key().public_bytes_raw().hex()


@dataclass(frozen=True)
class Release:
    revision: str
    sha256: str
    sequence: int
    signature: str

    def __post_init__(self):
        if not (hex_id(self.revision, 40) or hex_id(self.revision, 64)):
            raise ValueError("invalid full release revision")
        if not hex_id(self.sha256, 64) or not hex_id(self.signature, 128):
            raise ValueError("invalid release digest or signature")
        if type(self.sequence) is not int or not 1 <= self.sequence < 2**53:
            raise ValueError("release sequence must be a positive integer below 2^53")

    @classmethod
    def parse(cls, value: object) -> Release:
        if not isinstance(value, dict) or set(value) != {"revision", "sha256", "sequence", "signature"}:
            raise ValueError("invalid ShookLink release manifest")
        return cls(**value)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def name(self) -> str:
        return f"ShookLink-{self.revision[:12]}"

    def _signed_bytes(self) -> bytes:
        return f"ShookLink release v1\0{self.sequence}\n{self.revision}\n{self.sha256}".encode("ascii")

    @classmethod
    def sign(cls, path: Path, sequence: int, key: Signer) -> Release:
        data = read_bounded(path, MAX_ZIP)
        with ZipFile(io.BytesIO(data)) as archive:
            revision = archive.comment.decode("ascii")
        draft = cls(revision, hashlib.sha256(data).hexdigest(), sequence, "0" * 128)
        return cls(revision, draft.sha256, sequence, key.sign(draft._signed_bytes()).hex())

    def verify(self, public_key: str) -> None:
        if not hex_id(public_key, 64):
            raise ValueError("invalid enrolled publisher public key")
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(
                bytes.fromhex(self.signature), self._signed_bytes()
            )
        except InvalidSignature as error:
            raise ValueError("release signature does not match the enrolled publisher") from error


def _runtime_member(name: str, directory: bool) -> bool:
    if not directory and (name in _ROOT_FILES or name in TERMINAL_ASSETS):
        return True
    parts = name.split("/")
    if parts[0] != "shooklink" or (not directory and len(parts) == 1):
        return False
    if not directory:
        if not parts[-1].endswith(".py"):
            return False
        parts[-1] = parts[-1][:-3]
    return all(re.fullmatch("[a-z_][a-z_0-9]*", part)
               and not re.fullmatch("con|prn|aux|nul|com[1-9]|lpt[1-9]", part)
               for part in parts)


def stage_archive(path: Path, release: Release, public_key: str, destination: Path) -> None:
    release.verify(public_key)
    data = read_bounded(path, MAX_ZIP)
    if hashlib.sha256(data).hexdigest() != release.sha256:
        raise ValueError("release ZIP digest mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("release slot already exists")
    with ZipFile(io.BytesIO(data)) as archive, tempfile.TemporaryDirectory(dir=destination.parent) as work:
        entries = archive.infolist()
        if (archive.comment != release.revision.encode() or len(entries) > 512
                or sum(m.file_size for m in entries) > MAX_EXPANDED):
            raise ValueError("invalid ShookLink ZIP revision or size")
        root = Path(work) / "release"
        root.mkdir()
        prefix = release.name + "/"
        seen = set()
        for entry in entries:
            if entry.filename == prefix and entry.is_dir():
                continue
            if not entry.filename.startswith(prefix) or len(entry.filename) > 240:
                raise ValueError("unexpected ShookLink ZIP root")
            name = entry.filename[len(prefix):].removesuffix("/")
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if (not _runtime_member(name, entry.is_dir()) or name in seen
                    or mode not in {0, stat.S_IFDIR if entry.is_dir() else stat.S_IFREG}
                    or entry.compress_type not in {ZIP_STORED, ZIP_DEFLATED}):
                raise ValueError(f"unexpected ShookLink ZIP entry: {name!r}")
            seen.add(name)
            output = root / name
            if entry.is_dir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as stream:
                    stream.write(archive.read(entry))
                if name == "run_serial_text_chat.sh":
                    output.chmod(0o755)
        if not _REQUIRED.issubset(seen):
            raise ValueError("archive is missing the managed ShookLink runtime")
        if seen & TERMINAL_ASSETS and not TERMINAL_ASSETS.issubset(seen):
            raise ValueError("archive is missing offline terminal resources")
        root.rename(destination)
