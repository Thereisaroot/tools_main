"""Local consent for an update signer advertised in authenticated TRUST features."""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import asdict, dataclass
from pathlib import Path

from shooklink.protocol.crypto import Identity, IdentityStore
from .release import FEATURE, atomic_json, hex_id, read_bounded

OFFER_PREFIX = "update_publisher_v1:"


def local_signer(data_dir: Path) -> Identity:
    # Never use the serial identity or put signing material in a release ZIP.
    return IdentityStore(data_dir / "release-signing.key").load_or_create()


def offered_publisher(features: Collection[str]) -> str | None:
    offers = [f[len(OFFER_PREFIX):] for f in features if f.startswith(OFFER_PREFIX)]
    if FEATURE in features and len(offers) == 1 and hex_id(offers[0], 64):
        return offers[0]
    return None


@dataclass(frozen=True, slots=True)
class PublisherPin:
    publisher: str
    peer: str | None = None  # None is the pre-existing, explicitly pinned offline signer.

    def __post_init__(self):
        if not hex_id(self.publisher, 64):
            raise ValueError("invalid publisher public key")
        if self.peer is not None and (
            not isinstance(self.peer, str) or not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", self.peer)
        ):
            raise ValueError("invalid peer in publisher enrollment")

    def allows(self, fingerprint: str) -> bool:
        return self.peer is None or self.peer == fingerprint

    @classmethod
    def load(cls, home: Path) -> PublisherPin | None:
        path = home / "publisher"
        if not path.exists():
            return None
        try:
            raw = read_bounded(path, 512).decode("ascii").strip()
            if hex_id(raw, 64):
                return cls(raw)
            fields = json.loads(raw)
            if not isinstance(fields, dict) or set(fields) != {"peer", "publisher"} or not fields["peer"]:
                raise ValueError("missing pinned peer")
            return cls(**fields)
        except (ValueError, TypeError) as error:
            raise ValueError("invalid local publisher enrollment") from error

    def persist(self, home: Path) -> PublisherPin:
        # Writers own the supervisor/child lock; never rotate or widen an existing pin.
        existing = self.load(home)
        if existing is not None:
            if existing.publisher == self.publisher and (self.peer is None or self == existing):
                return existing
            raise ValueError("another publisher or peer is already enrolled locally")
        if self.peer is None:
            with (home / "publisher").open("x", encoding="ascii") as handle:
                handle.write(self.publisher + "\n")
        else:
            atomic_json(home / "publisher", asdict(self))
        return self
