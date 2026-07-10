"""Persistent ShookLink application settings."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

VALID_PEER_SIDES = frozenset({"right", "left", "top", "bottom"})
MIN_BAUD_RATE = 300
MAX_BAUD_RATE = 4_000_000


def _default_download_dir() -> str:
    return str(Path.home() / "Downloads" / "ShookLink")


@dataclass(frozen=True, slots=True)
class AppSettings:
    last_port: str = ""
    baud_rate: int = 115_200
    peer_side: str = "right"
    auto_edge_enabled: bool = False
    download_dir: str = field(default_factory=_default_download_dir)
    allow_remote_shell: bool = False
    allow_input: bool = False


class SettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> AppSettings:
        defaults = AppSettings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return defaults

        if not isinstance(raw, dict):
            return defaults

        last_port = raw.get("last_port")
        baud_rate = raw.get("baud_rate")
        peer_side = raw.get("peer_side")
        auto_edge_enabled = raw.get("auto_edge_enabled")
        download_dir = raw.get("download_dir")

        return AppSettings(
            last_port=last_port if isinstance(last_port, str) else defaults.last_port,
            baud_rate=(
                baud_rate
                if type(baud_rate) is int
                and MIN_BAUD_RATE <= baud_rate <= MAX_BAUD_RATE
                else defaults.baud_rate
            ),
            peer_side=(
                peer_side
                if isinstance(peer_side, str) and peer_side in VALID_PEER_SIDES
                else defaults.peer_side
            ),
            auto_edge_enabled=(
                auto_edge_enabled
                if isinstance(auto_edge_enabled, bool)
                else defaults.auto_edge_enabled
            ),
            download_dir=(
                download_dir
                if isinstance(download_dir, str) and download_dir
                else defaults.download_dir
            ),
            allow_remote_shell=False,
            allow_input=False,
        )

    def save(self, settings: AppSettings) -> None:
        payload = asdict(settings)
        payload["allow_remote_shell"] = False
        payload["allow_input"] = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, ensure_ascii=False, sort_keys=True)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, self.path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
