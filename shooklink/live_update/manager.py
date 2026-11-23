"""Receiver-side journal and staging. Only the live GUI owns the serial link."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .release import Release, atomic_json, hex_id, read_bounded, stage_archive

SWITCH_EXIT = 75
MANAGED_ENV = "SHOOKLINK_MANAGED_HOME"
PENDING = {"staging", "prepared", "restarting", "trial", "healthy"}


@dataclass(frozen=True)
class State:
    active: str = ""  # Empty selects the stable bootstrap directory.
    highest: int = 0
    phase: str = "idle"
    job: str = ""
    target: str = ""
    peer: str = ""
    old_instance: str = ""
    reason: str = ""

    @classmethod
    def load(cls, home: Path) -> State:
        path = home / "state.json"
        if not path.exists():
            return cls()
        try:
            state = cls(**json.loads(read_bounded(path, 4096)))
        except (ValueError, TypeError) as error:
            raise ValueError("invalid local update journal") from error
        if (state.active != "" and not hex_id(state.active, 32)
                or state.job != "" and not hex_id(state.job, 32)
                or not isinstance(state.phase, str)
                or state.phase not in PENDING | {"idle", "installed", "failed", "rolled_back"}
                or type(state.highest) is not int or state.highest < 0
                or (state.phase in PENDING and not state.job)):
            raise ValueError("invalid local update journal")
        return state

    def save(self, home: Path) -> State:
        atomic_json(home / "state.json", asdict(self))
        return self


class ManagedReceiver:
    def __init__(self, home: Path):
        self.home = home
        self.publisher = read_bounded(home / "publisher", 100).decode("ascii").strip()
        if not hex_id(self.publisher, 64):
            raise ValueError("invalid local publisher enrollment")
        self._state = State.load(home)
        self._lock = threading.RLock()
        self._observed = False
        self._worker: threading.Thread | None = None

    def _change(self, **changes):
        self._state = replace(self._state, **changes).save(self.home)

    def report(self) -> dict:
        with self._lock:
            s = self._state
            return dict(managed=True, publisher=self.publisher, highest=s.highest,
                        phase=s.phase, job=s.job, target=s.target, reason=s.reason)

    def apply(self, job: str, release: Release, path: Path, peer: str, instance: str) -> None:
        if not hex_id(job, 32):
            raise ValueError("invalid transaction")
        release.verify(self.publisher)
        with self._lock:
            s = self._state
            if job == s.job and peer == s.peer and release.revision == s.target:
                return
            if s.phase in PENDING:
                raise ValueError("an update is already in progress")
            if release.sequence <= s.highest:
                raise ValueError("release sequence was already attempted or superseded")
            self._change(phase="staging", job=job, target=release.revision, peer=peer,
                         old_instance=instance, reason="")
            self._worker = threading.Thread(
                target=self._stage, args=(path, release, job), name="shooklink-stage"
            )
            self._worker.start()

    def _stage(self, path: Path, release: Release, job: str) -> None:
        try:
            stage_archive(path, release, self.publisher, self.home / "slots" / job)
            with self._lock:
                self._change(phase="prepared", highest=release.sequence)
        except Exception as error:
            with self._lock:
                self._change(phase="failed", reason=str(error)[:240])

    def confirm(self, job: str, peer: str, instance: str, actual_revision: str) -> None:
        with self._lock:
            s = self._state
            if (s.phase not in {"trial", "healthy", "installed"} or job != s.job
                    or peer != s.peer or actual_revision != s.target or instance == s.old_instance):
                raise ValueError("health confirmation does not match this trial")
            self._observed = True

    def tick(self, ready_peer: str | None, idle: bool) -> bool:
        """Called from the GUI event loop. True requests an orderly switch exit."""
        with self._lock:
            s = self._state
            if s.phase == "prepared":
                if not idle or ready_peer != s.peer:
                    self._change(phase="failed", reason="peer disconnected or another session is busy")
                else:
                    self._change(phase="restarting")
                    return True
            elif s.phase == "trial" and self._observed and ready_peer == s.peer:
                # The supervisor owns the last-good pointer and checks that we are still alive.
                self._change(phase="healthy")
            elif s.phase == "healthy":
                self._state = State.load(self.home)
            return False

    def close(self):
        with self._lock:
            worker = self._worker
        if worker:
            worker.join()
