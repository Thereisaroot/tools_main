"""Foreground launcher. A child exits and releases its COM handle before the next starts."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from .manager import MANAGED_ENV, PENDING, SWITCH_EXIT, State
from .release import hex_id, read_bounded


def _stop(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def _wait_for_gui(child, home: Path, handoff_timeout: float):
    deadline = None
    while True:
        try:
            return child.wait(timeout=min(1.0, handoff_timeout))
        except subprocess.TimeoutExpired:
            if State.load(home).phase != "restarting":
                continue
            if deadline is None:
                deadline = time.monotonic() + handoff_timeout
            elif time.monotonic() >= deadline:
                _stop(child)
                return None


def supervise(home: Path, bootstrap: Path, launch, *, trial_timeout=60.0,
              handoff_timeout=15.0, poll=0.15) -> int:
    s = State.load(home)
    if s.phase in PENDING:
        s = replace(s, phase="rolled_back", reason="previous update was interrupted").save(home)
    trial = ""
    while True:
        slot = trial or s.active
        folder = home / "slots" / slot if slot else bootstrap
        try:
            child = launch(folder)
        except OSError as error:
            if not trial:
                raise
            s = replace(s, phase="rolled_back", reason=f"could not start candidate: {error}").save(home)
            trial = ""
            continue
        try:
            if trial:
                deadline = time.monotonic() + trial_timeout
                while child.poll() is None and time.monotonic() < deadline:
                    s = State.load(home)
                    if s.phase == "healthy" and s.job == trial:
                        s = replace(s, phase="installed", active=trial).save(home)
                        break
                    time.sleep(poll)
                if s.active != trial:
                    _stop(child)
                    s = replace(State.load(home), phase="rolled_back",
                                reason="candidate exited or health check timed out").save(home)
                    trial = ""
                    continue
                trial = ""
            code = _wait_for_gui(child, home, handoff_timeout)
        finally:
            _stop(child)
        s = State.load(home)
        if code is None:
            s = replace(s, phase="rolled_back", reason="old GUI did not finish shutdown").save(home)
            continue
        if code != SWITCH_EXIT or s.phase != "restarting":
            if s.phase in PENDING:
                replace(s, phase="rolled_back", reason="GUI exited before handoff").save(home)
            return code
        trial = s.job
        s = replace(s, phase="trial").save(home)


def run_launcher(publisher: str | None, debug: bool) -> int:
    from PySide6.QtCore import QLockFile
    from shooklink.local_control.client import LocalControlClient, LocalControlError, application_data_dir

    data = application_data_dir()
    endpoint = data / "control.json"
    home = data / "managed-updates"
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = QLockFile(str(home / "supervisor.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        raise ValueError("a managed ShookLink launcher is already running")
    try:
        try:
            LocalControlClient(endpoint, timeout=1).request("status")
        except LocalControlError:
            pass
        else:
            raise ValueError("close the existing ShookLink GUI before starting the managed launcher")
        # An orphaned child may still own COM even if its loopback endpoint is down.
        owner = QLockFile(str(home / "child.lock"))
        owner.setStaleLockTime(0)
        if not owner.tryLock(0):
            raise ValueError("a managed ShookLink GUI still owns the installation")
        owner.unlock()
        pin = home / "publisher"
        if publisher is not None:
            if not hex_id(publisher, 64):
                raise ValueError("--publisher expects the separate release public key (64 hex digits)")
            if pin.exists():
                if read_bounded(pin, 100).decode().strip() != publisher:
                    raise ValueError("another publisher is already enrolled locally")
            else:
                with pin.open("x", encoding="ascii") as handle:
                    handle.write(publisher + "\n")
        if not pin.exists():
            raise ValueError("first managed launch requires --publisher PUBLIC_KEY")
        if not hex_id(read_bounded(pin, 100).decode().strip(), 64):
            raise ValueError("invalid publisher in the local enrollment file")
        env = dict(os.environ, **{MANAGED_ENV: str(home)})
        args = [sys.executable, "-E", "-m", "shooklink"]
        if debug:
            args.append("--debug")
        return supervise(home, Path(__file__).resolve().parents[2],
                         lambda folder: subprocess.Popen(args, cwd=folder, env=env))
    finally:
        lock.unlock()
