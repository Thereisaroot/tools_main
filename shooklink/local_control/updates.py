"""Package a committed ShookLink release for the existing serial file service."""

from __future__ import annotations

import filecmp
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

from shooklink.live_update.release import TERMINAL_ASSET_DIR, TERMINAL_ASSETS
_RUNTIME_FILES = (
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "run_serial_text_chat.bat",
    "run_serial_text_chat.sh",
)


def _git(directory: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(directory), *args], capture_output=True, check=True
        ).stdout
    except FileNotFoundError as error:
        raise RuntimeError(
            "building an update requires Git on the sending computer"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot package this Git checkout: {detail}") from error


def build_update_archive(checkout: Path, output_dir: Path | None = None) -> Path:
    toplevel = _git(checkout.expanduser(), "rev-parse", "--show-toplevel")
    root = Path(os.fsdecode(toplevel.rstrip(b"\r\n")))
    revision = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    tracked = set(_git(root, "ls-tree", "-r", "--name-only", revision, "--", TERMINAL_ASSET_DIR)
                  .decode().splitlines())
    assets = TERMINAL_ASSETS & tracked
    if assets and assets != TERMINAL_ASSETS:
        raise ValueError("incomplete offline terminal assets in release")
    release = f"ShookLink-{revision[:12]}"
    directory = (output_dir or root / "dist").expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{release}.zip"

    # Keep a partial rebuild away from any transfer still reading the published ZIP.
    with tempfile.TemporaryDirectory(prefix=".shooklink-update-", dir=directory) as staging:
        staged = Path(staging) / destination.name
        _git(
            root,
            "-c", "core.autocrlf=false",
            "archive",
            "--format=zip",
            f"--prefix={release}/",
            f"--output={staged}",
            revision,
            "--",
            ":(top,glob)shooklink/**/*.py",
            *_RUNTIME_FILES,
            *sorted(assets),
        )
        with ZipFile(staged) as archive:
            names = set(archive.namelist())
            entry_points = ("shooklink/__init__.py", "shooklink/__main__.py", "shooklink/app.py")
            for name in (*_RUNTIME_FILES, *entry_points):
                if f"{release}/{name}" not in names:
                    raise ValueError(f"ShookLink update is missing {name}")
            if any(stat.S_ISLNK(m.external_attr >> 16) for m in archive.infolist()):
                raise ValueError("ShookLink update contains a symbolic link")
        if destination.is_file() and filecmp.cmp(staged, destination, shallow=False):
            return destination
        staged.replace(destination)
    return destination
