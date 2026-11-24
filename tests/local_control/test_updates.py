import subprocess
from zipfile import ZipFile

import pytest

from shooklink.core import CoreState
from shooklink.local_control import cli
from shooklink.local_control.server import LocalControlServer
from tests.integration.test_peer_pair import (
    approve_pair,
    build_core,
    endpoint_pair,
    wait_for,
)


def commit(root, message):
    subprocess.run(
        [
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgSign=false", "commit", "-qm", message,
        ],
        cwd=root,
        check=True,
    )


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "source checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in (
        "README.md", "requirements.txt", "pyproject.toml",
        "run_serial_text_chat.bat", "run_serial_text_chat.sh",
        "shooklink/__init__.py", "shooklink/__main__.py", "shooklink/app.py",
        "shooklink/chat/service.py", "shooklink/private-settings.json",
        "docs/working-notes.md",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"committed\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    commit(root, "release")
    return root


def test_pack_update_uses_committed_runtime_only_and_needs_no_gui(checkout, tmp_path):
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=checkout, check=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    (checkout / "shooklink/app.py").write_bytes(b"unreleased local edits")
    (checkout / "shooklink/local_only.py").write_bytes(b"untracked")
    output_dir = tmp_path / "packages"

    assert cli.main(
        ["pack-update", "--repo", str(checkout), "--output-dir", str(output_dir)],
        client_factory=lambda _: pytest.fail("packaging must work without a GUI"),
    ) == 0

    release = f"ShookLink-{revision[:12]}"
    with ZipFile(output_dir / f"{release}.zip") as archive:
        assert archive.comment.decode() == revision
        assert archive.read(f"{release}/shooklink/app.py") == b"committed\n"
        assert {item.filename for item in archive.infolist() if not item.is_dir()} == {
            f"{release}/{name}" for name in (
                "shooklink/__init__.py", "shooklink/__main__.py", "shooklink/app.py",
                "shooklink/chat/service.py", "README.md", "pyproject.toml", "requirements.txt",
                "run_serial_text_chat.sh", "run_serial_text_chat.bat",
            )
        }


def test_offline_terminal_packaging_includes_exact_resources_and_not_extra_scripts(checkout, tmp_path):
    from shooklink.live_update.release import TERMINAL_ASSETS, TERMINAL_ASSET_DIR
    from shooklink.local_control.updates import build_update_archive

    directory = checkout / TERMINAL_ASSET_DIR
    directory.mkdir(parents=True)
    (directory / "index.html").write_text("first")
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    commit(checkout, "incomplete renderer")
    with pytest.raises(ValueError, match="incomplete offline"):
        build_update_archive(checkout, tmp_path / "packages")
    for name in TERMINAL_ASSETS:
        (checkout / name).write_bytes(b"test resource")
    (directory / "not-shipped.js").write_text("extra")
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    commit(checkout, "complete renderer")
    path = build_update_archive(checkout, tmp_path / "packages")
    with ZipFile(path) as archive:
        resources = {n.split("/", 1)[1] for n in archive.namelist() if TERMINAL_ASSET_DIR in n and not n.endswith("/")}
        assert resources == TERMINAL_ASSETS


def test_send_update_reaches_a_peer_via_its_existing_file_service(checkout, tmp_path):
    endpoint = tmp_path / "control.json"
    sender, _, _ = build_core(tmp_path, "sender")
    receiver, _, _ = build_core(tmp_path, "receiver")
    server = LocalControlServer(sender, endpoint)
    try:
        left, right = endpoint_pair()
        sender.connect_endpoint(left)
        receiver.connect_endpoint(right)
        assert wait_for(
            lambda: sender.snapshot.state is receiver.snapshot.state is CoreState.UNTRUSTED
        )
        server.start()
        args = ["--endpoint", str(endpoint), "send-update", "--repo", str(checkout)]
        assert cli.main(args) == 1
        assert not list(receiver.files.download_dir.glob("*.zip"))
        approve_pair(sender, receiver)
        assert cli.main(args) == 0

        package, = (checkout / "dist").glob("*.zip")
        received, = receiver.files.download_dir.iterdir()
        assert received.name == package.name
        assert received.read_bytes() == package.read_bytes()
    finally:
        server.close()
        sender.close()
        receiver.close()


def test_packaging_failure_does_not_publish_or_send_a_partial_update(
    checkout, tmp_path, capsys
):
    subprocess.run(
        ["git", "update-index", "--force-remove", "run_serial_text_chat.bat"],
        cwd=checkout,
        check=True,
    )
    commit(checkout, "remove launcher")
    output_dir = tmp_path / "packages"
    assert cli.main(
        ["send-update", "--repo", str(checkout), "--output-dir", str(output_dir)],
        client_factory=lambda _: pytest.fail("invalid package must not be sent"),
    ) == 1
    assert "run_serial_text_chat.bat" in capsys.readouterr().err
    assert not list(output_dir.glob("*"))
