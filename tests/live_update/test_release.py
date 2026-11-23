from zipfile import ZipFile

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from shooklink.live_update.release import Release, stage_archive
from shooklink.live_update.manager import ManagedReceiver, State


REVISION = "a" * 40


def test_signed_release_binds_bytes_revision_and_sequence(tmp_path):
    from hashlib import sha256

    key = Ed25519PrivateKey.generate()
    archive = tmp_path / "release.zip"
    with ZipFile(archive, "w") as package:
        package.comment = REVISION.encode()
        for name in ("shooklink/__init__.py", "shooklink/__main__.py", "shooklink/app.py",
                     "shooklink/live_update/supervisor.py", "requirements.txt",
                     "run_serial_text_chat.bat"):
            package.writestr(f"ShookLink-{REVISION[:12]}/{name}", "# runtime\n")
    release = Release.sign(archive, 7, key)
    public = key.public_key().public_bytes_raw().hex()
    assert release.sha256 == sha256(archive.read_bytes()).hexdigest()
    assert release.revision == REVISION
    release.verify(public)
    other = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    with pytest.raises(ValueError, match="signature"):
        release.verify(other)
    with pytest.raises(ValueError, match="signature"):
        Release.parse(dict(release.as_dict(), sequence=8)).verify(public)
    destination = tmp_path / "slots" / ("b" * 32)
    stage_archive(archive, release, public, destination)
    assert (destination / "shooklink/app.py").read_text() == "# runtime\n"
    with archive.open("ab") as changed:
        changed.write(b"changed")
    with pytest.raises(ValueError, match="digest"):
        stage_archive(archive, release, public, tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_release_schema_excludes_commands_and_paths():
    fields = dict(revision=REVISION, sha256="f" * 64, sequence=1, signature="0" * 128)
    for extra in ({"command": "unused"}, {"path": "unused"}):
        with pytest.raises(ValueError):
            Release.parse(dict(fields, **extra))


def test_health_requires_matching_peer_job_build_new_instance_and_main_loop(tmp_path):
    key = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    (tmp_path / "publisher").write_text(key)
    state = State(job="9" * 32, phase="trial", target=REVISION,
                  peer="peer-one", old_instance="b" * 32, highest=8)
    state.save(tmp_path)
    receiver = ManagedReceiver(tmp_path)
    new = "c" * 32
    for args in ((state.job, "other-peer", new, REVISION), ("3" * 32, state.peer, new, REVISION),
                 (state.job, state.peer, state.old_instance, REVISION),
                 (state.job, state.peer, new, "checkout")):
        with pytest.raises(ValueError):
            receiver.confirm(*args)
    receiver.tick(state.peer, True)
    assert State.load(tmp_path) == state
    receiver.confirm(state.job, state.peer, new, REVISION)
    assert State.load(tmp_path).phase == "trial"
    receiver.tick(None, True)
    assert State.load(tmp_path).phase == "trial"
    receiver.tick(state.peer, True)
    assert State.load(tmp_path).phase == "healthy" and State.load(tmp_path).active == ""


def test_receiver_refuses_replay_and_vetoes_switch_if_busy(tmp_path):
    key = Ed25519PrivateKey.generate()
    (tmp_path / "publisher").write_text(key.public_key().public_bytes_raw().hex())
    archive = tmp_path / "release.zip"
    with ZipFile(archive, "w") as package:
        package.comment = REVISION.encode()
    release = Release.sign(archive, 2, key)
    State(highest=2).save(tmp_path)
    with pytest.raises(ValueError, match="superseded"):
        ManagedReceiver(tmp_path).apply("a" * 32, release, archive, "peer", "b" * 32)
    State(highest=2, job="a" * 32, phase="prepared", peer="peer").save(tmp_path)
    assert not ManagedReceiver(tmp_path).tick("peer", False)
    assert State.load(tmp_path).phase == "failed"
