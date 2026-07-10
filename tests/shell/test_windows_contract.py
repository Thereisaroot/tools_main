import sys
import threading

import pytest

from shooklink.shell.windows_conpty import WindowsConPtyProcess


class FakePty:
    def __init__(self):
        self.writes = []
        self.sizes = []
        self.alive = True
        self.read_count = 0
        self.closed = threading.Event()
        self.terminated = 0
        self.close_calls = 0
        self.pid = 1234

    def write(self, text):
        self.writes.append(text)

    def read(self, _size=4096):
        if self.read_count == 0:
            self.read_count += 1
            return "WINDOWS_OK"
        self.closed.wait(2)
        self.alive = False
        raise EOFError()

    def setwinsize(self, rows, columns):
        self.sizes.append((rows, columns))

    def isalive(self):
        return self.alive

    def terminate(self, force=False):
        self.terminated += 1
        self.alive = False
        self.closed.set()

    def close(self, force=False):
        self.close_calls += 1
        self.alive = False
        self.closed.set()


class FakeTreeGuard:
    def __init__(self, pid):
        self.pid = pid
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class FakePtyType:
    spawned = []

    @classmethod
    def spawn(cls, command, dimensions, env=None):
        process = FakePty()
        cls.spawned.append((command, dimensions, process))
        return process


def test_windows_backend_module_does_not_import_winpty_on_macos():
    if sys.platform != "win32":
        assert "winpty" not in sys.modules


def test_windows_backend_obeys_terminal_process_contract():
    output = []
    output_ready = threading.Event()
    exited = threading.Event()

    def receive_output(data):
        output.append(data)
        output_ready.set()

    process = WindowsConPtyProcess(
        receive_output,
        lambda code: exited.set(),
        command="cmd.exe",
        pty_process_type=FakePtyType,
        process_tree_guard_factory=FakeTreeGuard,
    )

    process.start(80, 24)
    process.write("한글\r".encode())
    process.resize(100, 40)

    assert output_ready.wait(2)
    command, dimensions, fake = FakePtyType.spawned[-1]
    assert command == "cmd.exe"
    assert dimensions == (24, 80)
    assert fake.writes == ["한글\r"]
    assert fake.sizes == [(40, 100)]
    assert output == [b"WINDOWS_OK"]
    process.terminate()
    assert exited.wait(2)
    assert fake.close_calls == 1
    assert process._tree_guard.pid == 1234
    assert process._tree_guard.close_calls == 1


def test_windows_backend_preserves_utf8_split_across_input_frames():
    process = WindowsConPtyProcess(
        lambda _data: None,
        lambda _code: None,
        command="cmd.exe",
        pty_process_type=FakePtyType,
        process_tree_guard_factory=FakeTreeGuard,
    )
    process.start(80, 24)
    encoded = "한🙂".encode("utf-8")

    process.write(encoded[:1])
    process.write(encoded[1:4])
    process.write(encoded[4:])

    fake = FakePtyType.spawned[-1][2]
    assert "".join(fake.writes) == "한🙂"
    process.terminate()


def test_windows_backend_fails_closed_when_process_tree_guard_cannot_attach():
    class FailingTreeGuard:
        def __init__(self, _pid):
            raise OSError("job unavailable")

    process = WindowsConPtyProcess(
        lambda _data: None,
        lambda _code: None,
        command="cmd.exe",
        pty_process_type=FakePtyType,
        process_tree_guard_factory=FailingTreeGuard,
    )

    with pytest.raises(OSError, match="job unavailable"):
        process.start(80, 24)

    fake = FakePtyType.spawned[-1][2]
    assert fake.close_calls == 1
    assert process.is_running() is False


def test_windows_backend_closes_handles_before_direct_termination():
    events = []
    read_started = threading.Event()
    release_read = threading.Event()

    class BlockingPty(FakePty):
        def read(self, _size=4096):
            read_started.set()
            release_read.wait(2)
            raise EOFError()

        def terminate(self, force=False):
            events.append(("terminate", force))
            super().terminate(force=force)

        def close(self, force=False):
            events.append(("close", force))
            self.close_calls += 1
            self.alive = False
            release_read.set()

    class BlockingPtyType:
        @classmethod
        def spawn(cls, command, dimensions, env=None):
            return BlockingPty()

    class OrderedGuard(FakeTreeGuard):
        def close(self):
            events.append(("guard", None))
            super().close()

    process = WindowsConPtyProcess(
        lambda _data: None,
        lambda _code: None,
        command="cmd.exe",
        pty_process_type=BlockingPtyType,
        process_tree_guard_factory=OrderedGuard,
    )
    process.start(80, 24)
    assert read_started.wait(1)

    process.terminate()

    assert events[:2] == [("guard", None), ("close", True)]
    assert not any(event[0] == "terminate" for event in events)


def test_windows_backend_passes_negotiated_term_in_child_environment(monkeypatch):
    captured = {}

    class EnvironmentPtyType:
        @classmethod
        def spawn(cls, command, dimensions, env):
            captured.update(command=command, dimensions=dimensions, env=env)
            return FakePty()

    monkeypatch.setenv("SHOOKLINK_PARENT_ENV", "preserved")
    process = WindowsConPtyProcess(
        lambda _data: None,
        lambda _code: None,
        command="cmd.exe",
        term="screen-256color",
        pty_process_type=EnvironmentPtyType,
        process_tree_guard_factory=FakeTreeGuard,
    )

    process.start(80, 24)

    assert captured["env"]["TERM"] == "screen-256color"
    assert captured["env"]["SHOOKLINK_PARENT_ENV"] == "preserved"
    process.terminate()
