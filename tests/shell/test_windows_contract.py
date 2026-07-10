import sys
import threading

from shooklink.shell.windows_conpty import WindowsConPtyProcess


class FakePty:
    def __init__(self):
        self.writes = []
        self.sizes = []
        self.alive = True
        self.reads = ["WINDOWS_OK", EOFError()]
        self.terminated = 0
        self.close_calls = 0
        self.pid = 1234

    def write(self, text):
        self.writes.append(text)

    def read(self, _size=4096):
        value = self.reads.pop(0)
        if isinstance(value, BaseException):
            self.alive = False
            raise value
        return value

    def setwinsize(self, rows, columns):
        self.sizes.append((rows, columns))

    def isalive(self):
        return self.alive

    def terminate(self, force=False):
        self.terminated += 1
        self.alive = False

    def close(self):
        self.close_calls += 1


class FakeTreeGuard:
    def __init__(self, pid):
        self.pid = pid
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class FakePtyType:
    spawned = []

    @classmethod
    def spawn(cls, command, dimensions):
        process = FakePty()
        cls.spawned.append((command, dimensions, process))
        return process


def test_windows_backend_module_does_not_import_winpty_on_macos():
    if sys.platform != "win32":
        assert "winpty" not in sys.modules


def test_windows_backend_obeys_terminal_process_contract():
    output = []
    exited = threading.Event()
    process = WindowsConPtyProcess(
        output.append,
        lambda code: exited.set(),
        command="cmd.exe",
        pty_process_type=FakePtyType,
        process_tree_guard_factory=FakeTreeGuard,
    )

    process.start(80, 24)
    process.write("한글\r".encode())
    process.resize(100, 40)

    assert exited.wait(2)
    command, dimensions, fake = FakePtyType.spawned[-1]
    assert command == "cmd.exe"
    assert dimensions == (24, 80)
    assert fake.writes == ["한글\r"]
    assert fake.sizes == [(40, 100)]
    assert output == [b"WINDOWS_OK"]
    process.terminate()
    assert fake.close_calls == 1
    assert process._tree_guard.pid == 1234
    assert process._tree_guard.close_calls == 1
