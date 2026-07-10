import os
import sys
import threading

import pytest

from shooklink.shell.unix_pty import UnixPtyProcess


pytestmark = pytest.mark.skipif(os.name != "posix", reason="Unix PTY only")


@pytest.mark.filterwarnings("error:This process.*fork")
def test_real_unix_pty_runs_interactive_shell_and_reports_exit(monkeypatch):
    def reject_fork():
        raise AssertionError("Unix PTY must not fork the multi-threaded GUI process")

    monkeypatch.setattr(os, "fork", reject_fork)
    output = bytearray()
    exited = threading.Event()
    exit_codes = []
    process = UnixPtyProcess(
        lambda data: output.extend(data),
        lambda code: (exit_codes.append(code), exited.set()),
        command=["/bin/sh"],
    )

    process.start(80, 24)
    process.resize(100, 30)
    process.write(b"printf 'PTY_OK\\n'\nexit\n")

    assert exited.wait(5), output.decode(errors="replace")
    assert b"PTY_OK" in output
    assert exit_codes == [0]
    assert not process.is_running()
    process.terminate()


def test_default_shell_command_is_available():
    process = UnixPtyProcess(lambda data: None, lambda code: None)

    assert process.command[0] == os.environ.get("SHELL", "/bin/sh")
    if sys.platform == "darwin":
        assert process.command[-1] == "-l"
