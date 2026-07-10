"""Single-threaded helper that attaches a spawned shell to its PTY."""

from __future__ import annotations

import fcntl
import os
import sys
import termios


def main() -> int:
    command = sys.argv[1:]
    if not command:
        return 127
    try:
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        os.execvpe(command[0], command, os.environ)
    except BaseException as error:
        try:
            os.write(2, f"ShookLink shell start failed: {error}\r\n".encode())
        except BaseException:
            pass
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
