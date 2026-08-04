# ShookLink

ShookLink is a PySide6 desktop application for two computers connected by a
serial null-modem link. macOS and Windows are equal peers: either side can send
text and files, open a remote shell, or control the other side's keyboard and
mouse.

> [!IMPORTANT]
> ShookLink's binary protocol is incompatible with the legacy
> `serial_text_chat.py` application. Both computers must run ShookLink. Do not
> connect one ShookLink peer to the old app.

## Supported Platforms

- Python 3.11 or newer
- macOS and Windows as full desktop clients
- A serial port or null-modem adapter supported by `pyserial`

The Windows application is not a server-only companion. It can initiate and
receive every supported session, including opening the macOS login shell and
hosting a PowerShell ConPTY session for macOS. Linux is not an acceptance target
for the first ShookLink release.

## Install and Run

The Windows launcher first runs `git pull --ff-only` and stops if the update
fails. The launchers then check that `shooklink`, `PySide6`, `serial`,
`cryptography`, and `pyte` can be imported. They install `requirements.txt`
only when one of those imports is missing, then start the package entry point.
Platform-marked packages install PyObjC only on macOS and `pywinpty` only on
Windows.

macOS:

```bash
chmod +x run_serial_text_chat.sh
./run_serial_text_chat.sh
```

Windows:

```bat
run_serial_text_chat.bat
```

Both launchers forward command-line arguments, including `--debug`:

```bash
./run_serial_text_chat.sh --debug
```

You can also install and run the package directly:

```bash
python3 -m pip install -r requirements.txt
python3 -m shooklink
```

On Windows, use the Python launcher if `python` is not on `PATH`:

```bat
py -3 -m pip install -r requirements.txt
py -3 -m shooklink
```

## Connect and Establish Trust

1. Connect the two computers with a null-modem serial cable or equivalent
   adapter.
2. Start ShookLink on both computers and select each local serial port.
3. Select the same baud rate on both peers, then connect.
4. On a first connection, compare the displayed peer fingerprints through a
   separate trusted channel and approve the peer on both computers.
5. Wait for both applications to report the peer as trusted before using secure
   text, files, remote shell, or input sharing.

Trust is bilateral: one approval is not enough. ShookLink remembers an approved
identity for later connections. If the identity fingerprint changes, ShookLink
blocks protected features; investigate and verify the new identity instead of
approving an unexpected change.

Plain text is intentionally unencrypted and can be sent before trust is
established. Secure text, file contents, shell traffic, and shared input are
authenticated and encrypted after trust.

Typical serial port names are `/dev/cu.usbserial-*` on macOS and `COM3` on
Windows.

## Baud Rates

The connection screen provides these common rates:

- `115200` (default)
- `230400`
- `460800`
- `750000`
- `921600`
- `1000000`
- `1500000`
- `2000000`

The baud field is editable. Persisted custom values are accepted from `300`
through `4000000`, but the operating system, UART, cable, and adapter must also
support the selected rate. Both peers must use the same value. If a link is
unreliable, return to `115200` before troubleshooting higher rates.

ShookLink opens ports as 8 data bits, no parity, 1 stop bit, and no software or
hardware flow control.

## Remote Shell Authorization

Remote shell access requires a trusted peer and explicit permission on the
computer that will execute the shell:

1. Enable `Allow Remote Shell` on the executing computer.
2. Select `Open Remote Shell` on the requesting computer.
3. Use `Terminate Session`, disable `Allow Remote Shell`, disconnect, or quit to
   stop the child process.

`Allow Remote Shell` is off by default. Its local checkbox state is saved and
restored on the next launch; a peer still cannot enable it remotely. Only one
shell session per peer is accepted, and all shell input and output is encrypted.
macOS hosts the user's login shell; Windows hosts PowerShell and falls back to
`cmd.exe` when needed.

## Keyboard and Mouse Sharing

Enable `Allow Remote Input` on the computer that may be controlled. Choose which
side of the local desktop touches the peer, then use `Toggle Remote Control` or
enable `Auto Edge (This Computer -> Peer)`. Input permission is local, requires
trust, and its local checkbox state is restored on restart.

Auto edge is a controller-side option. To use one computer's physical mouse to
enter the peer and return, enable auto edge only on that controller; the peer
only needs `Allow Remote Input`. To let either computer initiate from its own
physical mouse, enable auto edge on both and configure mirrored sides, such as
`Right` on one computer and `Left` on the other.

Automatic return is disabled when Auto Edge is off. When enabled, the remote
pointer must reach the peer's real multi-monitor outer edge and continue moving
outward through a small resistance zone before control returns locally.

While input capture is active, these local emergency shortcuts are consumed
before they can be sent to the peer:

- `Ctrl+Alt+Shift+Backspace`: stop input sharing and release tracked keys and
  mouse buttons on both peers.
- `Ctrl+Alt+Shift+Escape`: release shared input and exit ShookLink.

Use Control, not Command, for these emergency combinations on macOS.

Application shortcuts:

- `Ctrl+Alt+V`: send plain text
- `Ctrl+Shift+Alt+V`: send secure text
- `Ctrl+Alt+C`: copy the last received text
- `Ctrl+Shift+Alt+C`: copy the last received secure text

## macOS Permissions

Keyboard and mouse sharing fails closed unless macOS grants both event capture
and event injection permission. Open `System Settings -> Privacy & Security` and
enable the terminal application used to launch ShookLink, and the Python
executable if macOS lists it separately, under:

- `Accessibility`
- `Input Monitoring`

Fully quit and relaunch the terminal application and ShookLink after changing
either permission. A permission denial disables input sharing without granting
the remote peer partial control; text, file, and shell features remain separate.

## Text and Files

- `Send Plain` sends visible UTF-8 without application encryption.
- `Send Secure` requires completed trust and sends authenticated encrypted text.
- Current ShookLink peers acknowledge text delivery and automatically retry a
  lost message or acknowledgement. `Sent` means the peer acknowledged the
  message; exhausted retries are reported as `Delivery failed`.
- `Choose File` and the drop target use the same encrypted, windowed transfer
  service.
- Received files are finalized only after SHA-256 verification and are stored in
  `~/Downloads/ShookLink` by default.
- Chat remains usable while a file or input session is active.

## Development

Install test dependencies and run the same portable checks used by CI:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m compileall shooklink
QT_QPA_PLATFORM=offscreen python3 -m pytest -q
```

CI runs this suite on macOS and Windows with Python 3.11 and 3.13. Native PTY,
ConPTY, and input behavior still requires manual acceptance on both operating
systems.
