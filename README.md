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

The launchers do not update the Git checkout automatically. They check that
`shooklink`, `PySide6`, `serial`, `cryptography`, and `pyte` can be imported,
install `requirements.txt` only when one of those imports is missing, then
start the package entry point. Platform-marked packages install PyObjC only on
macOS and `pywinpty` only on Windows.

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

Next to `Ready`, `Peer v…` shows the version running on the other computer.
Peers that support build reporting also show the first 12 characters of the
running release commit (or `checkout` for a source run). Older peers show
only the version; this does not confirm that a particular update was installed.

Plain text is intentionally unencrypted and can be sent before trust is
established. Secure text, file contents, shell traffic, and shared input are
authenticated and encrypted after trust.

Typical serial port names are `/dev/cu.usbserial-*` on macOS and `COM3` on
Windows.

If **Connect** immediately shows **Error**, read the serial driver's message
below the port and baud fields, or use **Copy Error**. In older builds it is
at the bottom of the main window, to the left of **Send Plain**. **Handshaking**
is different: the local port is open but the peer handshake is not complete.

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

Remote shell access is enabled by default for mutually trusted peers. Nothing
starts automatically; a peer must request a session:

1. Connect and trust the peer on both computers. If you previously turned off
   `Allow Remote Shell`, turn it back on on the computer that will run the shell.
2. Select `Open Remote Shell` on the requesting computer.
3. Use `Terminate Session`, disable `Allow Remote Shell`, disconnect, or quit to
   stop the child process.

The local checkbox is remembered across reconnects and restarts. Updating
does not override a saved `off`, and a peer cannot enable it remotely. Only one
shell session per peer is accepted, and all shell input and output is encrypted.
macOS hosts the user's login shell; Windows hosts PowerShell and falls back to
`cmd.exe` when needed.

The terminal starts in **live keys** mode: typing, Tab, arrows and Ctrl+C go
directly to the remote terminal, including full-screen programs. Clicking old
output does not move the remote cursor. Drag to select and copy; typing resumes
the live display. For longer commands on a slow link, switch on **Edit command
locally** to edit at the bottom and send the line once with Enter. Use live
keys for passwords, completion and interactive programs. On a Windows host
with a configured WSL distribution, run `wsl.exe` at the remote Windows prompt.

The default display uses the [xterm.js](https://xtermjs.org/) engine. It has
scrollback, alternate screens for vim/top/tmux, ANSI colors, and bracketed
paste when the remote program enables it. **Find** searches terminal history;
use Enter/Next or Previous to cycle through matches. Use **Copy** or **Paste**,
Cmd+C/V on macOS, or Ctrl+Shift+C/V on Windows. Ctrl+Shift+F opens Find.
Use the platform zoom shortcuts or Ctrl+0 (Cmd+0 on macOS) to reset. For
completion, type a partial command/path and press Tab. Tab, Shift+Tab and
command history are handled by the remote shell, not a Mac-side file lookup.

The viewer loads its bundled scripts and styles locally, with no CDN, network
listener or Node.js dependency on either user machine. If WebEngine or its
assets cannot load, a visible native fallback keeps the existing session.
Set `SHOOKLINK_TERMINAL_RENDERER=native` to choose it before launch.

Real UART reads return as soon as bytes arrive instead of waiting for a 4 KiB
buffer to fill. Bursts of output repaint only the changed terminal rows.
Two updated peers compress larger output blocks before encryption only when
it saves bandwidth. Short prompts and keystrokes are not delayed for compression;
older peers continue to receive the original bytes.

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

### Local Control CLI

An already-running GUI exposes a token-authenticated control endpoint on
`127.0.0.1`. This lets scripts or local automation send through the GUI without
closing it or trying to open its serial port a second time:

```bash
shooklinkctl status
shooklinkctl send-file "/path/to/file.html"
shooklinkctl send-plain "plain message"
shooklinkctl send-secure "secure message"
```

Without an installed console script, use
`python3 -m shooklink.local_control.cli` followed by the same arguments. File
commands return only after final hash verification by the peer. Text commands
return only after peer acknowledgement. This is a local application feature;
the remote peer does not need an update unless it should also accept local CLI
commands.

### Update a peer over serial

From the ShookLink Git checkout on the sending computer, run:

```bash
python3 -m shooklink.local_control.cli send-update
```

This packages the current **committed HEAD** as `dist/ShookLink-<commit>.zip` and
sends it through the running GUI. The ZIP contains the tracked ShookLink Python
modules, allowlisted offline terminal assets and licenses, launchers, requirements,
project metadata and README. Use `pack-update`
instead of `send-update` to prepare the ZIP offline for `Choose File` or drag and
drop. Both commands accept `--repo PATH` and `--output-dir PATH`.

Any ShookLink peer that already supports encrypted file transfers can receive
the first update. After reception reports `complete`, on Windows:

1. Click `Open Downloads`. In Explorer, select the ZIP and choose **Extract All**
   into a fresh folder. The release folder name includes the sender's commit ID.
2. Close the running ShookLink, then run `run_serial_text_chat.bat` in the new
   release folder using the same Windows account and Python environment.
3. To return to the previous release, close the new one and start the old launcher.

The new release reuses the account's saved settings, identity and downloads.
Windows does not need Git or GitHub access. Python and dependency wheels are
not bundled. For a fully offline machine, dependencies for that release must
already be installed or supplied separately. On macOS use the `.sh` launcher
in the extracted release.

### Managed live updates on one serial port

The sending GUI keeps its port open. The Windows GUI receives the ZIP and
stages it while connected. A small foreground supervisor waits for that GUI
process to exit before starting the candidate on the same COM port, with the
same saved settings and identity. There is a brief gap in communication.
The supervisor never opens COM. Management is limited to probing the running
app and activating a publisher-signed ShookLink release.

**Allow updates once on Windows (no key entry)**

1. Start the updated Mac app. For an older Windows installation, send the
   ZIP as described above, extract it to a permanent folder, close the old
   app and double-click the new `run_serial_text_chat.bat`.
2. Connect and approve the peer as usual. At `Ready`, click **Allow Peer
   Updates** on Windows, using its own mouse. The button changes to
   **Peer Updates Allowed**. If it says **Update Peer App First**, update
   and restart the Mac app. Finish any active input, shell or file session first.
3. On subsequent Windows launches just double-click the same `.bat`.
   Starting the launcher or trusting the serial connection alone does **not**
   authorize updates. The button pins both the authenticated peer fingerprint
   and its separate release signer. Another peer or signer cannot replace it.
   The signing key stays in the Mac account's ShookLink app data, outside the ZIP.

   Agent/CLI commands talk to the sending GUI over its existing endpoint. Check the actual
   peer with `python3 -m shooklink.local_control.cli peer-update-status`. Its
   reply includes `managed`, `publisher`, `highest`, `revision`, `instance`,
   and the most recent transaction. A legacy peer requires the local bootstrap.

**Publish and apply a release**

From the release checkout on the signing computer, choose a positive sequence
greater than the receiver's `highest`:

```bash
python3 -m shooklink.local_control.cli sign-update --sequence 1
```

Give the generated ZIP and matching `.release.json` to the controller, in the
same folder. Replace `COMMIT_PREFIX` below with the 12 hex digits in the name:

```bash
python3 -m shooklink.local_control.cli live-update dist/ShookLink-COMMIT_PREFIX.release.json
```

If you already use a separately provisioned offline publisher, keep using
`sign-update --key PATH --sequence N` and the original local
`--managed --publisher PUBLIC_KEY_HEX` bootstrap. The button is not a key
rotation mechanism and does not replace that enrollment.

The controller checks for an idle, trusted, enrolled peer before sending. The
receiver accepts only a verified file transfer from that connection, checks the
release signature, ZIP digest, full revision and allowed archive contents, and
stages a fresh slot. It checks again for active file, shell and input sessions
before switching. Staged sequences are consumed even if rolled back; use a new
higher sequence for a new attempt. Repeating an already-running release reports
`already_running` without restarting it.

After reconnecting, the controller must get a fresh authenticated reply from
the same peer identity and installation, carrying the new process instance,
transaction ID and exact build. The controller acknowledges that instance,
the candidate's GUI event loop reports healthy, and the supervisor commits
the slot. Only a subsequent live reply for the committed slot yields
`verified`. A lost apply reply is safe to retry within the same transaction.

A failed start or a trial without this round trip for 60 seconds brings back
the previous healthy slot. An interrupted trial is also recovered on the next
managed launch. The CLI has a 180-second overall deadline (override with
`--timeout SECONDS`). If the cable is out, the local timeout cannot establish
the remote outcome. Reconnect and use `peer-update-status --job JOB_ID` to
inspect the last transaction and its reason. No remote shell, admin rights, second port, or
online package install is part of the handoff. Changes to the stable supervisor
itself require a new local bootstrap.

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
