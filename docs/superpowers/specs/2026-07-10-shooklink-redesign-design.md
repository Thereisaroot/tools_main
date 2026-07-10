# ShookLink Redesign Design

## Summary

Replace the current single-file Tkinter application and its legacy serial
protocol with a modular PySide6 application named ShookLink. The new program
keeps the user's primary workflows: plain and secure text messages, file
transfer, bidirectional keyboard and mouse sharing, and a remote interactive
shell. Protocol compatibility with the old application is intentionally not
preserved.

macOS and Windows are equal peers. Either operating system can initiate text,
file, shell, or input-sharing sessions, and either can receive them. Linux is
not an acceptance target for the first release, although transport and Unix
PTY code should avoid unnecessary platform coupling.

Barrier is used only as an architectural reference. No Barrier source code is
copied or translated because Barrier is GPL-licensed. The new implementation
independently applies the concepts of explicit screen enter/leave events,
screen geometry exchange, logical cursor coordinates, absolute remote pointer
placement, and motion-event compression.

## Goals

- Preserve fast plain-text and secure-text messaging.
- Make file transfer use the available serial bandwidth without a per-chunk
  stop-and-wait delay.
- Provide a real interactive remote terminal with PTY semantics on macOS and
  ConPTY semantics on Windows.
- Require the executing peer to explicitly enable remote shell access.
- Replace the current delta/dead-space mouse implementation with deterministic
  screen topology and absolute remote cursor coordinates.
- Support input sharing in both directions between macOS and Windows.
- Keep text, file, terminal, keyboard, and mouse traffic responsive while they
  share one serial connection.
- Recover safely from corruption, disconnects, stale sessions, and application
  shutdown.

## Non-goals

- Compatibility with the legacy NUL-terminated JSON protocol.
- Direct compatibility with Barrier, Synergy, Input Leap, or SSH clients.
- More than two peers on one serial link.
- Remote desktop video or audio.
- Running a persistent background service when the GUI is closed.
- Supporting full input sharing on Linux in the first release.

## Application Architecture

The application will be rebuilt as a Python package rather than extending
`serial_text_chat.py`.

```text
shooklink/
  app.py                       Application bootstrap and dependency wiring
  settings.py                  Versioned persistent settings
  protocol/
    framing.py                 COBS framing, header codec, CRC validation
    messages.py                Typed message definitions and codecs
    crypto.py                  Peer identity, handshake, and AEAD helpers
  transport/
    serial_link.py             Serial reader/writer worker
    multiplexer.py             Streams, priorities, sequencing, flow control
  chat/
    service.py                 Plain and secure message workflows
  files/
    service.py                 Windowed sender, receiver, and SHA-256 finalize
  shell/
    service.py                 Shell session protocol and authorization
    unix_pty.py                macOS PTY process backend
    windows_conpty.py          Windows ConPTY process backend
  input/
    service.py                 Input session state and wire integration
    topology.py                Monitor geometry and edge mapping
    pointer.py                 Logical cursor and enter/leave state machine
    macos_backend.py           Quartz capture and injection
    windows_backend.py         Win32 hooks and SendInput injection
  ui/
    main_window.py             Connection, chat, files, and input controls
    terminal_window.py         ANSI terminal view and keyboard input
tests/
  ...                          Unit, integration, and transport simulations
```

PySide6 replaces Tkinter. This avoids mixing Tk's event loop with macOS AppKit
global event APIs, provides queued cross-thread signals, and gives the terminal
and drag-and-drop views predictable behavior on both target platforms.

## Serial Transport

### Framing

Each wire frame is COBS encoded and terminated by `0x00`. The decoded frame
contains a fixed binary header, payload, and CRC32. The header carries:

- protocol magic and version
- message type and flags
- stream identifier
- sequence number
- cumulative acknowledgement number
- payload length

Malformed COBS frames, invalid lengths, unsupported versions, and CRC failures
are discarded without terminating the reader. The delimiter allows the next
valid frame to resynchronize after corruption. Decoded frames are capped at
64 KiB to prevent unbounded allocation.

### Multiplexing

The writer owns multiple priority queues:

1. connection, authorization, key/button, and shell-input traffic
2. chat, shell-output, file control, and pointer enter/leave traffic
3. coalesced pointer movement and wheel traffic
4. file data traffic

Pointer movement is represented by only the newest pending absolute position.
It cannot create an unbounded queue. File data uses available capacity after
interactive traffic and obeys receiver-advertised flow control.

Every session-scoped message contains a random 128-bit session identifier.
Sequence numbers reject duplicates and stale packets. Disconnect resets all
streams and sessions.

### Capability Handshake

Peers exchange protocol version, application version, operating system,
supported features, maximum frame size, and monitor topology after the serial
port opens. A feature is enabled only when both peers advertise support.

Each installation creates a persistent Ed25519 identity. An ephemeral X25519
exchange derives session keys with HKDF, and the identity signs the handshake.
The first connection uses trust-on-first-use: both applications display the
peer fingerprint and require acceptance before shell, files, secure text, or
input sharing is enabled. A changed fingerprint is blocked and shown clearly.

Plain text remains deliberately unencrypted. Secure text, file contents,
remote shell traffic, and shared input use ChaCha20-Poly1305 with independent
nonces derived from stream and sequence numbers. This provides confidentiality
and integrity instead of the legacy marker-based obfuscation.

## Text Messaging

The message editor remains multiline and supports native selection, copy,
paste, undo, and platform shortcuts. `Send Plain` transmits the UTF-8 payload
without application encryption. `Send Secure` transmits an AEAD-encrypted
payload and is disabled until the peer is trusted and the secure handshake is
complete.

The receive view shows only message content plus a small Plain/Secure state;
wire metadata is never mixed into the text. Copy actions continue to target
the last received text message. Debug mode may log message type and byte count,
but never logs secure plaintext or cryptographic keys.

## File Transfer

The sender first transmits file name, size, modification time, and SHA-256.
The receiver sanitizes the file name, reserves a temporary `.part` file, and
either accepts or rejects the transfer.

Accepted files are sent as 16 KiB chunks with a configurable in-flight window
starting at 16 chunks. The receiver sends cumulative acknowledgements plus a
bitmap for missing chunks. Only missing chunks are retransmitted. The frame
CRC protects each chunk in transit, so a separate hash per chunk is not used.
After all chunks arrive, the receiver verifies the complete SHA-256 and uses an
atomic rename into the download directory. A mismatch deletes the temporary
file and reports failure.

File data is encrypted after peer trust. Drag-and-drop and file selection use
the same service. The UI shows transferred bytes, effective throughput, and
cancel state. Cancellation is session-scoped and cannot cancel unrelated text,
shell, or input traffic.

## Remote Interactive Shell

### Authorization and Lifecycle

`Allow Remote Shell` is off by default and is not silently enabled on restart.
The requesting peer can open a terminal only while the executing peer has this
toggle enabled and the peer identity is trusted. Only one shell session may be
active per peer in the first release. The executing UI clearly displays the
active shell and provides a local `Terminate Session` action.

Opening the shell creates a random session ID and negotiates terminal type,
columns, rows, and UTF-8 support. All shell frames are encrypted. Disconnect,
permission disable, application exit, or explicit close terminates the child
process tree and closes its handles.

### Platform Backends

On macOS, the backend creates the user's configured shell through a real PTY,
defaults to `$SHELL -l`, assigns a controlling terminal, and propagates window
size changes and signals.

On Windows, the backend uses ConPTY through `pywinpty` and starts PowerShell by
default, falling back to `cmd.exe` if PowerShell is unavailable. It propagates
terminal resize and raw input bytes and terminates the complete process tree.

The terminal popup uses `pyte` to interpret ANSI/VT output. It forwards normal
text, arrows, function keys, Tab, Enter, Backspace, Ctrl combinations, and
paste as terminal byte sequences. `Ctrl+C` is sent to the remote PTY rather
than copied when the terminal has focus; copying uses the platform terminal
shortcut or context menu. Full-screen applications such as `vim` and `top`
are acceptance cases.

## Keyboard and Mouse Sharing

### Session Model

Input sharing is symmetric: macOS can control Windows, Windows can control
macOS, and either can return control. Sessions use explicit request, accept,
enter, leave, stop, and release-all messages. The receiving peer remains
available only after trust and may disable input sharing locally.

Emergency stop and emergency exit shortcuts are handled in the platform
capture backend before event forwarding and are never transmitted. Session
shutdown releases every tracked key and mouse button on both peers.

### Native Backends

The macOS backend uses a Quartz event tap for keyboard and mouse capture and
Quartz event posting for injection. All event-tap setup is performed on a
dedicated CFRunLoop thread; UI work is returned to Qt through queued signals.
Accessibility and Input Monitoring permissions are checked before accepting a
session.

The Windows backend uses `SetWindowsHookEx` low-level keyboard and mouse hooks
for capture and `SendInput` for injection. Hook callbacks do bounded work and
enqueue normalized events. Windows scan codes and extended-key flags preserve
punctuation, keypad keys, modifiers, Caps Lock, and function keys.

### Monitor Topology and Cursor Model

Each peer advertises every connected monitor as a rectangle in its native
virtual-desktop coordinate space. Internal monitor seams and empty bounding-box
areas are not considered switchable edges. The configured peer side selects
which outer edge segments connect the two desktops.

When the source cursor pushes through a connected segment, it maps the
cross-axis position to the destination edge and sends `POINTER_ENTER` with an
absolute destination coordinate. The controller then captures relative local
motion at a stable anchor, but applies each delta immediately to a logical
cursor in the destination's coordinate system. The logical coordinate is
clamped to actual destination monitor rectangles and impossible movement is
discarded rather than accumulated.

The transmitted movement is an absolute destination coordinate, not an
accumulated raw delta. The receiver places its cursor at that coordinate and
periodically reports the applied position so the controller can reconcile any
OS-level clamp. Crossing the destination edge connected to the source sends
`POINTER_LEAVE`; the source cursor is restored one pixel inside its matching
edge. This eliminates the legacy dead-space debt caused by retaining deltas
while the real pointer did not move.

Move messages carry sequence numbers and are coalesced to at most 120 updates
per second. Button and key events flush any pending pointer position first so
clicks occur at the intended coordinate.

## UI

The main window contains five focused areas:

- connection and peer trust state
- multiline chat with `Send Plain` and `Send Secure`
- file selection, drop target, progress, cancel, and download-folder action
- input sharing state, peer side, auto-edge toggle, manual toggle, and emergency
  shortcut help
- remote shell authorization, active-session state, and `Open Remote Shell`

Opening a shell creates a separate resizable terminal window. Connection and
feature states are explicit: Disconnected, Handshaking, Untrusted, Ready,
Transferring, Controlling, Being controlled, Shell active, Permission required,
and Error.

The last available serial port and normal UI preferences are restored. Security
permissions such as remote shell authorization are not auto-enabled.

## Error Handling and Safety

- A reader or writer failure closes the serial link once and notifies every
  feature service.
- Stale session messages are ignored.
- File writes stay in temporary paths until hash verification succeeds.
- Terminal output and paste payloads have bounded buffers.
- Remote shell and input sharing require a trusted peer and explicit local
  enablement.
- Remote process handles, PTYs, hooks, event taps, pressed inputs, and temporary
  files are cleaned up on disconnect and shutdown.
- The GUI thread never performs serial I/O, hashing, file I/O, terminal reads,
  or native input-hook work.

## Dependencies

Core dependencies are PySide6, pyserial, cryptography, pyte, and platform-marked
packages. `pywinpty` is installed only on Windows. PyObjC Quartz frameworks are
installed only on macOS. pytest is used for tests.

Run scripts create or use a virtual environment, install the correct platform
dependencies, and launch `python -m shooklink`. The old entry point remains only
as a short migration notice during development and is removed before release.

## Testing and Acceptance

Pure unit tests cover COBS round trips, frame corruption and resynchronization,
message limits, sequence handling, crypto handshake and fingerprint changes,
topology mapping, logical cursor clamping, and stale session rejection.

Transport simulation tests use paired in-memory byte streams with fragmented,
delayed, duplicated, dropped, and corrupted frames. They verify interactive
traffic priority while a large file is transferring.

File tests cover Unicode names, traversal attempts, cancellation, selective
retransmission, final hash mismatch, and atomic completion. Shell tests run a
real PTY on macOS and a ConPTY contract suite on Windows. Terminal tests verify
ANSI rendering, resize, Ctrl+C, and process cleanup.

Input tests cover all four screen directions, multiple non-aligned monitors,
gaps, negative virtual coordinates, rapid enter/leave, coordinate
reconciliation, and release-all. Platform tests verify native key mapping for
letters, digits, punctuation, keypad keys, modifiers, Caps Lock, arrows,
function keys, mouse buttons, and wheel input.

GitHub Actions runs the portable test suite on macOS and Windows. Manual
acceptance is performed in both directions:

- macOS controller to Windows receiver
- Windows controller to macOS receiver
- macOS shell client to Windows ConPTY host
- Windows shell client to macOS PTY host
- plain and secure text during file transfer
- file transfer while moving the pointer and typing
- edge entry and immediate return without dead space
- cable removal during file, shell, and input sessions

## Migration

The implementation lives on `codex/remote-shell-barrier-mouse`. The legacy app
remains runnable until the new transport, chat, and file paths pass acceptance.
After both peers can run the new application, the launch scripts switch to
ShookLink and the legacy implementation is removed in a final cleanup commit.
