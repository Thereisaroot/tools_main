# ShookLink Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy Tkinter serial chat with a modular PySide6 application that provides reliable plain/secure chat, windowed file transfer, PTY/ConPTY remote shell, and Barrier-inspired bidirectional input sharing on macOS and Windows.

**Architecture:** Build a versioned COBS-framed binary transport and priority multiplexer first, then add feature services that communicate only through typed messages. Keep platform-specific terminal and input code behind narrow interfaces, while the PySide6 UI subscribes to service events and never performs blocking work.

**Tech Stack:** Python 3.11+, PySide6, pyserial, cryptography, pyte, pywinpty on Windows, PyObjC Quartz on macOS, pytest.

---

## File Map

- `pyproject.toml`: package metadata, core dependencies, pytest configuration.
- `requirements.txt`: editable runtime dependency list used by existing launch scripts.
- `requirements-dev.txt`: test dependencies.
- `shooklink/settings.py`: persistent non-secret UI settings and security-safe defaults.
- `shooklink/protocol/framing.py`: COBS, frame header, CRC, incremental parser.
- `shooklink/protocol/messages.py`: message enum and typed payload codec.
- `shooklink/protocol/crypto.py`: identity, trust store, signed ephemeral handshake, AEAD.
- `shooklink/transport/multiplexer.py`: prioritized queues, sequence numbers, coalescing.
- `shooklink/transport/serial_link.py`: serial reader/writer lifecycle.
- `shooklink/chat/service.py`: plain and secure text state.
- `shooklink/files/service.py`: selective-repeat file sender and receiver.
- `shooklink/shell/service.py`: shell authorization and session protocol.
- `shooklink/shell/unix_pty.py`: Unix PTY process implementation.
- `shooklink/shell/windows_conpty.py`: Windows ConPTY process implementation.
- `shooklink/input/topology.py`: monitor rectangles and connected edge mapping.
- `shooklink/input/pointer.py`: logical remote cursor and enter/leave transitions.
- `shooklink/input/service.py`: input session protocol and release-all safety.
- `shooklink/input/macos_backend.py`: Quartz event capture and injection.
- `shooklink/input/windows_backend.py`: Win32 hooks and SendInput.
- `shooklink/ui/main_window.py`: main PySide6 application window.
- `shooklink/ui/terminal_window.py`: ANSI terminal popup.
- `shooklink/app.py`, `shooklink/__main__.py`: dependency wiring and entry point.
- `shooklink/core.py`: peer handshake and feature-service routing without UI.
- `tests/`: platform-neutral and platform-marked tests.
- `.github/workflows/test.yml`: macOS and Windows test matrix.

### Task 1: Package Scaffold and Settings

**Files:**
- Create: `pyproject.toml`
- Create: `requirements-dev.txt`
- Modify: `requirements.txt`
- Create: `shooklink/__init__.py`
- Create: `shooklink/__main__.py`
- Create: `shooklink/settings.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write the failing settings tests**

```python
from shooklink.settings import AppSettings, SettingsStore


def test_settings_round_trip(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.save(AppSettings(last_port="COM7", baud_rate=460800, peer_side="left"))
    assert store.load().last_port == "COM7"
    assert store.load().peer_side == "left"


def test_remote_permissions_never_restore_enabled(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"allow_remote_shell": true, "allow_input": true}')
    settings = SettingsStore(path).load()
    assert settings.allow_remote_shell is False
    assert settings.allow_input is False
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python3 -m pytest tests/test_settings.py -q`

Expected: import failure because `shooklink.settings` does not exist.

- [ ] **Step 3: Add packaging and the minimal settings implementation**

Use a frozen dataclass with `last_port`, `baud_rate`, `peer_side`,
`auto_edge_enabled`, `download_dir`, `allow_remote_shell`, and `allow_input`.
`SettingsStore.load()` must ignore unknown keys, validate the peer side and baud
rate, and force both permission fields to false after reading disk.

`pyproject.toml` must expose `shooklink = "shooklink.app:main"` and use platform
markers for `pywinpty` and PyObjC Quartz.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/test_settings.py -q`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt shooklink tests/test_settings.py
git commit -m "build: scaffold ShookLink package"
```

### Task 2: Binary Framing and Resynchronization

**Files:**
- Create: `shooklink/protocol/__init__.py`
- Create: `shooklink/protocol/framing.py`
- Create: `tests/protocol/test_framing.py`

- [ ] **Step 1: Write failing frame tests**

```python
import pytest

from shooklink.protocol.framing import Frame, FrameParser, decode_frame, encode_frame


def test_frame_round_trip_with_zero_bytes():
    frame = Frame(message_type=7, flags=1, priority=2, stream_id=4,
                  sequence=9, acknowledgement=3, payload=b"a\x00b")
    assert decode_frame(encode_frame(frame)[:-1]) == frame


def test_parser_resynchronizes_after_corrupt_frame():
    valid = encode_frame(Frame(1, 0, 1, 0, 1, 0, b"ok"))
    parser = FrameParser()
    frames = parser.feed(b"broken\x00" + valid[:4])
    frames += parser.feed(valid[4:])
    assert [frame.payload for frame in frames] == [b"ok"]


def test_oversized_payload_is_rejected():
    with pytest.raises(ValueError):
        encode_frame(Frame(1, 0, 1, 0, 1, 0, b"x" * 65536))
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python3 -m pytest tests/protocol/test_framing.py -q`

Expected: import failure for `shooklink.protocol.framing`.

- [ ] **Step 3: Implement framing**

Define `HEADER = struct.Struct(">2sBBBBIIII")`, `MAGIC = b"SL"`, and
`VERSION = 1`. CRC32 covers the decoded header and payload. `encode_frame()`
returns COBS bytes followed by `b"\x00"`. `FrameParser.feed()` keeps a bounded
byte buffer, parses every complete delimiter-separated packet, drops malformed
packets, and resumes at the next delimiter.

- [ ] **Step 4: Run frame tests and full tests**

Run: `python3 -m pytest tests/protocol/test_framing.py -q`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add shooklink/protocol tests/protocol
git commit -m "feat: add resilient binary framing"
```

### Task 3: Typed Messages and Secure Sessions

**Files:**
- Create: `shooklink/protocol/messages.py`
- Create: `shooklink/protocol/crypto.py`
- Create: `tests/protocol/test_messages.py`
- Create: `tests/protocol/test_crypto.py`

- [ ] **Step 1: Write failing typed-message tests**

```python
from shooklink.protocol.messages import Message, MessageType, decode_message, encode_message


def test_message_metadata_and_binary_body_round_trip():
    original = Message(MessageType.FILE_CHUNK, {"transfer_id": "abc", "index": 2}, b"\x00data")
    assert decode_message(encode_message(original)) == original


def test_unknown_message_type_is_rejected():
    metadata = b'{"type":999,"meta":{}}'
    encoded = len(metadata).to_bytes(4, "big") + metadata
    assert decode_message(encoded) is None
```

Use a deterministic envelope: four-byte JSON metadata length, compact UTF-8
JSON metadata, then raw binary body. Metadata is capped at 16 KiB.

- [ ] **Step 2: Write failing cryptographic tests**

```python
from shooklink.protocol.crypto import IdentityStore, SecureSession, TrustStore


def test_two_signed_handshakes_derive_matching_keys(tmp_path):
    alice = IdentityStore(tmp_path / "alice.key").load_or_create()
    bob = IdentityStore(tmp_path / "bob.key").load_or_create()
    a, b = SecureSession.initiator(alice), SecureSession.responder(bob)
    b.receive_hello(a.create_hello())
    a.receive_hello(b.create_hello())
    sealed = a.encrypt(4, 7, b"secret")
    assert b.decrypt(4, 7, sealed) == b"secret"


def test_changed_identity_is_not_trusted(tmp_path):
    trust = TrustStore(tmp_path / "trusted.json")
    trust.accept("peer", "fingerprint-a")
    assert trust.check("peer", "fingerprint-b") is False
```

- [ ] **Step 3: Run message and crypto tests and verify RED**

Run: `python3 -m pytest tests/protocol/test_messages.py tests/protocol/test_crypto.py -q`

Expected: imports fail because both modules are missing.

- [ ] **Step 4: Implement messages and crypto**

Define a stable `MessageType(IntEnum)` covering HELLO, TRUST, CHAT_PLAIN,
CHAT_SECURE, FILE_OFFER, FILE_ACCEPT, FILE_CHUNK, FILE_ACK, FILE_FINISH,
FILE_CANCEL, SHELL_OPEN, SHELL_ACCEPT, SHELL_DENY, SHELL_INPUT, SHELL_OUTPUT,
SHELL_RESIZE, SHELL_EXIT, INPUT_REQUEST, INPUT_ACCEPT, INPUT_BUSY, INPUT_ENTER,
INPUT_LEAVE, INPUT_KEY, INPUT_BUTTON, INPUT_MOVE, INPUT_WHEEL,
INPUT_POINTER_STATE, INPUT_STOP, and INPUT_RELEASE_ALL.

Use Ed25519 persistent identities, ephemeral X25519 keys, signed hello payloads,
HKDF-SHA256, and ChaCha20-Poly1305. Derive the 12-byte nonce from a direction
prefix plus stream ID and sequence number. Reject nonce reuse and invalid
signatures. Store private keys with owner-only permissions where supported.

- [ ] **Step 5: Verify GREEN**

Run: `python3 -m pytest tests/protocol -q`

Expected: all protocol tests pass.

- [ ] **Step 6: Commit**

```bash
git add shooklink/protocol tests/protocol
git commit -m "feat: add typed messages and secure sessions"
```

### Task 4: Priority Multiplexer and Serial Worker

**Files:**
- Create: `shooklink/transport/__init__.py`
- Create: `shooklink/transport/multiplexer.py`
- Create: `shooklink/transport/serial_link.py`
- Create: `tests/transport/test_multiplexer.py`
- Create: `tests/transport/test_serial_link.py`

- [ ] **Step 1: Write failing multiplexer tests**

```python
from shooklink.transport.multiplexer import Multiplexer, OutboundItem, Priority


def test_interactive_traffic_overtakes_file_data():
    mux = Multiplexer()
    mux.enqueue(OutboundItem(Priority.FILE, 5, b"chunk"))
    mux.enqueue(OutboundItem(Priority.INTERACTIVE, 2, b"key"))
    assert mux.pop().payload == b"key"


def test_pointer_moves_are_coalesced_per_stream():
    mux = Multiplexer()
    mux.enqueue_pointer(9, b"first")
    mux.enqueue_pointer(9, b"latest")
    assert mux.pop().payload == b"latest"
    assert mux.empty()
```

- [ ] **Step 2: Write a failing fragmented-link test**

Create a fake serial endpoint whose `read()` returns one to seven bytes and
whose peer receives writes. Assert that two `SerialLink` instances exchange a
typed payload, report one disconnect, and terminate both worker threads.

- [ ] **Step 3: Run transport tests and verify RED**

Run: `python3 -m pytest tests/transport -q`

Expected: missing transport modules.

- [ ] **Step 4: Implement the multiplexer and serial lifecycle**

Use a condition-protected heap for normal priority items and a dictionary for
coalesced pointer moves. `SerialLink` owns one reader and one writer thread,
uses `FrameParser`, never invokes UI callbacks while holding internal locks,
and exposes `start()`, `send()`, and idempotent `close()` methods. Configure
8N1, no flow control, bounded read timeout, and bounded write timeout.

- [ ] **Step 5: Verify GREEN and thread cleanup**

Run: `python3 -m pytest tests/transport -q`

Expected: all tests pass without leaked-thread warnings.

- [ ] **Step 6: Commit**

```bash
git add shooklink/transport tests/transport
git commit -m "feat: add multiplexed serial transport"
```

### Task 5: Chat Service and Initial Qt Window

**Files:**
- Create: `shooklink/chat/__init__.py`
- Create: `shooklink/chat/service.py`
- Create: `shooklink/ui/__init__.py`
- Create: `shooklink/ui/main_window.py`
- Create: `shooklink/app.py`
- Modify: `shooklink/__main__.py`
- Create: `tests/chat/test_service.py`
- Create: `tests/ui/test_main_window.py`

- [ ] **Step 1: Write failing chat tests**

```python
import pytest

from shooklink.chat.service import ChatService, PeerNotTrusted


class FakeBus:
    def __init__(self, trusted=False):
        self.trusted = trusted
        self.last_message = None

    def send(self, message, *, secure=False):
        self.last_message = message


def test_plain_chat_sends_utf8():
    bus = FakeBus()
    service = ChatService(bus)
    service.send_plain("한글 message")
    assert bus.last_message.body.decode() == "한글 message"


def test_secure_chat_requires_trust():
    with pytest.raises(PeerNotTrusted):
        ChatService(FakeBus()).send_secure("secret")
```

- [ ] **Step 2: Run chat tests and verify RED**

Run: `python3 -m pytest tests/chat -q`

- [ ] **Step 3: Implement chat and the minimal UI**

`ChatService` validates UTF-8 byte limits, sends typed messages, decrypts secure
messages before emitting `message_received`, and retains only the last text for
copy actions. Build a PySide6 window with connection controls, multiline editor,
received-message view, `Send Plain`, `Send Secure`, and both copy buttons. Use
Qt's native shortcuts; do not intercept printable key events manually.

- [ ] **Step 4: Add an offscreen Qt smoke test**

Set `QT_QPA_PLATFORM=offscreen`, create `MainWindow`, put Korean and punctuation
in the editor, invoke both copy actions, and close the window without starting
a serial worker.

- [ ] **Step 5: Verify GREEN**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/chat tests/ui/test_main_window.py -q`

- [ ] **Step 6: Commit**

```bash
git add shooklink/chat shooklink/ui shooklink/app.py shooklink/__main__.py tests/chat tests/ui
git commit -m "feat: add chat service and Qt application shell"
```

### Task 6: Windowed File Transfer

**Files:**
- Create: `shooklink/files/__init__.py`
- Create: `shooklink/files/service.py`
- Create: `tests/files/test_service.py`
- Modify: `shooklink/ui/main_window.py`

- [ ] **Step 1: Write failing receiver safety tests**

Test that `../../escape.txt` is reduced to a safe basename, data remains in a
`.part` file before completion, a bad final SHA-256 removes the partial file,
and a valid hash atomically produces the final file.

- [ ] **Step 2: Write failing selective-repeat tests**

Construct a 40-chunk in-memory file, drop chunk indexes 3 and 17, deliver a
cumulative ACK plus missing bitmap, and assert only those two indexes are
requeued. Assert the sender never has more than 16 unacknowledged chunks.

- [ ] **Step 3: Run file tests and verify RED**

Run: `python3 -m pytest tests/files -q`

- [ ] **Step 4: Implement sender and receiver state machines**

Use 16 KiB chunks, a 16-chunk initial window, monotonic retransmission timers,
and final SHA-256 verification. Hash files on worker threads. File chunks use
the lowest multiplexer priority and are encrypted before framing. Cancellation
must close file handles, remove partial files, and affect only its transfer ID.

- [ ] **Step 5: Integrate file picker, drop target, progress, cancel, and folder action**

The Qt drop target accepts local file URLs only. Show bytes, percentage,
throughput, and final path. Keep chat controls active during a transfer.

- [ ] **Step 6: Verify GREEN**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/files tests/ui -q`

- [ ] **Step 7: Commit**

```bash
git add shooklink/files shooklink/ui/main_window.py tests/files tests/ui
git commit -m "feat: add windowed verified file transfer"
```

### Task 7: PTY and ConPTY Remote Shell

**Files:**
- Create: `shooklink/shell/__init__.py`
- Create: `shooklink/shell/process.py`
- Create: `shooklink/shell/unix_pty.py`
- Create: `shooklink/shell/windows_conpty.py`
- Create: `shooklink/shell/service.py`
- Create: `shooklink/ui/terminal_window.py`
- Create: `tests/shell/test_service.py`
- Create: `tests/shell/test_unix_pty.py`
- Create: `tests/shell/test_windows_contract.py`
- Modify: `shooklink/ui/main_window.py`

- [ ] **Step 1: Define the process contract with failing tests**

```python
class TerminalProcess(Protocol):
    def start(self, columns: int, rows: int) -> None:
        raise NotImplementedError

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def resize(self, columns: int, rows: int) -> None:
        raise NotImplementedError

    def terminate(self) -> None:
        raise NotImplementedError

    def is_running(self) -> bool:
        raise NotImplementedError
```

Test one start, raw writes, resize, output callback, idempotent terminate, and
process-tree cleanup using a fake implementation.

- [ ] **Step 2: Write a failing real Unix PTY test**

Start `/bin/sh`, write `printf 'PTY_OK\\n'\nexit\n`, collect output, and assert
`PTY_OK` and an exit callback arrive within five seconds.

- [ ] **Step 3: Run shell tests and verify RED**

Run: `python3 -m pytest tests/shell -q`

- [ ] **Step 4: Implement Unix PTY, Windows ConPTY, and shell service**

Unix uses `pty.openpty`, `setsid`, `TIOCSCTTY`, and `TIOCSWINSZ`. Windows imports
`winpty.PtyProcess` only inside the Windows backend, starts PowerShell or
`cmd.exe`, and implements the same contract. `ShellService` accepts `SHELL_OPEN`
only when permission is currently enabled, peer trust is valid, and no other
session is active. Shell input/output/resize/exit messages are always encrypted.

- [ ] **Step 5: Implement the terminal popup**

Use `pyte.Screen` and `pyte.Stream` for VT rendering. Convert Qt key events to
terminal sequences, forward paste, resize the PTY from character-cell geometry,
send Ctrl+C as `b"\x03"`, and bound scrollback to 10,000 lines.

- [ ] **Step 6: Verify locally and through Windows contract tests**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/shell tests/ui -q`

Expected on macOS: Unix PTY tests pass and Windows-only runtime tests skip; the
Windows backend contract imports without importing `winpty` on macOS.

- [ ] **Step 7: Commit**

```bash
git add shooklink/shell shooklink/ui tests/shell tests/ui
git commit -m "feat: add authorized remote terminal sessions"
```

### Task 8: Monitor Topology and Barrier-Inspired Pointer Model

**Files:**
- Create: `shooklink/input/__init__.py`
- Create: `shooklink/input/topology.py`
- Create: `shooklink/input/pointer.py`
- Create: `tests/input/test_topology.py`
- Create: `tests/input/test_pointer.py`

- [ ] **Step 1: Write failing topology tests**

Use monitors `Rect(0, 0, 1920, 1080)` and `Rect(1920, 200, 1280, 1024)`.
Assert the internal seam is not an outer edge, the right edge exists only over
the second monitor's vertical span, negative-coordinate monitors work, and a
source cross-axis fraction maps inside the destination edge.

- [ ] **Step 2: Write failing dead-space regression tests**

Enter a 1920x1080 destination at its right-connected left edge, push upward for
two seconds while already clamped at y=0, then move downward by one pixel.
Assert the logical coordinate becomes y=1 immediately. Repeat for left, right,
and bottom edges and for monitor gaps.

- [ ] **Step 3: Run input geometry tests and verify RED**

Run: `python3 -m pytest tests/input/test_topology.py tests/input/test_pointer.py -q`

- [ ] **Step 4: Implement geometry and logical cursor**

Use immutable `Rect`, `Monitor`, `EdgeSegment`, and `Topology` values. Movement
must select the containing monitor, clamp to the nearest valid point without
retaining overflow, and emit `ENTER`, `MOVE`, or `LEAVE` transitions. Entry
maps the cross-axis fraction to an actual destination edge segment. Absolute
positions are integers in the destination's native virtual-desktop space.

- [ ] **Step 5: Verify GREEN**

Run: `python3 -m pytest tests/input/test_topology.py tests/input/test_pointer.py -q`

- [ ] **Step 6: Commit**

```bash
git add shooklink/input tests/input
git commit -m "feat: add deterministic screen transition model"
```

### Task 9: Native macOS and Windows Input Backends

**Files:**
- Create: `shooklink/input/events.py`
- Create: `shooklink/input/backend.py`
- Create: `shooklink/input/macos_backend.py`
- Create: `shooklink/input/windows_backend.py`
- Create: `tests/input/test_events.py`
- Create: `tests/input/test_backend_contract.py`

- [ ] **Step 1: Write failing normalized-event tests**

Define keyboard events by physical usage/scan code plus text and modifier state,
not by guessed printable characters. Cover letters, top-row digits, keypad,
`/ ? . > = + - _`, left/right modifiers, Caps Lock, arrows, function keys,
media-independent mouse buttons, movement, and wheel axes.

- [ ] **Step 2: Write a backend lifecycle contract test**

Every backend must report permissions and monitors, start capture once, stop
idempotently, inject normalized events, suppress its own injected events, expose
cursor position, warp the capture anchor, and release tracked inputs.

- [ ] **Step 3: Run tests and verify RED**

Run: `python3 -m pytest tests/input/test_events.py tests/input/test_backend_contract.py -q`

- [ ] **Step 4: Implement the macOS backend**

Use Quartz event taps on a dedicated CFRunLoop thread. Preserve hardware key
codes and flags, identify injected events with a private event-source marker,
post keyboard/mouse events through Quartz, enumerate `NSScreen`/CoreGraphics
display bounds, and fail closed when Accessibility or Input Monitoring is not
granted. Never call AppKit from a serial or hook callback thread.

- [ ] **Step 5: Implement the Windows backend**

Use a dedicated message-loop thread with `WH_KEYBOARD_LL` and `WH_MOUSE_LL`.
Normalize scan code, virtual key, extended flag, injected flag, and modifier
state. Inject with `SendInput`, mark events with a private `dwExtraInfo`, use
`EnumDisplayMonitors` for topology, and support negative virtual coordinates.
The emergency shortcuts are consumed before forwarding on both platforms.

- [ ] **Step 6: Verify platform-neutral tests and imports**

Run: `python3 -m pytest tests/input -q`

Expected on macOS: pure and macOS tests pass, Windows runtime tests skip, and
the Windows module imports without executing Win32-only setup.

- [ ] **Step 7: Commit**

```bash
git add shooklink/input tests/input
git commit -m "feat: add native macOS and Windows input backends"
```

### Task 10: Input Session Service and UI Integration

**Files:**
- Create: `shooklink/input/service.py`
- Create: `tests/input/test_service.py`
- Modify: `shooklink/ui/main_window.py`
- Modify: `shooklink/app.py`

- [ ] **Step 1: Write failing session-state tests**

Test idle to requesting to controlling, idle to being-controlled, simultaneous
requests with deterministic busy resolution, stale session rejection,
disconnect release-all, manual stop, emergency stop, and auto-edge enter/leave.
Assert a pending absolute pointer position is sent before button-down.

- [ ] **Step 2: Run session tests and verify RED**

Run: `python3 -m pytest tests/input/test_service.py -q`

- [ ] **Step 3: Implement the service**

Wire normalized backend events to typed encrypted messages. Exchange topology
before accepting control, send explicit enter/leave, coalesce absolute motion
to 120 Hz, reconcile receiver pointer reports, track pressed keys/buttons, and
call release-all on every stop/error path. Use a stable peer-ID tie-breaker for
simultaneous requests.

- [ ] **Step 4: Add input controls to the main window**

Add local input permission, peer side, auto-edge, manual toggle, state label,
permission status, and emergency shortcut help. Disable only conflicting
actions; text and file services remain operational during input sharing.

- [ ] **Step 5: Verify GREEN**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/input tests/ui -q`

- [ ] **Step 6: Commit**

```bash
git add shooklink/input shooklink/ui/main_window.py shooklink/app.py tests/input tests/ui
git commit -m "feat: integrate bidirectional input sharing"
```

### Task 11: End-to-End Wiring, Launchers, CI, and Documentation

**Files:**
- Create: `shooklink/core.py`
- Modify: `shooklink/app.py`
- Modify: `run_serial_text_chat.sh`
- Modify: `run_serial_text_chat.bat`
- Modify: `README.md`
- Create: `.github/workflows/test.yml`
- Create: `tests/integration/test_peer_pair.py`

- [ ] **Step 1: Write a failing paired-peer integration test**

Connect two application cores through fragmented in-memory serial endpoints.
Complete trust, exchange plain and secure messages, transfer a multi-window
file, open a fake shell, exchange input enter/move/leave, then disconnect.
Assert no service retains a session, pressed input, temporary file, thread, or
process handle.

- [ ] **Step 2: Run integration test and verify RED**

Run: `python3 -m pytest tests/integration/test_peer_pair.py -q`

- [ ] **Step 3: Complete dependency wiring and startup behavior**

Create services after the serial link connects, route decoded messages by
feature, expose trust approval in the UI, auto-select and auto-connect only a
still-present saved port, and close all services before Qt exits. Preserve
`--debug`, but redact secure payloads and suppress pointer-move logs.

- [ ] **Step 4: Update launchers and documentation**

Both launchers must test `import shooklink, PySide6, serial, cryptography, pyte`
and install requirements only when needed, then run `python -m shooklink` while
forwarding arguments. Document first trust, shell authorization, macOS
permissions, Windows client support, emergency shortcuts, supported baud rates,
and protocol incompatibility with the old app.

- [ ] **Step 5: Add cross-platform CI**

Run Python 3.11 and 3.13 on `macos-latest` and `windows-latest`, install the
platform-marked dependencies, run `python -m compileall shooklink`, and run
pytest with `QT_QPA_PLATFORM=offscreen` where supported.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
python3 -m compileall -q shooklink
QT_QPA_PLATFORM=offscreen python3 -m pytest -q
```

Expected: all local tests pass and platform-specific non-macOS tests skip with
an explicit reason.

- [ ] **Step 7: Commit**

```bash
git add shooklink run_serial_text_chat.sh run_serial_text_chat.bat README.md .github tests
git commit -m "feat: complete ShookLink cross-platform application"
```

### Task 12: Migration Cleanup and Manual Acceptance

**Files:**
- Delete after acceptance: `serial_text_chat.py`
- Rename: `run_serial_text_chat.sh` to `run_shooklink.sh`
- Rename: `run_serial_text_chat.bat` to `run_shooklink.bat`
- Modify: `README.md`

- [ ] **Step 1: Run the complete automated suite**

Run:

```bash
python3 -m compileall -q shooklink
QT_QPA_PLATFORM=offscreen python3 -m pytest -q
git diff --check
```

Expected: zero failures and no whitespace errors.

- [ ] **Step 2: Perform macOS-to-Windows acceptance**

Verify plain/secure Unicode and punctuation chat, a file larger than 32 MiB,
macOS terminal client to Windows PowerShell ConPTY, keyboard matrix, mouse
movement/click/wheel, all four edge directions, immediate edge return, cable
removal, emergency stop, and application exit.

- [ ] **Step 3: Perform Windows-to-macOS acceptance**

Repeat the same checks with Windows controlling macOS and Windows opening the
macOS login-shell PTY. Confirm macOS permission denial does not crash the app.

- [ ] **Step 4: Remove the legacy implementation only after both directions pass**

Delete `serial_text_chat.py`, rename the launchers, remove Tkinter/pynput
dependencies, and update every README command to `python -m shooklink` or the
new launcher name.

- [ ] **Step 5: Run final verification and commit**

```bash
python3 -m compileall -q shooklink
QT_QPA_PLATFORM=offscreen python3 -m pytest -q
git diff --check
git add -A
git commit -m "chore: retire legacy serial application"
```
