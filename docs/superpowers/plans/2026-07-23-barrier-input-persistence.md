# Barrier-Aligned Input and Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize bidirectional pointer sharing with Barrier-style anchored capture, deterministic edge switching, and persisted local authorizations.

**Architecture:** Native backends convert physical pointer motion into relative deltas around a stable monitor-center anchor. `InputService` maps those deltas into a peer-side absolute logical pointer and owns a cancellable edge-delay timer. Existing JSON settings remain the single atomic persistence source and are applied before UI construction.

**Tech Stack:** Python 3.11+, PySide6, Win32 low-level hooks and `SendInput`, macOS Quartz event taps, pytest/pytest-qt.

---

### Task 1: Persist Local Authorizations

**Files:**
- Modify: `tests/test_settings.py`
- Modify: `tests/test_app.py`
- Modify: `tests/ui/test_main_window.py`
- Modify: `shooklink/settings.py`
- Modify: `shooklink/app.py`
- Modify: `shooklink/ui/main_window.py`

- [ ] **Step 1: Write failing settings and UI tests**

Require `allow_remote_shell=True` and `allow_input=True` to survive a settings
round trip, reject non-boolean persisted values, expose both values from
`persistent_preferences()`, initialize both checkboxes from service state, and
emit a preference-change signal after a successful toggle.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest -q tests/test_settings.py tests/test_app.py tests/ui/test_main_window.py`

Expected: failures show that authorization values are forced to `False`, the UI
snapshot omits them, and startup does not apply them.

- [ ] **Step 3: Implement minimal persistence wiring**

Load and save validated booleans without overriding them. Extend the UI snapshot
to `(port, baud, peer_side, auto_edge, allow_shell, allow_input)`, add a
`preferences_changed` signal, and save immediately from
`ApplicationController`. Before constructing `MainWindow`, apply settings with:

```python
core.shell.set_allow_remote_shell(settings.allow_remote_shell)
if core.input is not None:
    core.input.set_allow_remote_input(settings.allow_input)
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest -q tests/test_settings.py tests/test_app.py tests/ui/test_main_window.py`

- [ ] **Step 5: Commit**

Commit message: `feat: persist local remote authorizations`

### Task 2: Correct Physical Outer-Edge Geometry

**Files:**
- Modify: `tests/input/test_topology.py`
- Modify: `shooklink/input/topology.py`

- [ ] **Step 1: Write failing stair-step edge tests**

For staggered monitors, require the exposed right edge of each cross-axis span
to be switchable while rejecting the shared internal seam and empty gaps.

- [ ] **Step 2: Run the topology tests and verify RED**

Run: `python3 -m pytest -q tests/input/test_topology.py`

Expected: the exposed edge of the non-extreme monitor is missing.

- [ ] **Step 3: Build skyline edge segments**

Partition the cross-axis by all monitor starts and ends. For each interval,
choose the outermost monitor boundary in the requested direction, omit intervals
with no monitor, and merge adjacent collinear segments.

- [ ] **Step 4: Run topology and logical-pointer tests**

Run: `python3 -m pytest -q tests/input/test_topology.py tests/input/test_pointer.py`

- [ ] **Step 5: Commit**

Commit message: `fix: map real multi-monitor outer edges`

### Task 3: Replace Event-Count Edge Hold With a Timer

**Files:**
- Modify: `tests/input/test_service.py`
- Modify: `shooklink/input/service.py`

- [ ] **Step 1: Write failing timer tests**

Cover one-shot entry after 0.5 seconds without a second coordinate change,
cancellation when the cursor leaves the edge, no entry after disconnect or a
state transition, and timer cleanup on close.

- [ ] **Step 2: Run focused service tests and verify RED**

Run: `python3 -m pytest -q tests/input/test_service.py -k 'auto_edge or edge_timer'`

- [ ] **Step 3: Implement a cancellable edge timer**

Start a daemon `threading.Timer` only when idle, eligible, at the configured
edge, and moving outward. Keep zero-motion events from cancelling an active
hold. Cancel when movement is inward, the pointer leaves the edge, settings or
connection state changes, or a session starts. At expiry, re-check state,
topology, position, and configured side before `_request_control()`.

- [ ] **Step 4: Run all service tests and verify GREEN**

Run: `python3 -m pytest -q tests/input/test_service.py`

- [ ] **Step 5: Commit**

Commit message: `fix: make automatic edge switching deterministic`

### Task 4: Harden Windows Barrier-Style Anchored Capture

**Files:**
- Modify: `tests/input/test_windows_backend.py`
- Modify: `shooklink/input/windows_backend.py`

- [ ] **Step 1: Write failing warp-artifact and restoration tests**

Require a recenter event to produce no forwarded movement, discard an
implausible edge-to-center warp delta, and retain enough anchor state for safe
capture cleanup.

- [ ] **Step 2: Run Windows backend tests and verify RED**

Run: `python3 -m pytest -q tests/input/test_windows_backend.py`

- [ ] **Step 3: Implement bounded anchor handling**

Track the anchor monitor bounds, recognize the exact anchor event, filter
Barrier-style half-screen warp artifacts, and re-anchor every valid physical
packet without forwarding synthetic motion.

- [ ] **Step 4: Run Windows tests and verify GREEN**

Run: `python3 -m pytest -q tests/input/test_windows_backend.py`

- [ ] **Step 5: Commit**

Commit message: `fix: filter Windows pointer warp artifacts`

### Task 5: Port Anchored Capture to macOS

**Files:**
- Modify: `tests/input/test_macos_backend.py`
- Modify: `shooklink/input/macos_backend.py`

- [ ] **Step 1: Write failing macOS pointer tests**

Require suppressed capture to choose the current monitor center, derive delta
from the Quartz event location relative to that anchor, recenter every packet,
ignore self-injected movement, and use the HID event-tap location used by
Barrier.

- [ ] **Step 2: Run macOS backend tests and verify RED**

Run: `python3 -m pytest -q tests/input/test_macos_backend.py`

- [ ] **Step 3: Implement the macOS anchor path**

Initialize anchor state before the event-tap thread, use
`CGEventGetLocation(event)` for suppressed move/drag events, warp back to the
anchor after extracting the delta, and preserve local non-suppressed behavior
for idle edge observation.

- [ ] **Step 4: Run macOS tests and verify GREEN**

Run: `python3 -m pytest -q tests/input/test_macos_backend.py`

- [ ] **Step 5: Commit**

Commit message: `fix: anchor macOS remote pointer capture`

### Task 6: Clarify UI Semantics and Verify Regression

**Files:**
- Modify: `tests/ui/test_main_window.py`
- Modify: `shooklink/ui/main_window.py`
- Modify: `README.md`

- [ ] **Step 1: Add a failing UI-label assertion**

Require visible text explaining that auto edge is local to this computer and
that the peer only needs remote-input authorization for one-way initiation.

- [ ] **Step 2: Add concise UI and README guidance**

Use `Auto Edge (this computer -> peer)` and document mirrored peer-side settings
for bidirectional initiation.

- [ ] **Step 3: Run complete verification**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest -q`

Run: `python3 -m compileall -q shooklink`

Run: `git diff --check`

- [ ] **Step 4: Commit**

Commit message: `docs: clarify local auto-edge direction`
