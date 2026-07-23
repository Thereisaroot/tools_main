# Barrier-Aligned Input and Persistent Authorization Design

## Goal

Make cross-platform pointer sharing follow Barrier's proven capture model, make
automatic edge switching deterministic, and persist the two local `Allow`
authorizations across application restarts.

## Input Capture

The controlling computer keeps its native pointer at a stable anchor while the
remote pointer is active. Windows uses low-level hook coordinates relative to a
monitor-center anchor and recenters after each physical movement. macOS uses a
Quartz HID event tap, calculates movement from the event location relative to a
monitor-center anchor, and recenters after each movement. Both backends ignore
the synthetic recenter event and reject implausible half-screen warp deltas.

The controller continues to maintain an absolute logical pointer in the peer's
native virtual-desktop coordinate space. Coalesced wire messages contain that
absolute coordinate, and the receiver injects it directly. The receiver
acknowledges the commanded coordinate instead of immediately reading a
potentially stale OS cursor position.

## Edge Switching

Automatic entry is a local controller preference. Reaching the configured
physical outer edge in the outward direction starts a 0.5 second one-shot
timer. Leaving that edge or changing state cancels the timer. At expiry, the
cursor position and session eligibility are checked again before requesting
control. This matches Barrier's delayed jump-zone behavior and does not depend
on receiving repeated coordinate changes while the OS pointer is clamped.

Returning from the peer during an active session remains automatic and does not
depend on the peer's auto-edge preference. Crossing the source-facing edge of
the remote topology ends the active session and restores the source pointer one
pixel inside its original edge.

Outer-edge geometry is calculated per cross-axis span. Internal seams and empty
desktop gaps are not switchable, while exposed stair-step edges of real
monitors remain valid.

## Persisted Settings

`Allow Remote Input` and `Allow Remote Shell` are stored as local booleans in
the existing atomic settings file. A successful UI change saves immediately,
and normal shutdown saves the complete current preference snapshot again.
Startup applies the stored authorizations to the services before constructing
the window so checkboxes and service state agree.

The authorization remains local: no protocol message can enable either option
on the peer. Disabling an option still terminates the matching incoming input or
shell session.

## UI Semantics

The input section explains that auto edge applies from this computer to the
peer. One physical controller only needs auto edge enabled on its own computer;
the receiver needs `Allow Remote Input`. Bidirectional initiation requires auto
edge on both computers with mirrored peer sides.

## Verification

Automated tests cover settings round-trip and immediate persistence, startup
restoration, Windows and macOS anchor behavior, synthetic warp filtering,
timer-based edge entry and cancellation, stair-step monitor edges, pointer
return, and existing protocol/session regressions. The complete test suite,
bytecode compilation, and diff checks must pass before completion.
