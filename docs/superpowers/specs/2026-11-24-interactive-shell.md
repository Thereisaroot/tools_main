# Interactive terminal on slow serial links

Live keys remain the default: send each keystroke immediately, let the remote
PTY own echo and cursor position, and never replay command input. The optional
bottom line editor sends a command only when the user presses Enter. Selecting
past output is local and freezes painting, not the terminal parser. A short
frame timer groups output bursts; update dirty rows in contiguous style runs
and map terminal cells to UTF-16 offsets for the Qt cursor.

Read at least one byte when a physical UART is idle, then drain available
bytes up to the existing buffer size. Keep the existing baud settings.
Only paired peers advertising `shell_output_zlib_v1` compress output. Each
frame stands alone, is compressed only for a net saving, then authenticated
and encrypted as before. Validate the declared raw size, stream termination
and the existing 48 KiB output limit before rendering.

A valid output frame can confirm the caller's own pending session if its
accept was lost. A legacy caller gets a second accept with the first output
if it hasn't already sent input or a resize. An unanswered request times out;
denial, timeout and an exited session do not accept keyboard input. Shell
authorization is unchanged.
