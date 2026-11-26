# Interactive keyboard delivery

With `shell_keys_v1` on both trusted peers, the first key is dispatched without
waiting for an echo. Short bursts can share a packet (up to 4 ms); later keys
carry the unconfirmed prefix so a missing key can recover on the next keystroke.
An idle retry covers the last key as well. Bulk paste uses bounded chunks and a
1 MiB memory limit. Retry timing accounts for the serial baud rate.

The executing peer writes only the unseen contiguous suffix to its PTY and
acknowledges its byte offset after the write. The ack is not a statement that
the shell finished executing a command. Old, duplicate and gapped packets do
not execute input twice. An uncertain PTY write, prolonged lack of progress,
or input overflow closes that shell rather than resending the command in a
fresh session. Pending input is never persisted or logged.

Legacy peers use the previous single-send input protocol; no automatic replay
is safe without receiver-side deduplication. Screen output is separate from
input acknowledgements: live mode still shows the remote PTY's actual echo
(never a local copy of potentially hidden password input). For immediate
local line editing choose **Edit command locally** rather than live keys.
