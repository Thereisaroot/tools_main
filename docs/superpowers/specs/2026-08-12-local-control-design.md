# ShookLink Local Control Design

## Goal

Allow a local command-line process to ask an already-running ShookLink GUI to
send a file or text without taking ownership of the serial port or automating
the GUI.

## Architecture

The GUI starts a loopback-only TCP control server after constructing the core.
It writes an atomic endpoint file in the existing application data directory.
The endpoint contains the selected loopback port and a random per-process token.
The CLI reads this file, authenticates with the token, sends one newline-delimited
JSON request, and waits for one newline-delimited JSON response.

The control server calls the existing `ChatService` and `FileService`; it does
not add or change serial protocol messages. The server removes its endpoint file
when the GUI exits. A stale endpoint or unreachable process is reported as a
local control error.

## Commands

- `status`: reports GUI connection state, peer trust, and advertised features.
- `send-file`: validates an absolute or relative local path and waits for the
  receiver's final file-completion acknowledgement.
- `send-plain`: sends UTF-8 text and waits for peer chat acknowledgement.
- `send-secure`: requires trust, sends authenticated text, and waits for peer
  chat acknowledgement.

The installed command is `shooklinkctl`; `python -m shooklink.local_control.cli`
provides the same interface. Success exits with status 0 and prints a concise
result. Invalid input, disconnected peers, delivery failures, timeouts, and
unavailable GUI processes exit non-zero with an actionable message.

## Safety And Concurrency

The server binds only to `127.0.0.1`. Every request must contain the random
token from the endpoint file. On POSIX systems the endpoint file is mode `0600`;
the token is regenerated for every GUI launch. Requests run in independent
handler threads, while the existing service locks remain responsible for
serial and transfer concurrency. Control shutdown stops accepting requests,
waits for handlers briefly, and removes only the endpoint it created.

## Compatibility

The peer does not need an update because the wire protocol is unchanged. A
computer needs the update only when its local GUI should accept `shooklinkctl`
commands.

## Tests

- Endpoint files are atomic, private on POSIX, and removed on close.
- Missing or incorrect tokens are rejected.
- Status reflects the current core snapshot.
- Plain and secure sends complete only on delivery callback and surface failure.
- File sends complete only on terminal `complete` progress and surface terminal
  failure or timeout.
- CLI request encoding, success output, and error exit status are covered.
- Existing application, chat, file, and integration tests remain green.
