# ShookLink Local Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local CLI that delegates file and text sends to an already-running ShookLink GUI and waits for peer-confirmed completion.

**Architecture:** A focused `shooklink.local_control` package owns a loopback JSON server, endpoint discovery, and CLI. `app.py` only starts and closes the server with the GUI lifecycle, while existing chat/file services remain the serial protocol implementation.

**Tech Stack:** Python standard-library sockets/JSON/threading, existing ShookLink services, pytest

---

### Task 1: Local Control Server

**Files:**
- Create: `shooklink/local_control/server.py`
- Create: `shooklink/local_control/__init__.py`
- Test: `tests/local_control/test_server.py`

- [ ] Write failing tests for authenticated status, chat delivery/failure, file completion/failure, endpoint permissions, and cleanup.
- [ ] Run `python3 -m pytest tests/local_control/test_server.py -q` and confirm imports or assertions fail because the server does not exist.
- [ ] Implement the loopback server, request validation, service callback waiting, timeout handling, endpoint publication, and cleanup.
- [ ] Run `python3 -m pytest tests/local_control/test_server.py -q` and confirm all server tests pass.

### Task 2: Local Control Client And CLI

**Files:**
- Create: `shooklink/local_control/client.py`
- Create: `shooklink/local_control/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/local_control/test_client.py`
- Test: `tests/local_control/test_cli.py`

- [ ] Write failing tests for endpoint parsing, JSON request/response behavior, command arguments, success output, and non-zero error exits.
- [ ] Run `python3 -m pytest tests/local_control/test_client.py tests/local_control/test_cli.py -q` and confirm the missing implementation fails.
- [ ] Implement the client, argparse commands, module entry point, and `shooklinkctl` console script.
- [ ] Run `python3 -m pytest tests/local_control/test_client.py tests/local_control/test_cli.py -q` and confirm all client/CLI tests pass.

### Task 3: GUI Lifecycle Integration

**Files:**
- Modify: `shooklink/app.py`
- Modify: `tests/test_app.py`
- Modify: `README.md`

- [ ] Write a failing application test proving local control starts after core creation and closes during application shutdown.
- [ ] Run `python3 -m pytest tests/test_app.py -q` and confirm the lifecycle assertion fails.
- [ ] Start `LocalControlServer` from the application data directory, close it before the core, and document all four CLI commands and peer compatibility.
- [ ] Run `python3 -m pytest tests/test_app.py tests/local_control -q` and confirm integration tests pass.

### Task 4: Verification And Delivery

**Files:**
- Verify all modified and created files.

- [ ] Run `python3 -m pytest -q` and confirm the complete suite passes.
- [ ] Run `python3 -m compileall -q shooklink` and confirm it exits successfully.
- [ ] Restart ShookLink, run `python3 -m shooklink.local_control.cli status`, and verify the live GUI answers without releasing the serial port.
- [ ] Send a small file and plain/secure text through the live CLI and verify peer-confirmed success.
- [ ] Commit the implementation and push `codex/remote-shell-barrier-mouse-impl` without force.
