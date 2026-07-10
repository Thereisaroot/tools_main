from shooklink.protocol.messages import Message, MessageType
from shooklink.shell.service import ShellService
from shooklink.transport.multiplexer import Priority


class FakeBus:
    def __init__(self, trusted=True, authenticated=True):
        self.trusted = trusted
        self.authenticated = authenticated
        self.sent = []

    def send(self, message, *, secure=True, priority=Priority.NORMAL):
        self.sent.append((message, secure, priority))

    def decrypt_secure(self, message):
        if not self.authenticated:
            raise RuntimeError("authentication failed")
        return message.body


class FakeProcess:
    def __init__(self, on_output, on_exit):
        self.on_output = on_output
        self.on_exit = on_exit
        self.started = []
        self.writes = []
        self.resizes = []
        self.terminate_calls = 0
        self.running = False

    def start(self, columns, rows):
        self.started.append((columns, rows))
        self.running = True

    def write(self, data):
        self.writes.append(data)

    def resize(self, columns, rows):
        self.resizes.append((columns, rows))

    def terminate(self):
        self.terminate_calls += 1
        self.running = False

    def is_running(self):
        return self.running

    def emit_output(self, data):
        self.on_output(data)

    def emit_exit(self, exit_code):
        self.running = False
        self.on_exit(exit_code)


class FakeProcessFactory:
    def __init__(self):
        self.processes = []
        self.terms = []

    def __call__(self, on_output, on_exit, term="xterm-256color"):
        process = FakeProcess(on_output, on_exit)
        self.processes.append(process)
        self.terms.append(term)
        return process


def shell_message(message_type, session_id="a" * 32, body=b"", **metadata):
    if message_type is MessageType.SHELL_OPEN:
        metadata.setdefault("utf8", True)
    return Message(message_type, {"session_id": session_id, **metadata}, body)


def test_incoming_shell_is_denied_until_local_permission_is_enabled():
    bus = FakeBus()
    factory = FakeProcessFactory()
    service = ShellService(bus, factory)

    service.handle_message(
        shell_message(MessageType.SHELL_OPEN, columns=80, rows=24, term="xterm-256color")
    )

    assert factory.processes == []
    denial, secure, priority = bus.sent[-1]
    assert denial.message_type is MessageType.SHELL_DENY
    assert denial.metadata["reason"] == "permission"
    assert secure is True
    assert priority is Priority.INTERACTIVE


def test_authorized_incoming_shell_forwards_io_resize_and_exit():
    bus = FakeBus()
    factory = FakeProcessFactory()
    service = ShellService(bus, factory)
    service.set_allow_remote_shell(True)
    session_id = "b" * 32

    service.handle_message(
        shell_message(
            MessageType.SHELL_OPEN,
            session_id,
            columns=100,
            rows=40,
            term="xterm-256color",
        )
    )
    process = factory.processes[-1]
    service.handle_message(shell_message(MessageType.SHELL_INPUT, session_id, b"dir\r"))
    service.handle_message(
        shell_message(MessageType.SHELL_RESIZE, session_id, columns=120, rows=50)
    )
    process.emit_output(b"output")
    process.emit_exit(7)

    assert process.started == [(100, 40)]
    assert process.writes == [b"dir\r"]
    assert process.resizes == [(120, 50)]
    sent_types = [item[0].message_type for item in bus.sent]
    assert sent_types == [
        MessageType.SHELL_ACCEPT,
        MessageType.SHELL_OUTPUT,
        MessageType.SHELL_EXIT,
    ]
    assert all(item[1] is True for item in bus.sent)


def test_disabling_permission_terminates_the_executing_shell():
    bus = FakeBus()
    factory = FakeProcessFactory()
    service = ShellService(bus, factory)
    service.set_allow_remote_shell(True)
    service.handle_message(
        shell_message(MessageType.SHELL_OPEN, columns=80, rows=24, term="xterm")
    )
    process = factory.processes[-1]

    service.set_allow_remote_shell(False)

    assert process.terminate_calls == 1
    assert bus.sent[-1][0].message_type is MessageType.SHELL_EXIT


def test_requester_session_receives_output_and_sends_interactive_input():
    bus = FakeBus()
    service = ShellService(bus, FakeProcessFactory())
    outputs = []
    service.add_output_listener(outputs.append)

    session_id = service.open_remote(columns=80, rows=24)
    service.handle_message(shell_message(MessageType.SHELL_ACCEPT, session_id))
    service.handle_message(
        shell_message(MessageType.SHELL_OUTPUT, session_id, b"remote output")
    )
    service.send_input(session_id, b"echo ok\r")
    service.resize(session_id, 90, 30)

    assert outputs[-1].data == b"remote output"
    assert outputs[-1].session_id == session_id
    assert bus.sent[-2] == (
        shell_message(MessageType.SHELL_INPUT, session_id, b"echo ok\r"),
        True,
        Priority.INTERACTIVE,
    )
    assert bus.sent[-1][0].message_type is MessageType.SHELL_RESIZE


def test_shell_messages_rejected_by_authentication_do_not_spawn_process():
    bus = FakeBus(authenticated=False)
    factory = FakeProcessFactory()
    service = ShellService(bus, factory)
    service.set_allow_remote_shell(True)

    assert not service.handle_message(
        shell_message(MessageType.SHELL_OPEN, columns=80, rows=24, term="xterm")
    )
    assert factory.processes == []


def test_only_one_shell_session_is_allowed_per_peer():
    bus = FakeBus()
    factory = FakeProcessFactory()
    service = ShellService(bus, factory)
    service.set_allow_remote_shell(True)
    service.handle_message(
        shell_message(MessageType.SHELL_OPEN, "1" * 32, columns=80, rows=24, term="xterm")
    )

    service.handle_message(
        shell_message(MessageType.SHELL_OPEN, "2" * 32, columns=80, rows=24, term="xterm")
    )

    assert len(factory.processes) == 1
    assert bus.sent[-1][0].message_type is MessageType.SHELL_DENY
    assert bus.sent[-1][0].metadata["reason"] == "busy"


def test_shell_state_listeners_receive_request_active_and_exit():
    bus = FakeBus()
    service = ShellService(bus, FakeProcessFactory())
    states = []
    service.add_state_listener(states.append)

    session_id = service.open_remote(columns=80, rows=24)
    service.handle_message(shell_message(MessageType.SHELL_ACCEPT, session_id))
    service.handle_message(
        shell_message(MessageType.SHELL_EXIT, session_id, exit_code=0)
    )

    assert [(state.state, state.session_id) for state in states] == [
        ("requesting", session_id),
        ("active", session_id),
        ("exited", session_id),
    ]
    assert states[-1].exit_code == 0


def test_permission_revocation_during_process_creation_prevents_start():
    bus = FakeBus()
    holder = {}

    class RevokingFactory(FakeProcessFactory):
        def __call__(self, on_output, on_exit, term="xterm"):
            process = super().__call__(on_output, on_exit, term)
            holder["service"].set_allow_remote_shell(False)
            return process

    factory = RevokingFactory()
    service = ShellService(bus, factory)
    holder["service"] = service
    service.set_allow_remote_shell(True)

    service.handle_message(
        shell_message(MessageType.SHELL_OPEN, columns=80, rows=24, term="xterm")
    )

    assert factory.processes[-1].started == []
    assert bus.sent[-1][0].message_type is MessageType.SHELL_DENY
    assert bus.sent[-1][0].metadata["reason"] == "permission"


def test_permission_revocation_during_accept_send_prevents_start():
    holder = {}

    class RevokingAcceptBus(FakeBus):
        def send(self, message, *, secure=True, priority=Priority.NORMAL):
            super().send(message, secure=secure, priority=priority)
            if message.message_type is MessageType.SHELL_ACCEPT:
                holder["service"].set_allow_remote_shell(False)

    bus = RevokingAcceptBus()
    factory = FakeProcessFactory()
    service = ShellService(bus, factory)
    holder["service"] = service
    service.set_allow_remote_shell(True)

    service.handle_message(
        shell_message(MessageType.SHELL_OPEN, columns=80, rows=24, term="xterm")
    )

    assert factory.processes[-1].started == []
    assert service.active_session_id is None
    assert [item[0].message_type for item in bus.sent] == [
        MessageType.SHELL_ACCEPT,
        MessageType.SHELL_EXIT,
    ]


def test_shell_accept_precedes_output_emitted_synchronously_by_start():
    bus = FakeBus()

    class EagerFactory(FakeProcessFactory):
        def __call__(self, on_output, on_exit, term="xterm"):
            process = super().__call__(on_output, on_exit, term)
            original_start = process.start

            def start(columns, rows):
                original_start(columns, rows)
                process.emit_output(b"initial prompt")

            process.start = start
            return process

    service = ShellService(bus, EagerFactory())
    service.set_allow_remote_shell(True)

    service.handle_message(
        shell_message(MessageType.SHELL_OPEN, columns=80, rows=24, term="xterm")
    )

    assert [item[0].message_type for item in bus.sent] == [
        MessageType.SHELL_ACCEPT,
        MessageType.SHELL_OUTPUT,
    ]


def test_initial_resize_is_queued_while_shell_request_is_pending():
    bus = FakeBus()
    service = ShellService(bus, FakeProcessFactory())
    session_id = service.open_remote(columns=80, rows=24)

    service.resize(session_id, 120, 40)
    service.handle_message(shell_message(MessageType.SHELL_ACCEPT, session_id))

    assert bus.sent[0][0].metadata["utf8"] is True
    assert bus.sent[-1][0] == shell_message(
        MessageType.SHELL_RESIZE,
        session_id,
        columns=120,
        rows=40,
    )


def test_negotiated_terminal_type_reaches_process_factory():
    bus = FakeBus()
    factory = FakeProcessFactory()
    service = ShellService(bus, factory)
    service.set_allow_remote_shell(True)

    service.handle_message(
        shell_message(
            MessageType.SHELL_OPEN,
            columns=80,
            rows=24,
            term="screen-256color",
        )
    )

    assert factory.terms == ["screen-256color"]


def test_large_shell_input_is_split_to_frame_safe_chunks():
    bus = FakeBus()
    service = ShellService(bus, FakeProcessFactory())
    session_id = service.open_remote(columns=80, rows=24)
    service.handle_message(shell_message(MessageType.SHELL_ACCEPT, session_id))
    bus.sent.clear()

    service.send_input(session_id, b"x" * 100_000)

    assert len(bus.sent) == 3
    assert all(len(item[0].body) <= 48 * 1024 for item in bus.sent)
    assert b"".join(item[0].body for item in bus.sent) == b"x" * 100_000
