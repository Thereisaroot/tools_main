from __future__ import annotations

import threading
import time

import pytest

from shooklink.input.backend import BaseInputBackend, PermissionStatus
from shooklink.input.events import (
    KeyAction,
    KeyEvent,
    KeyLocation,
    Modifiers,
    MouseButton,
    MouseButtonEvent,
    PointerMotionEvent,
    PointerPositionEvent,
)
from shooklink.input.service import (
    EDGE_HOLD_SECONDS,
    MOTION_INTERVAL_SECONDS,
    InputService,
    InputSessionState,
    InputUnavailable,
)
from shooklink.input.topology import Monitor, Rect, Side
from shooklink.protocol.messages import Message, MessageType
from shooklink.transport.multiplexer import Priority


LOCAL_SESSION = "1" * 32
REMOTE_SESSION = "2" * 32


def _capture_error(callback, errors):
    try:
        callback()
    except BaseException as error:
        errors.append(error)


class FakeClock:
    def __init__(self, value=10.0):
        self.value = value

    def __call__(self):
        return self.value


def test_input_service_types_are_exported_from_input_package():
    from shooklink.input import InputService as ExportedService
    from shooklink.input import InputSessionState as ExportedState

    assert ExportedService is InputService
    assert ExportedState is InputSessionState


def test_disconnected_input_service_can_bind_authenticated_peer_before_connect():
    service = InputService(
        FakeBus(),
        FakeBackend(),
        local_peer_id="peer-a",
        peer_id="peer-pending",
        connected=False,
    )

    service.set_peer_id("peer-b")
    service.connection_changed(True)
    service.set_allow_remote_input(True)

    assert service.handle_message(input_request(controller_id="peer-b"))
    assert service.state is InputSessionState.BEING_CONTROLLED


def test_input_peer_id_cannot_change_while_connected_or_active():
    service = InputService(
        FakeBus(),
        FakeBackend(),
        local_peer_id="peer-a",
        peer_id="peer-b",
    )

    with pytest.raises(InputUnavailable, match="disconnected"):
        service.set_peer_id("peer-c")


class FakeBus:
    def __init__(self, *, trusted=True):
        self.trusted = trusted
        self.sent = []

    def send(self, message, *, secure=True, priority=Priority.NORMAL):
        self.sent.append((message, secure, priority))

    def decrypt_secure(self, message):
        return message.body


class FailingBus(FakeBus):
    def __init__(self, *, fail_types=()):
        super().__init__()
        self.fail_types = set(fail_types)

    def send(self, message, *, secure=True, priority=Priority.NORMAL):
        if message.message_type in self.fail_types:
            raise RuntimeError(f"cannot send {message.message_type.name}")
        super().send(message, secure=secure, priority=priority)


class FakeBackend(BaseInputBackend):
    def __init__(self, monitors=None, position=(99, 50)):
        super().__init__()
        self._monitors = (
            (Monitor("local", Rect(0, 0, 100, 100)),)
            if monitors is None
            else monitors
        )
        self.monitor_calls = 0
        self.position = position
        self.capture_starts = []
        self.capture_stops = 0
        self.native_injected = []
        self.warps = []
        self.capture_allowed = True
        self.inject_allowed = True
        self.emergency_callback = None

    def permission_status(self):
        return PermissionStatus(
            self.capture_allowed,
            self.inject_allowed,
            "ready" if self.capture_allowed and self.inject_allowed else "permission required",
        )

    def monitors(self):
        self.monitor_calls += 1
        return self._monitors

    def cursor_position(self):
        return self.position

    def warp_cursor(self, x, y):
        self.position = (x, y)
        self.warps.append((x, y))

    def _start_native_capture(self, suppress):
        self.capture_starts.append(suppress)
        self.emergency_callback = self._emergency_callback

    def _stop_native_capture(self):
        self.capture_stops += 1

    def _inject_native(self, event):
        self.native_injected.append(event)
        if isinstance(event, PointerPositionEvent):
            self.position = event.position

    def capture(self, event):
        self.emit_captured(event)

    def emergency_stop(self):
        assert self.emergency_callback is not None
        self.emergency_callback("stop")

    def emergency_exit(self):
        assert self.emergency_callback is not None
        self.emergency_callback("exit")


class ManualTimer:
    def __init__(self, delay, callback):
        self.delay = delay
        self.callback = callback
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.callback()


def install_manual_timers(monkeypatch):
    timers = []

    def factory(delay, callback):
        timer = ManualTimer(delay, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr("shooklink.input.service.threading.Timer", factory)
    return timers


def monitors_metadata(*rectangles):
    return [
        {
            "id": f"display-{index}",
            "x": rectangle.x,
            "y": rectangle.y,
            "width": rectangle.width,
            "height": rectangle.height,
        }
        for index, rectangle in enumerate(rectangles)
    ]


def input_request(
    session_id=REMOTE_SESSION,
    *,
    controller_id="peer-b",
    side="right",
    rectangles=(Rect(0, 0, 200, 100),),
):
    return Message(
        MessageType.INPUT_REQUEST,
        {
            "session_id": session_id,
            "controller_id": controller_id,
            "side": side,
            "monitors": monitors_metadata(*rectangles),
        },
    )


def input_accept(
    session_id,
    *,
    peer_id="peer-b",
    rectangles=(Rect(0, 0, 200, 100),),
    cursor=(0, 50),
):
    return Message(
        MessageType.INPUT_ACCEPT,
        {
            "session_id": session_id,
            "peer_id": peer_id,
            "monitors": monitors_metadata(*rectangles),
            "x": cursor[0],
            "y": cursor[1],
        },
    )


def start_controlling(*, clock=None, auto_edge=False, remote_cursor=(0, 50)):
    bus = FakeBus()
    backend = FakeBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        peer_side=Side.RIGHT,
        auto_edge_enabled=auto_edge,
        clock=clock or FakeClock(),
        session_factory=lambda: LOCAL_SESSION,
    )
    session_id = service.request_control()
    assert service.handle_message(
        input_accept(session_id, cursor=remote_cursor)
    )
    return service, bus, backend, session_id


def start_being_controlled():
    bus = FakeBus()
    backend = FakeBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
    )
    service.set_allow_remote_input(True)
    assert service.handle_message(input_request())
    return service, bus, backend


def test_outgoing_request_accepts_topology_and_enters_absolute_pointer_mode():
    bus = FakeBus()
    backend = FakeBackend(position=(99, 50))
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        peer_side=Side.RIGHT,
        clock=FakeClock(),
        session_factory=lambda: LOCAL_SESSION,
    )

    session_id = service.request_control()

    request, secure, priority = bus.sent[-1]
    assert session_id == LOCAL_SESSION
    assert service.state is InputSessionState.REQUESTING
    assert request.message_type is MessageType.INPUT_REQUEST
    assert request.metadata["side"] == "right"
    assert request.metadata["monitors"][0]["width"] == 100
    assert secure is True
    assert priority is Priority.INTERACTIVE

    assert service.handle_message(input_accept(session_id, cursor=(100, 50)))

    enter = bus.sent[-1][0]
    assert service.state is InputSessionState.CONTROLLING
    assert backend.capture_starts == [True]
    assert enter.message_type is MessageType.INPUT_ENTER
    assert (enter.metadata["x"], enter.metadata["y"]) == (100, 50)
    assert bus.sent[-1][2] is Priority.INTERACTIVE


@pytest.mark.parametrize("auto_edge_enabled", [False, True])
def test_manual_request_starts_at_the_remote_cursor_without_immediate_leave(
    auto_edge_enabled,
):
    service, bus, backend, _session_id = start_controlling(
        auto_edge=auto_edge_enabled,
        remote_cursor=(100, 50),
    )
    bus.sent.clear()

    try:
        backend.capture(PointerMotionEvent(-1, 0))

        assert service.state is InputSessionState.CONTROLLING
        assert all(
            item[0].message_type is not MessageType.INPUT_LEAVE
            for item in bus.sent
        )
    finally:
        service.close()


def test_cancelled_request_is_followed_by_stop_when_request_send_finishes_late():
    request_started = threading.Event()
    release_request = threading.Event()

    class BlockingRequestBus(FakeBus):
        def send(self, message, *, secure=True, priority=Priority.NORMAL):
            if message.message_type is MessageType.INPUT_REQUEST:
                request_started.set()
                assert release_request.wait(2)
            super().send(message, secure=secure, priority=priority)

    bus = BlockingRequestBus()
    service = InputService(
        bus,
        FakeBackend(),
        local_peer_id="peer-a",
        peer_id="peer-b",
        session_factory=lambda: LOCAL_SESSION,
    )
    errors = []
    requester = threading.Thread(
        target=lambda: _capture_error(service.request_control, errors)
    )
    requester.start()
    assert request_started.wait(1)

    service.stop_control(reason="cancelled")
    release_request.set()
    requester.join(2)

    assert not requester.is_alive()
    assert service.state is InputSessionState.IDLE
    assert isinstance(errors[0], InputUnavailable)
    assert [item[0].message_type for item in bus.sent][-2:] == [
        MessageType.INPUT_REQUEST,
        MessageType.INPUT_STOP,
    ]


def test_cancelled_old_request_cannot_stop_replacement_session_capture():
    old_request_started = threading.Event()
    release_old_request = threading.Event()

    class BlockingFirstRequestBus(FakeBus):
        def send(self, message, *, secure=True, priority=Priority.NORMAL):
            if (
                message.message_type is MessageType.INPUT_REQUEST
                and message.metadata["session_id"] == LOCAL_SESSION
            ):
                old_request_started.set()
                assert release_old_request.wait(2)
            super().send(message, secure=secure, priority=priority)

    session_ids = iter((LOCAL_SESSION, "3" * 32))
    bus = BlockingFirstRequestBus()
    backend = FakeBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        session_factory=lambda: next(session_ids),
    )
    old_errors = []
    old_requester = threading.Thread(
        target=lambda: _capture_error(service.request_control, old_errors)
    )
    old_requester.start()
    assert old_request_started.wait(1)
    service.stop_control(reason="cancelled")

    replacement_id = service.request_control()
    assert service.handle_message(input_accept(replacement_id))
    assert backend.capture_running is True
    release_old_request.set()
    old_requester.join(2)

    assert not old_requester.is_alive()
    assert isinstance(old_errors[0], InputUnavailable)
    assert service.state is InputSessionState.CONTROLLING
    assert service.active_session_id == replacement_id
    assert backend.capture_running is True
    service.stop_control(reason="test_complete")


def test_idle_refresh_rechecks_state_after_cancelled_request_cleanup_gap():
    old_request_started = threading.Event()
    release_old_request = threading.Event()
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    class BlockingFirstRequestBus(FakeBus):
        def send(self, message, *, secure=True, priority=Priority.NORMAL):
            if (
                message.message_type is MessageType.INPUT_REQUEST
                and message.metadata["session_id"] == LOCAL_SESSION
            ):
                old_request_started.set()
                assert release_old_request.wait(2)
            super().send(message, secure=secure, priority=priority)

    session_ids = iter((LOCAL_SESSION, "4" * 32))
    bus = BlockingFirstRequestBus()
    backend = FakeBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        session_factory=lambda: next(session_ids),
    )
    old_errors = []
    old_requester = threading.Thread(
        target=lambda: _capture_error(service.request_control, old_errors)
    )
    old_requester.start()
    assert old_request_started.wait(1)
    service.stop_control(reason="cancelled")
    original_refresh = service._refresh_idle_capture

    def delayed_refresh(*args, **kwargs):
        refresh_started.set()
        assert release_refresh.wait(2)
        return original_refresh(*args, **kwargs)

    service._refresh_idle_capture = delayed_refresh
    release_old_request.set()
    assert refresh_started.wait(1)

    replacement_id = service.request_control()
    assert service.handle_message(input_accept(replacement_id))
    assert backend.capture_running is True
    release_refresh.set()
    old_requester.join(2)

    assert not old_requester.is_alive()
    assert isinstance(old_errors[0], InputUnavailable)
    assert service.state is InputSessionState.CONTROLLING
    assert backend.capture_running is True
    service.stop_control(reason="test_complete")


def test_disconnected_service_rejects_buffered_incoming_request():
    service, bus, _backend = start_being_controlled()
    service.stop_control(reason="reset")
    service.disconnect()
    bus.sent.clear()

    assert not service.handle_message(input_request())

    assert service.state is InputSessionState.IDLE
    assert bus.sent == []


def test_late_incoming_accept_is_compensated_after_disconnect():
    accept_started = threading.Event()
    release_accept = threading.Event()

    class BlockingAcceptBus(FakeBus):
        def send(self, message, *, secure=True, priority=Priority.NORMAL):
            if message.message_type is MessageType.INPUT_ACCEPT:
                accept_started.set()
                assert release_accept.wait(2)
            super().send(message, secure=secure, priority=priority)

    bus = BlockingAcceptBus()
    service = InputService(
        bus,
        FakeBackend(),
        local_peer_id="peer-a",
        peer_id="peer-b",
    )
    service.set_allow_remote_input(True)
    receiver = threading.Thread(target=lambda: service.handle_message(input_request()))
    receiver.start()
    assert accept_started.wait(1)

    service.disconnect()
    release_accept.set()
    receiver.join(2)

    assert not receiver.is_alive()
    assert service.state is InputSessionState.IDLE
    assert [item[0].message_type for item in bus.sent][-2:] == [
        MessageType.INPUT_ACCEPT,
        MessageType.INPUT_STOP,
    ]


def test_allowed_incoming_request_accepts_and_injects_normalized_key():
    service, bus, backend = start_being_controlled()
    accept, secure, priority = bus.sent[-1]
    assert service.state is InputSessionState.BEING_CONTROLLED
    assert accept.message_type is MessageType.INPUT_ACCEPT
    assert accept.metadata["monitors"][0]["width"] == 100
    assert (accept.metadata["x"], accept.metadata["y"]) == (99, 50)
    assert secure is True
    assert priority is Priority.INTERACTIVE

    key = KeyEvent(
        KeyAction.DOWN,
        usage=0x38,
        scan_code=44,
        virtual_key=191,
        text="?",
        modifiers=Modifiers.SHIFT,
        location=KeyLocation.STANDARD,
        repeat=True,
        extended=False,
    )
    message = Message(
        MessageType.INPUT_KEY,
        {
            "session_id": REMOTE_SESSION,
            "action": key.action.value,
            "usage": key.usage,
            "scan_code": key.scan_code,
            "virtual_key": key.virtual_key,
            "text": key.text,
            "modifiers": int(key.modifiers),
            "location": key.location.value,
            "repeat": key.repeat,
            "extended": key.extended,
        },
    )

    assert service.handle_message(message)
    assert backend.native_injected[-1] == key


def test_denied_incoming_request_does_not_require_monitor_enumeration():
    bus = FakeBus()
    backend = FakeBackend(monitors=())
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
    )

    assert service.handle_message(input_request())

    busy = bus.sent[-1][0]
    assert busy.message_type is MessageType.INPUT_BUSY
    assert busy.metadata["reason"] == "permission"


def test_incoming_request_rolls_back_when_pressed_input_cleanup_fails():
    class FailingReleaseBackend(FakeBackend):
        def release_all(self):
            raise RuntimeError("release failed")

    bus = FakeBus()
    service = InputService(
        bus,
        FailingReleaseBackend(),
        local_peer_id="peer-a",
        peer_id="peer-b",
    )
    service.set_allow_remote_input(True)

    assert service.handle_message(input_request())

    assert service.state is InputSessionState.IDLE
    assert bus.sent[-1][0].message_type is MessageType.INPUT_BUSY
    assert bus.sent[-1][0].metadata["reason"] == "unavailable"


def test_simultaneous_requests_use_stable_peer_id_tie_breaker():
    lower_bus = FakeBus()
    lower = InputService(
        lower_bus,
        FakeBackend(),
        local_peer_id="peer-a",
        peer_id="peer-b",
        session_factory=lambda: LOCAL_SESSION,
    )
    lower.set_allow_remote_input(True)
    lower.request_control()
    lower_bus.sent.clear()

    assert lower.handle_message(input_request(controller_id="peer-b"))
    assert lower.state is InputSessionState.REQUESTING
    assert lower_bus.sent[-1][0].message_type is MessageType.INPUT_BUSY

    higher_bus = FakeBus()
    higher = InputService(
        higher_bus,
        FakeBackend(),
        local_peer_id="peer-b",
        peer_id="peer-a",
        session_factory=lambda: LOCAL_SESSION,
    )
    higher.set_allow_remote_input(True)
    higher.request_control()
    higher_bus.sent.clear()

    assert higher.handle_message(input_request(controller_id="peer-a"))
    assert higher.state is InputSessionState.BEING_CONTROLLED
    assert higher.active_session_id == REMOTE_SESSION
    assert higher_bus.sent[-1][0].message_type is MessageType.INPUT_ACCEPT


def test_stale_session_event_is_rejected_without_injection():
    service, _bus, backend = start_being_controlled()
    stale = Message(
        MessageType.INPUT_BUTTON,
        {
            "session_id": "f" * 32,
            "button": MouseButton.LEFT.value,
            "action": KeyAction.DOWN.value,
        },
    )

    assert service.handle_message(stale) is False
    assert backend.native_injected == []


def test_unknown_modifier_bits_are_rejected_before_native_injection():
    service, _bus, backend = start_being_controlled()
    message = Message(
        MessageType.INPUT_KEY,
        {
            "session_id": REMOTE_SESSION,
            "action": "down",
            "usage": 4,
            "scan_code": 30,
            "virtual_key": 65,
            "text": "a",
            "modifiers": 1 << 20,
            "location": "standard",
            "repeat": False,
            "extended": False,
        },
    )

    assert service.handle_message(message) is False
    assert backend.native_injected == []


def test_monitor_rectangle_must_stay_within_protocol_coordinate_bounds():
    bus = FakeBus()
    service = InputService(
        bus,
        FakeBackend(),
        local_peer_id="peer-a",
        peer_id="peer-b",
    )
    service.set_allow_remote_input(True)
    message = input_request()
    message.metadata["monitors"][0].update(x=10_000_000, width=2)

    assert service.handle_message(message) is False
    assert service.state is InputSessionState.IDLE
    assert bus.sent == []


def test_local_topology_cannot_exceed_wire_monitor_limit():
    monitors = tuple(
        Monitor(f"display-{index}", Rect(index * 10, 0, 10, 10))
        for index in range(33)
    )
    bus = FakeBus()
    service = InputService(
        bus,
        FakeBackend(monitors=monitors, position=(0, 0)),
        local_peer_id="peer-a",
        peer_id="peer-b",
    )

    try:
        service.request_control()
    except Exception as error:
        assert "monitor" in str(error).lower()
    else:
        raise AssertionError("oversized local topology was sent")
    assert bus.sent == []


def test_incoming_motion_reuses_session_topology_instead_of_enumerating_displays():
    service, _bus, backend = start_being_controlled()
    calls_after_accept = backend.monitor_calls
    assert service.handle_message(
        Message(
            MessageType.INPUT_ENTER,
            {"session_id": REMOTE_SESSION, "x": 0, "y": 50},
        )
    )
    for sequence in (1, 2):
        assert service.handle_message(
            Message(
                MessageType.INPUT_MOVE,
                {
                    "session_id": REMOTE_SESSION,
                    "motion_sequence": sequence,
                    "x": sequence,
                    "y": 50,
                },
            )
        )

    assert backend.monitor_calls == calls_after_accept


def test_out_of_order_incoming_motion_cannot_rewind_injected_pointer():
    service, _bus, backend = start_being_controlled()
    for sequence, x in ((2, 20), (1, 1)):
        assert service.handle_message(
            Message(
                MessageType.INPUT_MOVE,
                {
                    "session_id": REMOTE_SESSION,
                    "motion_sequence": sequence,
                    "x": x,
                    "y": 50,
                },
            )
        )

    assert backend.position == (20, 50)


def test_disconnect_releases_every_remotely_pressed_input():
    service, _bus, backend = start_being_controlled()
    assert service.handle_message(
        Message(
            MessageType.INPUT_KEY,
            {
                "session_id": REMOTE_SESSION,
                "action": "down",
                "usage": 4,
                "scan_code": 30,
                "virtual_key": 65,
                "text": "a",
                "modifiers": 0,
                "location": "standard",
                "repeat": False,
                "extended": False,
            },
        )
    )
    assert service.handle_message(
        Message(
            MessageType.INPUT_BUTTON,
            {
                "session_id": REMOTE_SESSION,
                "button": "left",
                "action": "down",
            },
        )
    )
    assert backend.pressed_keys == frozenset({4})
    assert backend.pressed_buttons == frozenset({MouseButton.LEFT})

    service.disconnect()

    assert service.state is InputSessionState.IDLE
    assert backend.pressed_keys == frozenset()
    assert backend.pressed_buttons == frozenset()
    assert [event.action for event in backend.native_injected[-2:]] == [
        KeyAction.UP,
        KeyAction.UP,
    ]


def test_manual_and_emergency_stop_release_remote_session():
    service, bus, backend, _session_id = start_controlling()
    bus.sent.clear()

    service.stop_control(reason="manual")

    assert service.state is InputSessionState.IDLE
    assert backend.capture_running is False
    assert [item[0].message_type for item in bus.sent] == [
        MessageType.INPUT_RELEASE_ALL,
        MessageType.INPUT_STOP,
    ]


def test_capture_stop_ignores_native_tail_event_before_idle_rearm(monkeypatch):
    timers = install_manual_timers(monkeypatch)

    class TailEventBackend(FakeBackend):
        def _stop_native_capture(self):
            super()._stop_native_capture()
            self.emit_captured(PointerMotionEvent(1, 0))

    bus = FakeBus()
    backend = TailEventBackend(position=(99, 50))
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        peer_side=Side.RIGHT,
        auto_edge_enabled=True,
        session_factory=lambda: LOCAL_SESSION,
    )
    session_id = service.request_control()
    assert service.handle_message(input_accept(session_id))
    timers.clear()

    service.stop_control(reason="manual")

    assert service.state is InputSessionState.IDLE
    assert backend.capture_running is True
    assert backend.capture_starts[-1] is False
    assert timers == []


def test_emergency_exit_releases_session_and_notifies_local_application():
    service, bus, backend, _session_id = start_controlling()
    exits = []
    service.add_emergency_listener(exits.append)
    bus.sent.clear()

    backend.emergency_exit()

    assert service.state is InputSessionState.IDLE
    assert backend.capture_running is False
    assert exits == ["exit"]
    assert [item[0].message_type for item in bus.sent] == [
        MessageType.INPUT_RELEASE_ALL,
        MessageType.INPUT_STOP,
    ]


def test_emergency_exit_is_available_during_idle_auto_edge_capture():
    bus = FakeBus()
    backend = FakeBackend(position=(99, 50))
    service = InputService(
        bus,
        backend,
        local_peer_id="local",
        peer_id="remote",
        peer_side=Side.RIGHT,
        auto_edge_enabled=True,
    )
    exits = []
    service.add_emergency_listener(exits.append)

    backend.emergency_exit()

    assert exits == ["exit"]
    service.close()

    service, bus, backend, _session_id = start_controlling()
    bus.sent.clear()
    backend.emergency_stop()

    assert service.state is InputSessionState.IDLE
    assert [item[0].message_type for item in bus.sent] == [
        MessageType.INPUT_RELEASE_ALL,
        MessageType.INPUT_STOP,
    ]


def test_auto_edge_hold_enters_and_remote_return_edge_leaves_without_dead_space(
    monkeypatch,
):
    timers = install_manual_timers(monkeypatch)
    clock = FakeClock(20.0)
    bus = FakeBus()
    backend = FakeBackend(position=(99, 50))
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        peer_side=Side.RIGHT,
        clock=clock,
        session_factory=lambda: LOCAL_SESSION,
    )
    service.set_auto_edge_enabled(True)
    assert backend.capture_starts == [False]

    backend.capture(PointerMotionEvent(2, 0))
    assert len(timers) == 1
    assert timers[0].delay == EDGE_HOLD_SECONDS
    assert timers[0].started is True
    timers[0].fire()

    assert service.state is InputSessionState.REQUESTING
    assert backend.capture_running is False
    assert service.handle_message(
        input_accept(LOCAL_SESSION, cursor=(100, 50))
    )
    enter = bus.sent[-1][0]
    assert enter.message_type is MessageType.INPUT_ENTER
    assert (enter.metadata["x"], enter.metadata["y"]) == (0, 50)
    bus.sent.clear()

    backend.capture(PointerMotionEvent(-1, 0))

    assert service.state is InputSessionState.IDLE
    assert backend.warps[-1] == (98, 50)
    assert [item[0].message_type for item in bus.sent] == [
        MessageType.INPUT_RELEASE_ALL,
        MessageType.INPUT_LEAVE,
    ]
    assert all(item[2] is Priority.INTERACTIVE for item in bus.sent)
    assert backend.capture_starts[-1] is False


def test_auto_edge_timer_survives_zero_motion_but_cancels_after_leaving_edge(
    monkeypatch,
):
    timers = install_manual_timers(monkeypatch)
    backend = FakeBackend(position=(99, 50))
    service = InputService(
        FakeBus(),
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        peer_side=Side.RIGHT,
    )
    service.set_auto_edge_enabled(True)

    backend.capture(PointerMotionEvent(1, 0))
    timer = timers[-1]
    backend.capture(PointerMotionEvent(0, 0))

    assert timer.cancelled is False

    backend.position = (98, 50)
    backend.capture(PointerMotionEvent(-1, 0))
    assert timer.cancelled is True

    timer.fire()
    assert service.state is InputSessionState.IDLE


def test_auto_edge_timer_is_cancelled_on_disconnect_and_cannot_fire_stale(
    monkeypatch,
):
    timers = install_manual_timers(monkeypatch)
    backend = FakeBackend(position=(99, 50))
    service = InputService(
        FakeBus(),
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        peer_side=Side.RIGHT,
    )
    service.set_auto_edge_enabled(True)
    backend.capture(PointerMotionEvent(1, 0))
    timer = timers[-1]

    service.disconnect()

    assert timer.cancelled is True
    timer.fire()
    assert service.state is InputSessionState.IDLE


def test_stale_auto_edge_request_rechecks_current_edge_and_setting():
    backend = FakeBackend(position=(98, 50))
    service = InputService(
        FakeBus(),
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        peer_side=Side.RIGHT,
        auto_edge_enabled=True,
    )

    with pytest.raises(InputUnavailable, match="edge"):
        service._request_control(enter_from_edge=True)

    backend.position = (99, 50)
    service.set_auto_edge_enabled(False)
    with pytest.raises(InputUnavailable, match="disabled"):
        service._request_control(enter_from_edge=True)

    assert service.state is InputSessionState.IDLE


def test_auto_edge_setting_rolls_back_when_idle_capture_cannot_start():
    class FailingStartBackend(FakeBackend):
        def _start_native_capture(self, suppress):
            raise RuntimeError("hook start failed")

    service = InputService(
        FakeBus(),
        FailingStartBackend(),
        local_peer_id="peer-a",
        peer_id="peer-b",
    )

    with pytest.raises(RuntimeError, match="hook start failed"):
        service.set_auto_edge_enabled(True)

    assert service.auto_edge_enabled is False
    assert service.state is InputSessionState.IDLE


def test_request_rolls_back_when_idle_capture_cannot_stop():
    class FailingStopBackend(FakeBackend):
        def _stop_native_capture(self):
            raise RuntimeError("hook stop failed")

    bus = FakeBus()
    backend = FailingStopBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        session_factory=lambda: LOCAL_SESSION,
    )
    service.set_auto_edge_enabled(True)

    with pytest.raises(RuntimeError, match="failed to stop"):
        service.request_control()

    assert service.state is InputSessionState.IDLE
    assert bus.sent == []


def test_pending_absolute_pointer_is_flushed_before_button_down():
    clock = FakeClock(30.0)
    service, bus, backend, _session_id = start_controlling(clock=clock)
    bus.sent.clear()

    backend.capture(PointerMotionEvent(5, 2))
    backend.capture(MouseButtonEvent(MouseButton.LEFT, KeyAction.DOWN))

    assert [item[0].message_type for item in bus.sent] == [
        MessageType.INPUT_MOVE,
        MessageType.INPUT_BUTTON,
    ]
    assert bus.sent[0][0].metadata["x"] == 5
    assert bus.sent[0][0].metadata["y"] == 52
    assert bus.sent[0][2] is Priority.INTERACTIVE
    assert bus.sent[1][2] is Priority.INTERACTIVE


def test_rate_flushed_pointer_cannot_be_overtaken_by_later_button():
    clock = FakeClock(30.0)
    service, bus, backend, _session_id = start_controlling(clock=clock)
    bus.sent.clear()
    clock.value += MOTION_INTERVAL_SECONDS * 2

    backend.capture(PointerMotionEvent(5, 2))
    backend.capture(MouseButtonEvent(MouseButton.LEFT, KeyAction.DOWN))

    assert [item[0].message_type for item in bus.sent] == [
        MessageType.INPUT_MOVE,
        MessageType.INPUT_BUTTON,
    ]
    assert [item[2] for item in bus.sent] == [
        Priority.INTERACTIVE,
        Priority.INTERACTIVE,
    ]


def test_stop_failure_still_releases_and_stops_remote_session():
    class FailingStopBackend(FakeBackend):
        def _stop_native_capture(self):
            raise RuntimeError("hook stop failed")

    bus = FakeBus()
    backend = FailingStopBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        session_factory=lambda: LOCAL_SESSION,
    )
    session_id = service.request_control()
    assert service.handle_message(input_accept(session_id))
    bus.sent.clear()

    with pytest.raises(RuntimeError, match="failed to stop"):
        service.stop_control(reason="manual")

    assert service.state is InputSessionState.IDLE
    assert [item[0].message_type for item in bus.sent] == [
        MessageType.INPUT_RELEASE_ALL,
        MessageType.INPUT_STOP,
    ]


def test_handler_send_failure_tears_down_incoming_session_and_releases_inputs():
    bus = FailingBus(fail_types={MessageType.INPUT_POINTER_STATE})
    backend = FakeBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
    )
    service.set_allow_remote_input(True)
    assert service.handle_message(input_request())
    backend.inject(KeyEvent(KeyAction.DOWN, usage=4))

    assert not service.handle_message(
        Message(
            MessageType.INPUT_ENTER,
            {"session_id": REMOTE_SESSION, "x": 0, "y": 50},
        )
    )

    assert service.state is InputSessionState.IDLE
    assert backend.pressed_keys == frozenset()
    assert bus.sent[-1][0].message_type is MessageType.INPUT_STOP


def test_pointer_state_reports_commanded_position_before_async_os_application():
    class DeferredBackend(FakeBackend):
        def _inject_native(self, event):
            self.native_injected.append(event)

    bus = FakeBus()
    backend = DeferredBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
    )
    service.set_allow_remote_input(True)
    assert service.handle_message(input_request())
    bus.sent.clear()

    assert service.handle_message(
        Message(
            MessageType.INPUT_MOVE,
            {
                "session_id": REMOTE_SESSION,
                "motion_sequence": 1,
                "x": 40,
                "y": 50,
            },
        )
    )

    report = bus.sent[-1][0]
    assert report.message_type is MessageType.INPUT_POINTER_STATE
    assert (report.metadata["x"], report.metadata["y"]) == (40, 50)


def test_motion_scheduler_flushes_the_final_coalesced_position():
    service, bus, backend, _session_id = start_controlling()
    bus.sent.clear()

    backend.capture(PointerMotionEvent(5, 2))

    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and not bus.sent:
        time.sleep(0.005)
    try:
        assert [item[0].message_type for item in bus.sent] == [
            MessageType.INPUT_MOVE
        ]
        assert bus.sent[0][2] is Priority.INTERACTIVE
        assert (bus.sent[0][0].metadata["x"], bus.sent[0][0].metadata["y"]) == (
            5,
            52,
        )
    finally:
        service.close()


def test_pointer_state_reconciles_controller_logical_position():
    service, bus, backend, session_id = start_controlling()
    bus.sent.clear()
    backend.capture(PointerMotionEvent(5, 0))
    backend.capture(MouseButtonEvent(MouseButton.LEFT, KeyAction.DOWN))
    sent_move = next(
        item[0]
        for item in bus.sent
        if item[0].message_type is MessageType.INPUT_MOVE
    )
    bus.sent.clear()

    assert service.handle_message(
        Message(
            MessageType.INPUT_POINTER_STATE,
            {
                "session_id": session_id,
                "motion_sequence": sent_move.metadata["motion_sequence"],
                "x": 150,
                "y": 20,
            },
        )
    )

    backend.capture(PointerMotionEvent(1, 1))
    backend.capture(MouseButtonEvent(MouseButton.LEFT, KeyAction.DOWN))

    move = next(
        item[0]
        for item in bus.sent
        if item[0].message_type is MessageType.INPUT_MOVE
    )
    assert (move.metadata["x"], move.metadata["y"]) == (151, 21)


def test_stale_pointer_state_does_not_rewind_newer_logical_position():
    service, bus, backend, session_id = start_controlling()
    bus.sent.clear()
    backend.capture(PointerMotionEvent(5, 0))
    backend.capture(MouseButtonEvent(MouseButton.LEFT, KeyAction.DOWN))
    first = next(item[0] for item in bus.sent if item[0].message_type is MessageType.INPUT_MOVE)
    bus.sent.clear()
    backend.capture(PointerMotionEvent(5, 0))
    backend.capture(MouseButtonEvent(MouseButton.LEFT, KeyAction.DOWN))
    second = next(item[0] for item in bus.sent if item[0].message_type is MessageType.INPUT_MOVE)
    bus.sent.clear()

    assert service.handle_message(
        Message(
            MessageType.INPUT_POINTER_STATE,
            {
                "session_id": session_id,
                "motion_sequence": first.metadata["motion_sequence"],
                "x": 1,
                "y": 1,
            },
        )
    )
    backend.capture(PointerMotionEvent(1, 0))
    backend.capture(MouseButtonEvent(MouseButton.LEFT, KeyAction.DOWN))

    latest = [item[0] for item in bus.sent if item[0].message_type is MessageType.INPUT_MOVE][-1]
    assert second.metadata["motion_sequence"] > first.metadata["motion_sequence"]
    assert (latest.metadata["x"], latest.metadata["y"]) == (11, 50)


def test_accept_enter_send_failure_rolls_back_suppressed_capture():
    bus = FailingBus(fail_types={MessageType.INPUT_ENTER})
    backend = FakeBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        session_factory=lambda: LOCAL_SESSION,
    )
    session_id = service.request_control()

    assert service.handle_message(input_accept(session_id)) is False
    assert service.state is InputSessionState.IDLE
    assert backend.capture_running is False
    assert backend.warps[-1] == (98, 50)


def test_accept_capture_start_failure_restores_local_cursor():
    class FailingStartBackend(FakeBackend):
        def _start_native_capture(self, suppress):
            assert suppress is True
            self.position = (50, 50)
            raise RuntimeError("capture failed")

    backend = FailingStartBackend(position=(99, 50))
    service = InputService(
        FakeBus(),
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        session_factory=lambda: LOCAL_SESSION,
    )
    session_id = service.request_control()

    assert service.handle_message(input_accept(session_id)) is False
    assert service.state is InputSessionState.IDLE
    assert backend.capture_running is False
    assert backend.warps[-1] == (98, 50)


def test_stop_cleans_up_even_when_pending_pointer_flush_fails():
    bus = FailingBus()
    backend = FakeBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        clock=FakeClock(),
        session_factory=lambda: LOCAL_SESSION,
    )
    session_id = service.request_control()
    assert service.handle_message(input_accept(session_id))
    backend.capture(PointerMotionEvent(1, 0))
    bus.fail_types.add(MessageType.INPUT_MOVE)

    service.stop_control(reason="manual")

    assert service.state is InputSessionState.IDLE
    assert backend.capture_running is False


def test_interactive_send_failure_stops_capture_without_escaping_hook_callback():
    bus = FailingBus()
    backend = FakeBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        session_factory=lambda: LOCAL_SESSION,
    )
    session_id = service.request_control()
    assert service.handle_message(input_accept(session_id))
    bus.fail_types.add(MessageType.INPUT_BUTTON)

    backend.capture(MouseButtonEvent(MouseButton.LEFT, KeyAction.DOWN))

    assert service.state is InputSessionState.IDLE
    assert backend.capture_running is False


def test_stop_during_capture_start_cannot_publish_late_enter():
    holder = {}

    class StoppingBackend(FakeBackend):
        def _start_native_capture(self, suppress):
            super()._start_native_capture(suppress)
            holder["service"].stop_control(reason="cancelled")

    bus = FakeBus()
    backend = StoppingBackend()
    service = InputService(
        bus,
        backend,
        local_peer_id="peer-a",
        peer_id="peer-b",
        session_factory=lambda: LOCAL_SESSION,
    )
    holder["service"] = service
    session_id = service.request_control()
    bus.sent.clear()

    service.handle_message(input_accept(session_id))

    assert service.state is InputSessionState.IDLE
    assert backend.capture_running is False
    assert MessageType.INPUT_ENTER not in [item[0].message_type for item in bus.sent]
