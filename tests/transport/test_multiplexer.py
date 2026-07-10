import threading
import time

import pytest

from shooklink.transport.multiplexer import (
    Multiplexer,
    MultiplexerClosed,
    OutboundItem,
    Priority,
)


def test_interactive_traffic_overtakes_file_data():
    mux = Multiplexer()
    mux.enqueue(OutboundItem(Priority.FILE, 5, b"chunk"))
    mux.enqueue(OutboundItem(Priority.INTERACTIVE, 2, b"key"))

    assert mux.pop().payload == b"key"
    assert mux.pop().payload == b"chunk"


def test_same_priority_is_fifo():
    mux = Multiplexer()
    mux.enqueue(OutboundItem(Priority.NORMAL, 1, b"first"))
    mux.enqueue(OutboundItem(Priority.NORMAL, 1, b"second"))

    assert [mux.pop().payload, mux.pop().payload] == [b"first", b"second"]


def test_pointer_moves_are_coalesced_per_stream():
    mux = Multiplexer()
    mux.enqueue_pointer(9, b"first", message_type=47)
    mux.enqueue_pointer(9, b"latest", message_type=47)

    item = mux.pop()

    assert item.payload == b"latest"
    assert item.priority is Priority.MOTION
    assert item.message_type == 47
    assert mux.empty()


def test_pointer_move_overtakes_file_but_not_interactive():
    mux = Multiplexer()
    mux.enqueue(OutboundItem(Priority.FILE, 1, b"file"))
    mux.enqueue_pointer(2, b"move")
    mux.enqueue(OutboundItem(Priority.INTERACTIVE, 3, b"key"))

    assert [mux.pop().payload for _index in range(3)] == [b"key", b"move", b"file"]


def test_sequence_reservations_are_independent_per_stream():
    mux = Multiplexer()

    assert mux.reserve_sequence(3) == 1
    assert mux.reserve_sequence(3) == 2
    assert mux.reserve_sequence(8) == 1


def test_enqueue_assigns_sequence_if_missing_and_preserves_explicit_sequence():
    mux = Multiplexer()
    automatic = mux.enqueue(OutboundItem(Priority.NORMAL, 3, b"auto"))
    explicit = mux.enqueue(OutboundItem(Priority.NORMAL, 3, b"retry", sequence=1))

    assert automatic.sequence == 1
    assert explicit.sequence == 1


def test_pop_blocks_until_data_arrives():
    mux = Multiplexer()
    result = []
    started = threading.Event()

    def pop_item():
        started.set()
        result.append(mux.pop(timeout=1))

    thread = threading.Thread(target=pop_item)
    thread.start()
    assert started.wait(1)
    time.sleep(0.01)
    mux.enqueue(OutboundItem(Priority.NORMAL, 1, b"ready"))
    thread.join(1)

    assert result[0].payload == b"ready"


def test_close_unblocks_waiter_and_rejects_new_work():
    mux = Multiplexer()
    result = []
    thread = threading.Thread(target=lambda: result.append(mux.pop()))
    thread.start()
    time.sleep(0.01)

    mux.close()
    thread.join(1)

    assert result == [None]
    with pytest.raises(MultiplexerClosed):
        mux.enqueue(OutboundItem(Priority.NORMAL, 1, b"late"))
    with pytest.raises(MultiplexerClosed):
        mux.reserve_sequence(1)


@pytest.mark.parametrize(
    "item",
    [
        OutboundItem(Priority.NORMAL, -1, b"bad"),
        OutboundItem(Priority.NORMAL, 1 << 32, b"bad"),
        OutboundItem(Priority.NORMAL, 1, b"bad", message_type=256),
        OutboundItem(Priority.NORMAL, 1, b"bad", flags=256),
        OutboundItem(Priority.NORMAL, 1, "not bytes"),
    ],
)
def test_invalid_items_are_rejected_on_enqueue(item):
    with pytest.raises((TypeError, ValueError)):
        Multiplexer().enqueue(item)
