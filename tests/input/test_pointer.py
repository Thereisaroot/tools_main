import pytest

from shooklink.input.pointer import LogicalPointer, TransitionKind
from shooklink.input.topology import Monitor, Rect, Side, Topology


def single_monitor_pointer():
    topology = Topology((Monitor("display", Rect(0, 0, 1920, 1080)),))
    return LogicalPointer(topology), topology


@pytest.mark.parametrize(
    ("entry_side", "start_fraction", "blocked_delta", "reverse_delta", "expected"),
    [
        (Side.LEFT, 0.0, (0, -50), (0, 1), (0, 1)),
        (Side.LEFT, 0.5, (-50, 0), (1, 0), (1, 540)),
        (Side.RIGHT, 0.5, (50, 0), (-1, 0), (1918, 540)),
        (Side.LEFT, 1.0, (0, 50), (0, -1), (0, 1078)),
    ],
)
def test_blocked_motion_never_accumulates_dead_space(
    entry_side,
    start_fraction,
    blocked_delta,
    reverse_delta,
    expected,
):
    pointer, _topology = single_monitor_pointer()
    pointer.enter(entry_side, start_fraction)

    for _index in range(240):
        pointer.move(*blocked_delta)
    transition = pointer.move(*reverse_delta)

    assert transition.kind is TransitionKind.MOVE
    assert transition.position == expected


def test_monitor_gap_motion_is_discarded_instead_of_becoming_debt():
    topology = Topology(
        (
            Monitor("main", Rect(0, 0, 1920, 1080)),
            Monitor("right", Rect(1920, 200, 1280, 1024)),
        )
    )
    pointer = LogicalPointer(topology)
    pointer.set_position(1919, 100)

    for _index in range(240):
        pointer.move(20, 0)
    transition = pointer.move(-1, 0)

    assert transition.position == (1918, 100)


def test_connected_return_edge_emits_leave_instead_of_clamping():
    topology = Topology((Monitor("display", Rect(0, 0, 1920, 1080)),))
    pointer = LogicalPointer(topology, return_side=Side.LEFT)
    pointer.enter(Side.LEFT, 0.5)

    transition = pointer.move(-1, 0)

    assert transition.kind is TransitionKind.LEAVE
    assert transition.position == (0, 540)


def test_large_delta_crossing_connected_return_edge_emits_leave():
    topology = Topology((Monitor("display", Rect(0, 0, 1920, 1080)),))
    pointer = LogicalPointer(topology, return_side=Side.LEFT)
    pointer.set_position(5, 540)

    transition = pointer.move(-10, 0)

    assert transition.kind is TransitionKind.LEAVE
    assert transition.position == (0, 540)


def test_large_delta_into_monitor_gap_stays_on_source_monitor():
    topology = Topology(
        (
            Monitor("main", Rect(0, 0, 1920, 1080)),
            Monitor("right", Rect(1920, 200, 1280, 1024)),
        )
    )
    pointer = LogicalPointer(topology)
    pointer.set_position(1919, 100)

    transition = pointer.move(200, 0)

    assert transition.position == (1919, 100)


def test_set_position_rejects_non_integer_coordinates():
    pointer, _topology = single_monitor_pointer()

    with pytest.raises(TypeError, match="integers"):
        pointer.set_position(1.5, 2.5)


def test_enter_and_move_use_absolute_destination_coordinates():
    pointer, topology = single_monitor_pointer()

    entered = pointer.enter(Side.TOP, 0.25)
    moved = pointer.move(10, 20)

    assert entered.kind is TransitionKind.ENTER
    assert topology.contains(*entered.position)
    assert moved.kind is TransitionKind.MOVE
    assert moved.position == (490, 20)
