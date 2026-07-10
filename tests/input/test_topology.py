from shooklink.input.topology import EdgeSegment, Monitor, Rect, Side, Topology


def staggered_topology():
    return Topology(
        (
            Monitor("main", Rect(0, 0, 1920, 1080)),
            Monitor("right", Rect(1920, 200, 1280, 1024)),
        )
    )


def test_internal_seam_is_not_an_outer_edge():
    topology = staggered_topology()

    assert not topology.is_on_outer_edge(Side.RIGHT, 1919, 500)
    assert not topology.is_on_outer_edge(Side.LEFT, 1920, 500)


def test_right_edge_exists_only_over_extreme_monitor_span():
    topology = staggered_topology()

    assert topology.edge_segments(Side.RIGHT) == (
        EdgeSegment(Side.RIGHT, 3199, 200, 1224),
    )
    assert topology.is_on_outer_edge(Side.RIGHT, 3199, 500)
    assert not topology.is_on_outer_edge(Side.RIGHT, 3199, 100)
    assert not topology.is_on_outer_edge(Side.RIGHT, 1919, 100)


def test_negative_coordinate_monitors_are_supported():
    topology = Topology(
        (
            Monitor("left", Rect(-1280, -200, 1280, 1024)),
            Monitor("main", Rect(0, 0, 1920, 1080)),
        )
    )

    assert topology.contains(-1279, -199)
    assert topology.edge_segments(Side.LEFT) == (
        EdgeSegment(Side.LEFT, -1280, -200, 824),
    )
    assert topology.edge_segments(Side.TOP) == (
        EdgeSegment(Side.TOP, -200, -1280, 0),
    )


def test_cross_axis_fraction_maps_inside_real_destination_segment():
    destination = Topology(
        (
            Monitor("upper", Rect(100, -500, 800, 300)),
            Monitor("lower", Rect(100, 200, 800, 600)),
        )
    )

    upper = destination.map_fraction_to_edge(Side.LEFT, 0.25)
    lower = destination.map_fraction_to_edge(Side.LEFT, 0.75)

    assert destination.contains(*upper)
    assert destination.contains(*lower)
    assert upper[0] == lower[0] == 100
    assert -500 <= upper[1] < -200
    assert 200 <= lower[1] < 800


def test_nearest_valid_point_does_not_land_in_monitor_gap():
    topology = staggered_topology()

    point = topology.nearest_point(1920, 100)

    assert point == (1919, 100)
    assert topology.contains(*point)
