from isac_labelr.io.video_stream import _time_ms_to_frame_index


def test_time_ms_to_frame_index_uses_rounding() -> None:
    # 89 / 24 fps = 3708.33ms. Truncation would give 88 for 3708ms.
    assert _time_ms_to_frame_index(3708, 24.0) == 89


def test_time_ms_to_frame_index_stays_monotonic() -> None:
    assert _time_ms_to_frame_index(3708, 24.0, None) == 89
    assert _time_ms_to_frame_index(3749, 24.0, 89) == 90
    # Even with a jittery timestamp that rounds backward, frame index must advance.
    assert _time_ms_to_frame_index(3748, 24.0, 90) == 91
