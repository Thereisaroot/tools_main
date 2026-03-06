from isac_labelr.analysis.engine import AnalysisEngine, PresenceState
from isac_labelr.models import AnalysisMode, ROI, Track, Vector2


def _track(track_id: int, x1: float, y1: float, x2: float, y2: float) -> Track:
    return Track(track_id=track_id, bbox=(x1, y1, x2, y2), confidence=0.9, prev_center=(x1, y1))


def test_entry_debounce_three_frames():
    engine = AnalysisEngine()
    state = PresenceState()
    roi = ROI("roi_1", 0, 0, 100, 100)
    direction = Vector2(1.0, 0.0)

    event = None
    for i in range(3):
        event = engine._step_presence(
            state=state,
            inside=True,
            mode=AnalysisMode.ENTRY_ONLY,
            rotation_deg=0,
            track=_track(1, 10, 10, 20, 20),
            roi=roi,
            direction=direction,
            frame_index=i,
            video_time_ms=i * 33,
            video_name="a.mp4",
        )
    assert event is not None
    assert event.event_type == "enter"
    assert event.frame_index == 0
    assert event.confirmed_frame_index == 2


def test_exit_in_mode_b_after_debounce():
    engine = AnalysisEngine()
    state = PresenceState(inside=True)
    roi = ROI("roi_1", 0, 0, 100, 100)

    event = None
    for i in range(3):
        event = engine._step_presence(
            state=state,
            inside=False,
            mode=AnalysisMode.ENTRY_EXIT_DIRECTION,
            rotation_deg=0,
            track=_track(1, 120, 120, 140, 140),
            roi=roi,
            direction=Vector2(1.0, 0.0),
            frame_index=i,
            video_time_ms=i * 33,
            video_name="a.mp4",
        )

    assert event is not None
    assert event.event_type == "exit"
    assert event.frame_index == 0
    assert event.confirmed_frame_index == 2
