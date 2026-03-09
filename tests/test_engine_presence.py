import logging

from isac_labelr.analysis.engine import AnalysisEngine, PresenceState
from isac_labelr.models import AnalysisMode, OverlayTSStatus, ROI, Track, Vector2


def _track(track_id: int, x1: float, y1: float, x2: float, y2: float) -> Track:
    return Track(track_id=track_id, bbox=(x1, y1, x2, y2), confidence=0.9, prev_center=(x1, y1))


def test_entry_debounce_one_frame_default():
    engine = AnalysisEngine()
    state = PresenceState()
    roi = ROI("roi_1", 0, 0, 100, 100)
    direction = Vector2(1.0, 0.0)

    event = engine._step_presence(
        state=state,
        inside=True,
        mode=AnalysisMode.ENTRY_ONLY,
        rotation_deg=0,
        track=_track(1, 10, 10, 20, 20),
        roi=roi,
        direction=direction,
        frame_index=0,
        video_time_ms=0,
        video_name="a.mp4",
    )
    assert event is not None
    assert event.event_type == "enter"
    assert event.frame_index == 0
    assert event.confirmed_frame_index == 0


def test_exit_in_mode_b_after_two_frames_default():
    engine = AnalysisEngine()
    state = PresenceState(inside=True)
    roi = ROI("roi_1", 0, 0, 100, 100)

    event = None
    for i in range(2):
        candidate = engine._step_presence(
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
        if candidate is not None:
            event = candidate

    assert event is not None
    assert event.event_type == "exit"
    assert event.frame_index == 0
    assert event.confirmed_frame_index == 1


def test_entry_and_roi_count_use_foot_point():
    engine = AnalysisEngine()
    presence: dict[tuple[str, int], PresenceState] = {}
    roi = ROI("roi_1", 0, 100, 100, 20)  # y in [100, 120]

    # center=(20,85) -> outside ROI, foot=(20,110) -> inside ROI
    in_track = _track(1, 10, 60, 30, 110)
    out_track = _track(2, 10, 10, 30, 40)

    seen_events = []
    for i in range(3):
        batch = engine._update_presence(
            rois=[roi],
            direction_vectors={},
            rotation_deg=0,
            mode=AnalysisMode.ENTRY_ONLY,
            enter_debounce_frames=1,
            exit_debounce_frames=2,
            frame_index=i,
            video_time_ms=i * 33,
            video_name="a.mp4",
            tracks=[in_track, out_track],
            presence=presence,
        )
        seen_events.extend(batch)

    assert len(seen_events) == 1
    event = seen_events[0]
    assert event.event_type == "enter"
    assert event.person_id == 1
    assert event.roi_person_count == 1


def test_fallback_ocr_timestamp_is_reanchored_to_first_frame() -> None:
    engine = AnalysisEngine()
    call_count = 0

    def fake_resolve_overlay_timestamp(**_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None, OverlayTSStatus.FAILED, None, ""
        return 1_772_526_275_506, OverlayTSStatus.OK, 0.9, "raw"

    engine._resolve_overlay_timestamp = fake_resolve_overlay_timestamp  # type: ignore[method-assign]

    anchor_video_ms = 3_416
    fallback_video_ms = 3_458
    expected = 1_772_526_275_506 - (fallback_video_ms - anchor_video_ms)

    overlay, _status, _conf, _raw, ocr_frame = engine._resolve_event_overlay_timestamp(
        ocr=None,  # type: ignore[arg-type]
        video_path="demo.mp4",
        rotation_deg=0,
        anchor_frame_index=82,
        anchor_video_time_ms=anchor_video_ms,
        anchor_frame_bgr=None,
        fallback_frame_index=83,
        fallback_video_time_ms=fallback_video_ms,
        fallback_frame_bgr=None,
        logger=logging.getLogger("test"),
    )

    assert call_count == 2
    assert ocr_frame == 83
    assert overlay == expected
    assert engine._last_overlay_ts == expected
    assert engine._last_overlay_video_time_ms == anchor_video_ms
