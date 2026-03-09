from isac_labelr.models import AnalysisMode, ROI, SessionConfig, Track, Vector2


def test_roi_normalized_contains():
    roi = ROI("r1", 100, 100, -50, -30).normalized()
    assert roi.x == 50
    assert roi.y == 70
    assert roi.w == 50
    assert roi.h == 30
    assert roi.contains(60, 80)
    assert not roi.contains(10, 10)


def test_session_round_trip():
    session = SessionConfig(
        video_path="/tmp/a.mp4",
        rotation_deg=90,
        mode=AnalysisMode.ENTRY_EXIT_DIRECTION,
        rois=[ROI("roi_1", 1, 2, 3, 4)],
        direction_vectors={"roi_1": Vector2(1.0, 0.0)},
        analyze_full=False,
        start_ms=100,
        duration_ms=1000,
        timestamp_correction_ms=-20,
        enter_debounce_frames=3,
        exit_debounce_frames=4,
    )

    restored = SessionConfig.from_dict(session.to_dict())
    assert restored.video_path == session.video_path
    assert restored.rotation_deg == 90
    assert restored.mode == AnalysisMode.ENTRY_EXIT_DIRECTION
    assert restored.rois[0].roi_id == "roi_1"
    assert restored.direction_vectors["roi_1"].dx == 1.0
    assert restored.enter_debounce_frames == 3
    assert restored.exit_debounce_frames == 4


def test_track_foot_point():
    track = Track(track_id=1, bbox=(10.0, 20.0, 30.0, 110.0), confidence=0.9)
    assert track.center == (20.0, 65.0)
    assert track.foot_point == (20.0, 110.0)


def test_session_debounce_defaults_for_legacy_payload():
    payload = {
        "video_path": "/tmp/a.mp4",
        "rotation_deg": 0,
        "mode": AnalysisMode.ENTRY_ONLY.value,
        "rois": [],
        "direction_vectors": {},
        "analyze_full": True,
        "start_ms": 0,
        "duration_ms": None,
        "timestamp_correction_ms": 0,
        "ocr_manual_roi": None,
    }
    restored = SessionConfig.from_dict(payload)
    assert restored.enter_debounce_frames == 1
    assert restored.exit_debounce_frames == 2
