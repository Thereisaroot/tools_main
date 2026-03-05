from isac_labelr.models import AnalysisMode, ROI, SessionConfig, Vector2


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
    )

    restored = SessionConfig.from_dict(session.to_dict())
    assert restored.video_path == session.video_path
    assert restored.rotation_deg == 90
    assert restored.mode == AnalysisMode.ENTRY_EXIT_DIRECTION
    assert restored.rois[0].roi_id == "roi_1"
    assert restored.direction_vectors["roi_1"].dx == 1.0
