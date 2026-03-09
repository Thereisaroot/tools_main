from __future__ import annotations

import logging
from pathlib import Path

from isac_labelr.analysis import worker as worker_mod
from isac_labelr.models import (
    AnalysisMode,
    AnalysisRequest,
    AnalysisResult,
    EventRecord,
    OverlayTSStatus,
)


def _event(event_id: str = "e1") -> EventRecord:
    return EventRecord(
        event_id=event_id,
        video_name="sample.mp4",
        roi_id="roi_1",
        person_id=1,
        event_type="enter",
        direction_label=None,
        frame_index=10,
        video_time_ms=333,
        overlay_ts_ms=1772526273000,
        overlay_ts_status=OverlayTSStatus.OK,
        correction_ms=0,
        corrected_ts_ms=1772526273000,
        det_conf=0.9,
        ocr_conf=0.9,
        ocr_raw_text="1772526273000",
        visible_person_count=1,
        roi_person_count=1,
        confirmed_frame_index=10,
        confirmed_video_time_ms=333,
        ocr_frame_index=10,
        rotation_deg=0,
        roi_x=0,
        roi_y=0,
        roi_w=10,
        roi_h=10,
    )


def _request(tmp_path: Path) -> AnalysisRequest:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")
    return AnalysisRequest(
        video_path=str(video_path),
        mode=AnalysisMode.ENTRY_ONLY,
        rotation_deg=0,
        rois=[],
        direction_vectors={},
        analyze_full=True,
    )


def test_worker_saves_buffer_on_abort_without_stop_short_circuit(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class FakeStore:
        def __init__(self, _video_path: str) -> None:
            self.primary_path = tmp_path / "out.json"
            self.debug_path = tmp_path / "out_debug.json"

        def save_all(self, **kwargs):
            captured.update(kwargs)
            return self.primary_path, self.primary_path

    class FakeEngine:
        def run(self, request, *, stop_event, on_progress, on_event, logger):
            on_event(_event("aborted"))
            return AnalysisResult(total_events=1, output_dir=Path("."), aborted=True)

    monkeypatch.setattr(worker_mod, "MetadataStore", FakeStore)
    monkeypatch.setattr(worker_mod, "AnalysisEngine", lambda: FakeEngine())
    monkeypatch.setattr(worker_mod, "build_run_logger", lambda _path: logging.getLogger("test"))

    worker = worker_mod.AnalysisWorker(_request(tmp_path))
    worker.run()

    assert captured["stop_event"] is None
    assert captured["resolve_missing_timestamps"] is False
    assert len(captured["events"]) == 1


def test_worker_uses_full_resolve_on_normal_finish(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class FakeStore:
        def __init__(self, _video_path: str) -> None:
            self.primary_path = tmp_path / "out.json"
            self.debug_path = tmp_path / "out_debug.json"

        def save_all(self, **kwargs):
            captured.update(kwargs)
            return self.primary_path, self.primary_path

    class FakeEngine:
        def run(self, request, *, stop_event, on_progress, on_event, logger):
            on_event(_event("done"))
            return AnalysisResult(total_events=1, output_dir=Path("."), aborted=False)

    monkeypatch.setattr(worker_mod, "MetadataStore", FakeStore)
    monkeypatch.setattr(worker_mod, "AnalysisEngine", lambda: FakeEngine())
    monkeypatch.setattr(worker_mod, "build_run_logger", lambda _path: logging.getLogger("test"))

    worker = worker_mod.AnalysisWorker(_request(tmp_path))
    worker.run()

    assert captured["stop_event"] is None
    assert captured["resolve_missing_timestamps"] is True
    assert len(captured["events"]) == 1
