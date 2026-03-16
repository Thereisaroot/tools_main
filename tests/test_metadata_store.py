from __future__ import annotations

import json
from pathlib import Path

from isac_labelr.io import metadata_writer
from isac_labelr.io.metadata_writer import MetadataStore, build_label_records
from isac_labelr.models import EventRecord, OverlayTSStatus


def _event(event_id: str, roi_id: str, frame_index: int, ts: int | None) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        video_name="sample.mp4",
        roi_id=roi_id,
        person_id=1,
        event_type="enter",
        direction_label=None,
        frame_index=frame_index,
        video_time_ms=frame_index * 33,
        overlay_ts_ms=ts,
        overlay_ts_status=OverlayTSStatus.OK if ts is not None else OverlayTSStatus.FAILED,
        correction_ms=0,
        corrected_ts_ms=ts,
        det_conf=0.9,
        ocr_conf=0.9,
        ocr_raw_text=str(ts) if ts is not None else "",
        visible_person_count=1,
        roi_person_count=1,
        confirmed_frame_index=frame_index,
        confirmed_video_time_ms=frame_index * 33,
        ocr_frame_index=frame_index,
        rotation_deg=0,
        roi_x=0,
        roi_y=0,
        roi_w=10,
        roi_h=10,
    )


class _FakeResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None]] = []

    def resolve(self, frame_index: int, *, preferred_ts: int | None = None) -> int:
        self.calls.append((frame_index, preferred_ts))
        if preferred_ts is not None:
            return int(preferred_ts)
        return 1770000000000 + int(frame_index)

    def close(self) -> None:
        return


def test_frame_timestamp_resolver_uses_only_requested_frame(monkeypatch, tmp_path: Path) -> None:
    class FakeOCR:
        def __init__(self) -> None:
            self.calls: list[tuple[int, bool]] = []

        def set_manual_roi(self, _roi) -> None:
            return

        def _extract_timestamp_for_video_frame(self, *, video_path: str, frame_index: int, rotation_deg: int, fast: bool):
            self.calls.append((int(frame_index), bool(fast)))
            return type(
                "FakeResult",
                (),
                {
                    "timestamp_ms": None,
                    "raw_text": "",
                    "confidence": 0.0,
                },
            )()

        def is_valid_timestamp(self, value) -> bool:
            return value is not None and str(value).startswith("177") and len(str(value)) == 13

        def extract_timestamp_candidates(self, raw_text: str) -> list[int]:
            return []

        def close(self) -> None:
            return

    monkeypatch.setattr(metadata_writer, "TimestampOCR", FakeOCR)
    monkeypatch.setattr(metadata_writer.FrameTimestampResolver, "_read_fps", staticmethod(lambda _path: 30.0))

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    resolver = metadata_writer.FrameTimestampResolver(
        video_path=str(video_path),
        rotation_deg=0,
        ocr_manual_roi=None,
    )
    try:
        resolved = resolver.resolve(124)
        assert resolved is None
        assert resolver.ocr.calls == [(124, True), (124, False)]
    finally:
        resolver.close()


def test_build_labels_creates_non_overlapping_zero_gap() -> None:
    events = [
        _event("e1", "roi_1", 10, 1772526273000),
        _event("e2", "roi_1", 50, 1772526274000),   # overlaps with e1 window
        _event("e3", "roi_2", 200, 1772526275000),
    ]
    resolver = _FakeResolver()
    labels = build_label_records(events, resolver=resolver, max_frame_index=999)

    rows = [row.to_dict() for row in labels]
    # ROI windows are kept per event (+100 inclusive), gap is computed from merged ROI union.
    assert rows[0]["label_id"] == 0 and rows[0]["start_frame"] == 0 and rows[0]["end_frame"] == 9
    assert rows[1]["label_id"] == 1 and rows[1]["start_frame"] == 10 and rows[1]["end_frame"] == 110
    assert rows[2]["label_id"] == 1 and rows[2]["start_frame"] == 50 and rows[2]["end_frame"] == 150
    assert rows[3]["label_id"] == 0 and rows[3]["start_frame"] == 151 and rows[3]["end_frame"] == 199
    assert rows[4]["label_id"] == 2 and rows[4]["start_frame"] == 200 and rows[4]["end_frame"] == 300
    assert rows[5]["label_id"] == 0 and rows[5]["start_frame"] == 301 and rows[5]["end_frame"] == 999


def test_parse_label_id_mapping() -> None:
    assert metadata_writer._parse_label_id("roi_1") == 1
    assert metadata_writer._parse_label_id("1") == 1
    assert metadata_writer._parse_label_id("camera_roi_12") == 12
    assert metadata_writer._parse_label_id("bad") == 0


def test_store_save_all_writes_primary_and_debug(monkeypatch, tmp_path: Path) -> None:
    class FakeFrameTimestampResolver(_FakeResolver):
        @staticmethod
        def read_max_frame_index(video_path: str) -> int:
            return 1000

        def __init__(self, *, video_path: str, rotation_deg: int, ocr_manual_roi):
            super().__init__()
            self.video_path = video_path
            self.rotation_deg = rotation_deg
            self.ocr_manual_roi = ocr_manual_roi

        def resolve_exact(self, frame_index: int):
            return metadata_writer.FrameTimestampPick(
                timestamp_ms=1770000000000 + int(frame_index),
                confidence=0.95,
                raw_text=str(1770000000000 + int(frame_index)),
            )

    monkeypatch.setattr(metadata_writer, "FrameTimestampResolver", FakeFrameTimestampResolver)

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    events = [
        _event("e1", "roi_1", 100, 1772526273000),
        _event("e2", "roi_2", 300, 1772526275000),
    ]
    store = MetadataStore(str(video_path))
    primary, debug = store.save_all(
        events=events,
        session={"video_path": str(video_path), "rotation_deg": 0, "ocr_manual_roi": None},
        max_frame_index=1000,
    )

    assert primary.exists()
    assert debug.exists()

    primary_payload = json.loads(primary.read_text(encoding="utf-8"))
    debug_payload = json.loads(debug.read_text(encoding="utf-8"))

    assert primary_payload["video_path"] == str(video_path)
    assert isinstance(primary_payload["labels"], list)
    assert len(primary_payload["labels"]) == 5  # 2 ROI windows + leading/mid/tail zero windows
    assert "index" not in primary_payload["labels"][0]

    assert debug_payload["video_path"] == str(video_path)
    assert debug_payload["version"] == 1
    assert isinstance(debug_payload["events"], list)
    assert len(debug_payload["events"]) == 2
    assert debug_payload["events"][0]["overlay_ts_ms"] == 1770000000100
    assert debug_payload["events"][0]["ocr_frame_index"] == 100
    assert debug_payload["events"][0]["label_end_timestamp_unix"] == 1770000000200
    assert debug_payload["events"][1]["overlay_ts_ms"] == 1770000000300
    assert debug_payload["events"][1]["label_end_timestamp_unix"] == 1770000000400


def test_manual_label_zero_with_end_frame_is_preserved() -> None:
    manual = _event("manual", "0", 123, 1772526273123)
    manual.event_type = "label"
    manual.label_end_frame = 222
    manual.label_end_timestamp_unix = 1772526273999

    resolver = _FakeResolver()
    labels = build_label_records([manual], resolver=resolver, max_frame_index=999)
    assert len(labels) == 1
    row = labels[0].to_dict()
    assert row["label_id"] == 0
    assert row["start_frame"] == 123
    assert row["end_frame"] == 222
    assert row["start_timestamp_unix"] == 1772526273123
    assert row["end_timestamp_unix"] == 1772526273999
