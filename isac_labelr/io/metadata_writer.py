from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from isac_labelr.models import EventRecord


CSV_FIELDS = [
    "event_id",
    "video_name",
    "roi_id",
    "person_id",
    "event_type",
    "direction_label",
    "frame_index",
    "video_time_ms",
    "overlay_ts_ms",
    "overlay_ts_status",
    "correction_ms",
    "corrected_ts_ms",
    "det_conf",
    "ocr_conf",
]


class MetadataWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.output_dir / "events.jsonl"
        self.csv_path = self.output_dir / "events.csv"

        self._jsonl_fh = self.jsonl_path.open("w", encoding="utf-8")
        self._csv_fh = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=CSV_FIELDS)
        self._csv_writer.writeheader()

    @staticmethod
    def default_output_dir(video_path: str) -> Path:
        video = Path(video_path)
        return video.parent / "output" / video.stem

    def write_event(self, event: EventRecord) -> None:
        payload = event.to_dict()
        self._jsonl_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._csv_writer.writerow(payload)

    def write_session_config(self, config: dict) -> Path:
        path = self.output_dir / "session_config.json"
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def flush(self) -> None:
        self._jsonl_fh.flush()
        self._csv_fh.flush()

    def close(self) -> None:
        self.flush()
        self._jsonl_fh.close()
        self._csv_fh.close()


def export_events(events: Iterable[EventRecord], output_dir: Path) -> tuple[Path, Path]:
    writer = MetadataWriter(output_dir)
    try:
        for event in events:
            writer.write_event(event)
    finally:
        writer.close()
    return writer.jsonl_path, writer.csv_path
