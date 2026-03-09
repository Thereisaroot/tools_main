from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import cv2

from isac_labelr.models import EventRecord, ROI
from isac_labelr.vision.ocr import TimestampOCR

KST = ZoneInfo("Asia/Seoul")
ROI_ID_DIGITS = re.compile(r"(\d+)$")


@dataclass(slots=True)
class LabelRecord:
    label_id: int
    start_frame: int
    end_frame: int
    start_timestamp_unix: int | None
    start_timestamp_kst: str | None
    end_timestamp_unix: int | None
    end_timestamp_kst: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class _LabelWindow:
    label_id: int
    start_frame: int
    end_frame: int
    start_ts_hint: int | None = None
    end_ts_hint: int | None = None


def _parse_label_id(roi_id: str) -> int:
    text = str(roi_id or "").strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    lower = text.lower()
    if lower.startswith("roi_"):
        tail = lower.split("_", 1)[1]
        if tail.isdigit():
            return int(tail)
    match = ROI_ID_DIGITS.search(lower)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 0
    return 0


def _to_kst_text(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    try:
        dt = datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=KST)
    except Exception:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tf:
            json.dump(payload, tf, ensure_ascii=False, indent=2)
            tf.write("\n")
            tmp_name = tf.name
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _is_stop_requested(stop_event: object | None) -> bool:
    if stop_event is None:
        return False
    checker = getattr(stop_event, "is_set", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:
        return False


def _atomic_write_debug_payload(
    path: Path,
    *,
    video_path: str,
    session: dict,
    events: Iterable[EventRecord],
    stop_event: object | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tf.write("{\n")
            tf.write('  "version": 1,\n')
            tf.write('  "video_path": ')
            json.dump(video_path, tf, ensure_ascii=False)
            tf.write(",\n")
            tf.write('  "session": ')
            json.dump(session, tf, ensure_ascii=False, indent=2)
            tf.write(",\n")
            tf.write('  "events": [\n')

            first = True
            for event in events:
                if _is_stop_requested(stop_event):
                    break
                row = event.to_dict() if isinstance(event, EventRecord) else dict(event)
                if not first:
                    tf.write(",\n")
                tf.write("    ")
                json.dump(row, tf, ensure_ascii=False)
                first = False

            tf.write("\n  ]\n")
            tf.write("}\n")
            tmp_name = tf.name
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals, key=lambda item: (item[0], item[1])):
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


class FrameTimestampResolver:
    def __init__(
        self,
        *,
        video_path: str,
        rotation_deg: int,
        ocr_manual_roi: ROI | None,
    ) -> None:
        self.video_path = str(video_path)
        self.rotation_deg = int(rotation_deg) % 360
        self.ocr = TimestampOCR()
        if ocr_manual_roi is not None:
            self.ocr.set_manual_roi(ocr_manual_roi.normalized())
        self._last_valid_ts: int | None = None
        self._last_valid_frame: int | None = None
        self._fps = self._read_fps(self.video_path)

    @staticmethod
    def _read_fps(video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return 30.0
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            return fps if fps > 1e-6 else 30.0
        finally:
            cap.release()

    @staticmethod
    def read_max_frame_index(video_path: str) -> int | None:
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                return None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames <= 0:
                return None
            return total_frames - 1
        finally:
            cap.release()

    def _estimate_from_last(self, frame_index: int) -> int | None:
        if self._last_valid_ts is None or self._last_valid_frame is None:
            return None
        delta_frames = max(0, int(frame_index) - int(self._last_valid_frame))
        delta_ms = int(round((delta_frames / max(self._fps, 1.0)) * 1000.0))
        return int(self._last_valid_ts) + delta_ms

    def _pick_timestamp(self, frame_index: int) -> int | None:
        result = self.ocr._extract_timestamp_for_video_frame(
            video_path=self.video_path,
            frame_index=int(frame_index),
            rotation_deg=self.rotation_deg,
            fast=True,
        )
        if self.ocr.is_valid_timestamp(result.timestamp_ms):
            return int(result.timestamp_ms)
        candidates = self.ocr.extract_timestamp_candidates(result.raw_text)
        for candidate in candidates:
            if self.ocr.is_valid_timestamp(candidate):
                return int(candidate)

        neighbor_ts, _neighbor_conf = self.ocr.interpolate_timestamp_from_neighbors(
            video_path=self.video_path,
            frame_index=int(frame_index),
            rotation_deg=self.rotation_deg,
            fast=True,
        )
        if self.ocr.is_valid_timestamp(neighbor_ts):
            return int(neighbor_ts)
        return None

    def resolve(self, frame_index: int, *, preferred_ts: int | None = None) -> int | None:
        frame_index = int(frame_index)
        if self.ocr.is_valid_timestamp(preferred_ts):
            picked = int(preferred_ts)
            self._last_valid_ts = picked
            self._last_valid_frame = frame_index
            return picked

        picked = self._pick_timestamp(frame_index)
        if picked is not None:
            self._last_valid_ts = int(picked)
            self._last_valid_frame = frame_index
            return int(picked)

        estimated = self._estimate_from_last(frame_index)
        if estimated is not None:
            self._last_valid_ts = int(estimated)
            self._last_valid_frame = frame_index
            return int(estimated)
        return None

    def close(self) -> None:
        self.ocr.close()


def _make_label_windows(
    events: Iterable[EventRecord],
    *,
    max_frame_index: int | None,
) -> tuple[list[_LabelWindow], list[_LabelWindow]]:
    roi_windows: list[_LabelWindow] = []
    manual_zero_windows: list[_LabelWindow] = []
    for event in sorted(events, key=lambda e: (int(e.frame_index), str(e.event_id))):
        label_id = _parse_label_id(event.roi_id)
        start = max(0, int(event.frame_index))
        end_override = event.label_end_frame
        if end_override is not None:
            end = int(end_override)
        else:
            end = int(start + 100)
        if max_frame_index is not None:
            end = min(end, int(max_frame_index))
        end = max(start, end)
        if end < start:
            continue
        hint = int(event.overlay_ts_ms) if event.overlay_ts_ms is not None else None
        end_hint = (
            int(event.label_end_timestamp_unix)
            if event.label_end_timestamp_unix is not None
            else None
        )
        window = _LabelWindow(
            label_id=label_id,
            start_frame=start,
            end_frame=end,
            start_ts_hint=hint,
            end_ts_hint=end_hint,
        )
        if label_id > 0:
            roi_windows.append(window)
        elif (event.event_type == "label") or (event.label_end_frame is not None):
            manual_zero_windows.append(window)

    merged = _merge_intervals([(w.start_frame, w.end_frame) for w in roi_windows])
    zero_windows: list[_LabelWindow] = list(manual_zero_windows)
    for idx in range(len(merged) - 1):
        current_end = int(merged[idx][1])
        next_start = int(merged[idx + 1][0])
        gap_start = current_end + 1
        gap_end = next_start - 1
        if gap_start <= gap_end:
            zero_windows.append(
                _LabelWindow(
                    label_id=0,
                    start_frame=gap_start,
                    end_frame=gap_end,
                    start_ts_hint=None,
                    end_ts_hint=None,
                )
            )
    deduped_zero: list[_LabelWindow] = []
    seen_zero: set[tuple[int, int]] = set()
    for window in sorted(zero_windows, key=lambda w: (w.start_frame, w.end_frame)):
        key = (int(window.start_frame), int(window.end_frame))
        if key in seen_zero:
            continue
        seen_zero.add(key)
        deduped_zero.append(window)
    return roi_windows, deduped_zero


def build_label_records(
    events: Iterable[EventRecord],
    *,
    resolver: FrameTimestampResolver | None = None,
    max_frame_index: int | None = None,
    resolve_missing_timestamps: bool = True,
    stop_event: object | None = None,
) -> list[LabelRecord]:
    roi_windows, zero_windows = _make_label_windows(events, max_frame_index=max_frame_index)
    windows = [*roi_windows, *zero_windows]
    windows.sort(key=lambda w: (int(w.start_frame), 1 if w.label_id == 0 else 0, int(w.label_id)))

    if resolve_missing_timestamps and resolver is None:
        raise ValueError("resolver is required when resolve_missing_timestamps=True")

    rows: list[LabelRecord] = []
    for window in windows:
        if _is_stop_requested(stop_event):
            break
        start_frame = int(window.start_frame)
        end_frame = int(window.end_frame)
        if resolve_missing_timestamps and resolver is not None:
            start_ts = resolver.resolve(start_frame, preferred_ts=window.start_ts_hint)
            end_ts = resolver.resolve(end_frame, preferred_ts=window.end_ts_hint)
        else:
            start_ts = int(window.start_ts_hint) if window.start_ts_hint is not None else None
            end_ts = int(window.end_ts_hint) if window.end_ts_hint is not None else None
        rows.append(
            LabelRecord(
                label_id=int(window.label_id),
                start_frame=start_frame,
                end_frame=end_frame,
                start_timestamp_unix=start_ts,
                start_timestamp_kst=_to_kst_text(start_ts),
                end_timestamp_unix=end_ts,
                end_timestamp_kst=_to_kst_text(end_ts),
            )
        )
    return rows


class MetadataStore:
    def __init__(self, video_path: str) -> None:
        self.video_path = str(video_path)
        self.primary_path, self.debug_path = self.metadata_paths(self.video_path)

    @staticmethod
    def metadata_paths(video_path: str) -> tuple[Path, Path]:
        video = Path(video_path)
        primary = video.with_suffix(".json")
        debug = video.with_name(f"{video.stem}_debug.json")
        return primary, debug

    @staticmethod
    def load_debug(path: Path) -> tuple[str | None, dict | None, list[dict]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        video_path = payload.get("video_path")
        session = payload.get("session")
        raw_events = payload.get("events", [])
        events = [dict(row) for row in raw_events if isinstance(row, dict)]
        return str(video_path) if video_path else None, (session if isinstance(session, dict) else None), events

    @staticmethod
    def load_primary(path: Path) -> tuple[str | None, list[dict]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        video_path = payload.get("video_path")
        labels = payload.get("labels", [])
        rows = [dict(row) for row in labels if isinstance(row, dict)]
        return str(video_path) if video_path else None, rows

    def save_all(
        self,
        *,
        events: Iterable[EventRecord],
        session: dict | None,
        max_frame_index: int | None = None,
        resolve_missing_timestamps: bool = True,
        stop_event: object | None = None,
    ) -> tuple[Path, Path]:
        events_list = list(events)
        session_dict = dict(session or {})
        if max_frame_index is None:
            max_frame_index = FrameTimestampResolver.read_max_frame_index(self.video_path)

        manual_roi = None
        raw_manual = session_dict.get("ocr_manual_roi")
        if isinstance(raw_manual, dict):
            try:
                manual_roi = ROI(**raw_manual).normalized()
            except Exception:
                manual_roi = None

        if resolve_missing_timestamps:
            rotation_deg = int(session_dict.get("rotation_deg", 0))
            resolver = FrameTimestampResolver(
                video_path=self.video_path,
                rotation_deg=rotation_deg,
                ocr_manual_roi=manual_roi,
            )
            try:
                labels = build_label_records(
                    events_list,
                    resolver=resolver,
                    max_frame_index=max_frame_index,
                    resolve_missing_timestamps=True,
                    stop_event=stop_event,
                )
            finally:
                resolver.close()
        else:
            labels = build_label_records(
                events_list,
                max_frame_index=max_frame_index,
                resolve_missing_timestamps=False,
                stop_event=stop_event,
            )

        primary_payload = {
            "video_path": self.video_path,
            "labels": [row.to_dict() for row in labels],
        }

        _atomic_write_json(self.primary_path, primary_payload)
        _atomic_write_debug_payload(
            self.debug_path,
            video_path=self.video_path,
            session=session_dict,
            events=events_list,
            stop_event=stop_event,
        )
        return self.primary_path, self.debug_path
