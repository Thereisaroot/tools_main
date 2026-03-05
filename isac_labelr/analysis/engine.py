from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Event as ThreadEvent
from typing import Callable

from isac_labelr.io.video_stream import VideoStream
from isac_labelr.models import (
    AnalysisMode,
    AnalysisProgress,
    AnalysisRequest,
    AnalysisResult,
    EventRecord,
    OverlayTSStatus,
    ROI,
    Track,
    Vector2,
)
from isac_labelr.vision.detector import DetectorConfig, OnnxYoloPersonDetector
from isac_labelr.vision.ocr import TimestampOCR
from isac_labelr.vision.tracker import ByteTrackLite


@dataclass(slots=True)
class PresenceState:
    inside: bool = False
    enter_streak: int = 0
    exit_streak: int = 0
    last_center: tuple[float, float] | None = None


class AnalysisEngine:
    def __init__(self) -> None:
        self._last_overlay_ts: int | None = None

    @staticmethod
    def _coerce_mode(mode: AnalysisMode | str) -> AnalysisMode:
        if isinstance(mode, AnalysisMode):
            return mode
        if isinstance(mode, str):
            try:
                return AnalysisMode(mode)
            except ValueError:
                return AnalysisMode.ENTRY_ONLY
        return AnalysisMode.ENTRY_ONLY

    def run(
        self,
        request: AnalysisRequest,
        *,
        stop_event: ThreadEvent,
        on_progress: Callable[[AnalysisProgress], None],
        on_event: Callable[[EventRecord], None],
        logger: logging.Logger,
    ) -> AnalysisResult:
        video_stream = VideoStream(request.video_path)
        info = video_stream.info

        detector = OnnxYoloPersonDetector(
            DetectorConfig(
                model_path=request.detector_model_path,
                confidence=request.detector_confidence,
                iou=request.detector_iou,
            )
        )
        tracker = ByteTrackLite(iou_threshold=request.detector_iou)
        ocr = TimestampOCR()
        ocr.set_manual_roi(request.ocr_manual_roi)

        processed_events = 0
        presence: dict[tuple[str, int], PresenceState] = {}
        mode = self._coerce_mode(request.mode)

        start_ms = 0 if request.analyze_full else request.start_ms
        duration_ms = None if request.analyze_full else request.duration_ms

        logger.info("analysis_start video=%s", request.video_path)

        def on_chunk(chunk_idx: int, time_ms: int) -> None:
            logger.info("chunk_flush chunk=%s time_ms=%s events=%s", chunk_idx, time_ms, processed_events)

        for packet in video_stream.iter_frames(
            rotation_deg=request.rotation_deg,
            start_ms=start_ms,
            duration_ms=duration_ms,
            chunk_seconds=request.chunk_seconds,
            on_chunk=on_chunk,
        ):
            if stop_event.is_set():
                logger.info("analysis_aborted frame_index=%s", packet.frame_index)
                return AnalysisResult(
                    total_events=processed_events,
                    output_dir=Path("."),
                    aborted=True,
                )

            detections = detector.detect(packet.image_bgr)
            tracks = tracker.update(detections)
            events = self._update_presence(
                rois=request.rois,
                direction_vectors=request.direction_vectors,
                mode=mode,
                frame_index=packet.frame_index,
                video_time_ms=packet.time_ms,
                video_name=Path(request.video_path).name,
                tracks=tracks,
                presence=presence,
            )

            for event in events:
                ocr_result = ocr.extract_timestamp(packet.image_bgr)
                if ocr_result.success:
                    overlay_ts_ms = ocr_result.timestamp_ms
                    overlay_status = OverlayTSStatus.OK
                    self._last_overlay_ts = overlay_ts_ms
                elif self._last_overlay_ts is not None:
                    overlay_ts_ms = self._last_overlay_ts
                    overlay_status = OverlayTSStatus.INTERPOLATED_PREV
                else:
                    overlay_ts_ms = None
                    overlay_status = OverlayTSStatus.FAILED

                corrected_ts = (
                    overlay_ts_ms + request.timestamp_correction_ms
                    if overlay_ts_ms is not None
                    else None
                )
                event.overlay_ts_ms = overlay_ts_ms
                event.overlay_ts_status = overlay_status
                event.corrected_ts_ms = corrected_ts
                event.correction_ms = request.timestamp_correction_ms
                event.ocr_conf = ocr_result.confidence

                on_event(event)
                processed_events += 1

            on_progress(
                AnalysisProgress(
                    frame_index=packet.frame_index,
                    total_frames=info.total_frames,
                    video_time_ms=packet.time_ms,
                    processed_events=processed_events,
                )
            )

        logger.info("analysis_complete events=%s", processed_events)
        return AnalysisResult(total_events=processed_events, output_dir=Path("."), aborted=False)

    def _update_presence(
        self,
        *,
        rois: list[ROI],
        direction_vectors: dict[str, Vector2],
        mode: AnalysisMode,
        frame_index: int,
        video_time_ms: int,
        video_name: str,
        tracks: list[Track],
        presence: dict[tuple[str, int], PresenceState],
    ) -> list[EventRecord]:
        active_ids = {t.track_id for t in tracks}
        tracks_by_id = {t.track_id: t for t in tracks}
        events: list[EventRecord] = []

        for roi in rois:
            roi_n = roi.normalized()

            # Step 1: active tracks
            for track in tracks:
                center = track.center
                inside = roi_n.contains(center[0], center[1])

                key = (roi_n.roi_id, track.track_id)
                state = presence.setdefault(key, PresenceState())
                event = self._step_presence(
                    state=state,
                    inside=inside,
                    mode=mode,
                    track=track,
                    roi=roi_n,
                    direction=direction_vectors.get(roi_n.roi_id),
                    frame_index=frame_index,
                    video_time_ms=video_time_ms,
                    video_name=video_name,
                )
                if event is not None:
                    events.append(event)

            # Step 2: unseen tracks this frame -> treat as outside
            keys_for_roi = [k for k in presence if k[0] == roi_n.roi_id]
            for key in keys_for_roi:
                _, person_id = key
                if person_id in active_ids:
                    continue
                state = presence[key]
                event = self._step_presence(
                    state=state,
                    inside=False,
                    mode=mode,
                    track=tracks_by_id.get(person_id),
                    roi=roi_n,
                    direction=direction_vectors.get(roi_n.roi_id),
                    frame_index=frame_index,
                    video_time_ms=video_time_ms,
                    video_name=video_name,
                    person_id_override=person_id,
                )
                if event is not None:
                    events.append(event)

        return events

    def _step_presence(
        self,
        *,
        state: PresenceState,
        inside: bool,
        mode: AnalysisMode,
        track: Track | None,
        roi: ROI,
        direction: Vector2 | None,
        frame_index: int,
        video_time_ms: int,
        video_name: str,
        person_id_override: int | None = None,
    ) -> EventRecord | None:
        debounce = 3
        person_id = person_id_override if person_id_override is not None else (track.track_id if track else -1)
        conf = track.confidence if track else 0.0

        if inside:
            state.enter_streak += 1
            state.exit_streak = 0
            state.last_center = track.center if track else state.last_center
            if not state.inside and state.enter_streak >= debounce:
                state.inside = True
                return EventRecord(
                    event_id=str(uuid.uuid4()),
                    video_name=video_name,
                    roi_id=roi.roi_id,
                    person_id=person_id,
                    event_type="enter",
                    direction_label=self._direction_label(track, direction),
                    frame_index=frame_index,
                    video_time_ms=video_time_ms,
                    overlay_ts_ms=None,
                    overlay_ts_status=OverlayTSStatus.FAILED,
                    correction_ms=0,
                    corrected_ts_ms=None,
                    det_conf=conf,
                    ocr_conf=None,
                )
            return None

        state.exit_streak += 1
        state.enter_streak = 0

        if state.inside and mode == AnalysisMode.ENTRY_EXIT_DIRECTION and state.exit_streak >= debounce:
            state.inside = False
            return EventRecord(
                event_id=str(uuid.uuid4()),
                video_name=video_name,
                roi_id=roi.roi_id,
                person_id=person_id,
                event_type="exit",
                direction_label=self._direction_label(track, direction),
                frame_index=frame_index,
                video_time_ms=video_time_ms,
                overlay_ts_ms=None,
                overlay_ts_status=OverlayTSStatus.FAILED,
                correction_ms=0,
                corrected_ts_ms=None,
                det_conf=conf,
                ocr_conf=None,
            )

        return None

    @staticmethod
    def _direction_label(track: Track | None, direction: Vector2 | None) -> str | None:
        if track is None or direction is None or track.prev_center is None:
            return None
        curr = track.center
        prev = track.prev_center
        mv_x = curr[0] - prev[0]
        mv_y = curr[1] - prev[1]
        dot = mv_x * direction.dx + mv_y * direction.dy
        return "forward" if dot >= 0 else "reverse"
