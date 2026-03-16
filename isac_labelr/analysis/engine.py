from __future__ import annotations

import gc
import logging
import os
import sys
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Event as ThreadEvent
from typing import Callable

from isac_labelr.io.video_stream import VideoStream
from isac_labelr.monitor.memory import get_memory_snapshot
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
    pending_enter_frame_index: int | None = None
    pending_enter_video_time_ms: int | None = None
    pending_exit_frame_index: int | None = None
    pending_exit_video_time_ms: int | None = None


class AnalysisEngine:
    def __init__(self) -> None:
        self._last_overlay_ts: int | None = None
        self._last_overlay_video_time_ms: int | None = None
        self._last_overlay_raw_text: str = ""
        self._last_ocr_frame_index: int | None = None
        self._allow_neighbor_interp = (
            str(os.getenv("ISAC_OCR_NEIGHBOR_INTERP", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._ts_diag_enabled = (
            str(os.getenv("ISAC_TS_DIAG", "0")).strip().lower() in {"1", "true", "yes", "on"}
        )
        self._mem_diag_enabled = (
            str(os.getenv("ISAC_MEM_DIAG", "1")).strip().lower() in {"1", "true", "yes", "on"}
            and (sys.platform != "win32")
        )

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
        detector_cfg = DetectorConfig(
            model_path=request.detector_model_path,
            confidence=request.detector_confidence,
            iou=request.detector_iou,
        )
        detector_reset_interval = max(
            0,
            int(os.getenv("ISAC_DETECTOR_RESET_INTERVAL", "20")),
        )
        ocr_reset_interval = max(
            0,
            int(os.getenv("ISAC_OCR_RESET_INTERVAL", "20")),
        )
        detector = OnnxYoloPersonDetector(detector_cfg)
        if detector.backend == "onnxruntime" and detector.providers:
            logger.info("detector_provider providers=%s", ",".join(detector.providers))
        force_hard_detector_reset = (
            str(os.getenv("ISAC_DETECTOR_HARD_RESET", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if detector.backend == "onnxruntime" and not force_hard_detector_reset:
            if detector_reset_interval > 0:
                logger.info(
                    "detector_reset_disabled backend=%s interval=%s",
                    detector.backend,
                    detector_reset_interval,
                )
            detector_reset_interval = 0
        tracker = ByteTrackLite(iou_threshold=request.detector_iou)

        processed_events = 0
        ocr_uses = 0
        ocr_resets = 0
        presence: dict[tuple[str, int], PresenceState] = {}
        mode = self._coerce_mode(request.mode)
        self._last_overlay_ts = None
        self._last_overlay_video_time_ms = None
        self._last_overlay_raw_text = ""
        self._last_ocr_frame_index = None

        start_ms = 0 if request.analyze_full else request.start_ms
        duration_ms = None if request.analyze_full else request.duration_ms

        logger.info("analysis_start video=%s", request.video_path)

        def on_chunk(chunk_idx: int, time_ms: int) -> None:
            logger.info(
                "chunk_flush chunk=%s time_ms=%s events=%s ocr_calls=%s ocr_resets=%s",
                chunk_idx,
                time_ms,
                processed_events,
                ocr_uses,
                ocr_resets,
            )

        processed_frames = 0
        frame_cache: OrderedDict[int, object] = OrderedDict()
        frame_cache_size = max(2, int(os.getenv("ISAC_FRAME_CACHE_SIZE", "6")))
        try:
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

                processed_frames += 1
                frame_cache[int(packet.frame_index)] = packet.image_bgr
                while len(frame_cache) > frame_cache_size:
                    frame_cache.popitem(last=False)
                if detector_reset_interval > 0 and processed_frames > 1:
                    if processed_frames % detector_reset_interval == 0:
                        old = detector
                        detector = OnnxYoloPersonDetector(detector_cfg)
                        try:
                            old.close()
                        except Exception:
                            pass
                        del old
                        gc.collect()
                        logger.info(
                            "detector_reset frame=%s interval=%s",
                            packet.frame_index,
                            detector_reset_interval,
                        )
                detections = detector.detect(packet.image_bgr)
                tracks = tracker.update(detections)
                events = self._update_presence(
                    rois=request.rois,
                    direction_vectors=request.direction_vectors,
                    rotation_deg=request.rotation_deg,
                    mode=mode,
                    enter_debounce_frames=max(1, int(request.enter_debounce_frames)),
                    exit_debounce_frames=max(1, int(request.exit_debounce_frames)),
                    frame_index=packet.frame_index,
                    video_time_ms=packet.time_ms,
                    video_name=Path(request.video_path).name,
                    tracks=tracks,
                    presence=presence,
                )

                if events:
                    for event in events:
                        confirmed_frame_index = int(
                            event.confirmed_frame_index
                            if event.confirmed_frame_index is not None
                            else packet.frame_index
                        )
                        confirmed_video_time_ms = int(
                            event.confirmed_video_time_ms
                            if event.confirmed_video_time_ms is not None
                            else packet.time_ms
                        )
                        event.overlay_ts_ms = None
                        event.overlay_ts_status = OverlayTSStatus.FAILED
                        event.corrected_ts_ms = None
                        event.correction_ms = request.timestamp_correction_ms
                        event.ocr_conf = None
                        event.ocr_raw_text = ""
                        event.ocr_frame_index = int(event.frame_index)
                        event.confirmed_frame_index = confirmed_frame_index
                        event.confirmed_video_time_ms = confirmed_video_time_ms
                        end_frame = int(event.frame_index) + 100
                        if int(info.total_frames) > 0:
                            end_frame = min(end_frame, int(info.total_frames) - 1)
                        event.label_end_frame = max(int(event.frame_index), int(end_frame))
                        event.label_end_timestamp_unix = None
                        if self._ts_diag_enabled:
                            logger.info(
                                (
                                    "ts_diag frame=%s confirmed_frame=%s ocr_frame=%s "
                                    "video_time_ms=%s confirmed_video_time_ms=%s "
                                    "roi=%s person=%s type=%s overlay=%s corrected=%s status=%s raw=%r"
                                ),
                                event.frame_index,
                                event.confirmed_frame_index,
                                event.ocr_frame_index,
                                event.video_time_ms,
                                event.confirmed_video_time_ms,
                                event.roi_id,
                                event.person_id,
                                event.event_type,
                                event.overlay_ts_ms,
                                event.corrected_ts_ms,
                                event.overlay_ts_status.value,
                                event.ocr_raw_text,
                            )

                        on_event(event)
                        processed_events += 1

                if processed_frames % 600 == 0:
                    gc.collect()
                if self._mem_diag_enabled and processed_frames % 300 == 0:
                    snap = get_memory_snapshot()
                    logger.info(
                        (
                            "mem_diag frame=%s rss_self_mb=%.1f rss_tree_mb=%.1f footprint_mb=%s child=%s src=%s "
                            "presence=%s events=%s ocr_calls=%s ocr_resets=%s"
                        ),
                        packet.frame_index,
                        snap.rss_mb,
                        snap.tree_rss_mb,
                        (
                            f"{snap.phys_footprint_mb:.1f}"
                            if snap.phys_footprint_mb is not None
                            else "-"
                        ),
                        snap.child_count,
                        snap.backend,
                        len(presence),
                        processed_events,
                        ocr_uses,
                        ocr_resets,
                    )

                on_progress(
                    AnalysisProgress(
                        frame_index=packet.frame_index,
                        total_frames=info.total_frames,
                        video_time_ms=packet.time_ms,
                        processed_events=processed_events,
                    )
                )
        finally:
            try:
                detector.close()
            except Exception:
                pass

        logger.info(
            "analysis_complete events=%s ocr_calls=%s ocr_resets=%s",
            processed_events,
            ocr_uses,
            ocr_resets,
        )
        return AnalysisResult(total_events=processed_events, output_dir=Path("."), aborted=False)

    def _update_presence(
        self,
        *,
        rois: list[ROI],
        direction_vectors: dict[str, Vector2],
        rotation_deg: int,
        mode: AnalysisMode,
        frame_index: int,
        video_time_ms: int,
        video_name: str,
        tracks: list[Track],
        presence: dict[tuple[str, int], PresenceState],
        enter_debounce_frames: int = 1,
        exit_debounce_frames: int = 2,
    ) -> list[EventRecord]:
        active_ids = {t.track_id for t in tracks}
        tracks_by_id = {t.track_id: t for t in tracks}
        events: list[EventRecord] = []
        visible_person_count = len(tracks)
        roi_person_counts: dict[str, int] = {}
        prune_exit_streak = 30

        for roi in rois:
            roi_n = roi.normalized()
            count = 0
            for track in tracks:
                fx, fy = track.foot_point
                if roi_n.contains(fx, fy):
                    count += 1
            roi_person_counts[roi_n.roi_id] = count

        for roi in rois:
            roi_n = roi.normalized()
            roi_person_count = roi_person_counts.get(roi_n.roi_id, 0)

            # Step 1: active tracks
            for track in tracks:
                foot = track.foot_point
                inside = roi_n.contains(foot[0], foot[1])

                key = (roi_n.roi_id, track.track_id)
                state = presence.setdefault(key, PresenceState())
                event = self._step_presence(
                    state=state,
                    inside=inside,
                    mode=mode,
                    enter_debounce_frames=enter_debounce_frames,
                    exit_debounce_frames=exit_debounce_frames,
                    rotation_deg=rotation_deg,
                    track=track,
                    roi=roi_n,
                    direction=direction_vectors.get(roi_n.roi_id),
                    frame_index=frame_index,
                    video_time_ms=video_time_ms,
                    video_name=video_name,
                    visible_person_count=visible_person_count,
                    roi_person_count=roi_person_count,
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
                    enter_debounce_frames=enter_debounce_frames,
                    exit_debounce_frames=exit_debounce_frames,
                    rotation_deg=rotation_deg,
                    track=tracks_by_id.get(person_id),
                    roi=roi_n,
                    direction=direction_vectors.get(roi_n.roi_id),
                    frame_index=frame_index,
                    video_time_ms=video_time_ms,
                    video_name=video_name,
                    person_id_override=person_id,
                    visible_person_count=visible_person_count,
                    roi_person_count=roi_person_count,
                )
                if event is not None:
                    events.append(event)
                if not state.inside and state.exit_streak >= prune_exit_streak:
                    presence.pop(key, None)

        if len(presence) > 100_000:
            target = 80_000
            removable = [
                key
                for key, state in presence.items()
                if (not state.inside) and state.exit_streak >= 5
            ]
            for key in removable[: max(0, len(presence) - target)]:
                presence.pop(key, None)

        return events

    def _step_presence(
        self,
        *,
        state: PresenceState,
        inside: bool,
        mode: AnalysisMode,
        rotation_deg: int,
        track: Track | None,
        roi: ROI,
        direction: Vector2 | None,
        frame_index: int,
        video_time_ms: int,
        video_name: str,
        visible_person_count: int = 0,
        roi_person_count: int = 0,
        person_id_override: int | None = None,
        enter_debounce_frames: int = 1,
        exit_debounce_frames: int = 2,
    ) -> EventRecord | None:
        enter_debounce = max(1, int(enter_debounce_frames))
        exit_debounce = max(1, int(exit_debounce_frames))
        person_id = person_id_override if person_id_override is not None else (track.track_id if track else -1)
        conf = track.confidence if track else 0.0

        if inside:
            if state.enter_streak == 0:
                state.pending_enter_frame_index = int(frame_index)
                state.pending_enter_video_time_ms = int(video_time_ms)
            state.enter_streak += 1
            state.exit_streak = 0
            state.pending_exit_frame_index = None
            state.pending_exit_video_time_ms = None
            state.last_center = track.center if track else state.last_center
            if not state.inside and state.enter_streak >= enter_debounce:
                state.inside = True
                anchor_frame_index = (
                    int(state.pending_enter_frame_index)
                    if state.pending_enter_frame_index is not None
                    else int(frame_index)
                )
                anchor_video_time_ms = (
                    int(state.pending_enter_video_time_ms)
                    if state.pending_enter_video_time_ms is not None
                    else int(video_time_ms)
                )
                state.pending_enter_frame_index = None
                state.pending_enter_video_time_ms = None
                return EventRecord(
                    event_id=str(uuid.uuid4()),
                    video_name=video_name,
                    roi_id=roi.roi_id,
                    person_id=person_id,
                    event_type="enter",
                    direction_label=self._direction_label(track, direction),
                    frame_index=anchor_frame_index,
                    video_time_ms=anchor_video_time_ms,
                    overlay_ts_ms=None,
                    overlay_ts_status=OverlayTSStatus.FAILED,
                    correction_ms=0,
                    corrected_ts_ms=None,
                    det_conf=conf,
                    ocr_conf=None,
                    ocr_raw_text="",
                    visible_person_count=visible_person_count,
                    roi_person_count=roi_person_count,
                    confirmed_frame_index=int(frame_index),
                    confirmed_video_time_ms=int(video_time_ms),
                    ocr_frame_index=anchor_frame_index,
                    rotation_deg=int(rotation_deg) % 360,
                    roi_x=roi.x,
                    roi_y=roi.y,
                    roi_w=roi.w,
                    roi_h=roi.h,
                )
            return None

        state.exit_streak += 1
        state.enter_streak = 0
        state.pending_enter_frame_index = None
        state.pending_enter_video_time_ms = None
        if state.exit_streak == 1:
            state.pending_exit_frame_index = int(frame_index)
            state.pending_exit_video_time_ms = int(video_time_ms)

        if state.inside and mode == AnalysisMode.ENTRY_EXIT_DIRECTION and state.exit_streak >= exit_debounce:
            state.inside = False
            anchor_frame_index = (
                int(state.pending_exit_frame_index)
                if state.pending_exit_frame_index is not None
                else int(frame_index)
            )
            anchor_video_time_ms = (
                int(state.pending_exit_video_time_ms)
                if state.pending_exit_video_time_ms is not None
                else int(video_time_ms)
            )
            state.pending_exit_frame_index = None
            state.pending_exit_video_time_ms = None
            return EventRecord(
                event_id=str(uuid.uuid4()),
                video_name=video_name,
                roi_id=roi.roi_id,
                person_id=person_id,
                event_type="exit",
                direction_label=self._direction_label(track, direction),
                frame_index=anchor_frame_index,
                video_time_ms=anchor_video_time_ms,
                overlay_ts_ms=None,
                overlay_ts_status=OverlayTSStatus.FAILED,
                correction_ms=0,
                corrected_ts_ms=None,
                det_conf=conf,
                ocr_conf=None,
                ocr_raw_text="",
                visible_person_count=visible_person_count,
                roi_person_count=roi_person_count,
                confirmed_frame_index=int(frame_index),
                confirmed_video_time_ms=int(video_time_ms),
                ocr_frame_index=anchor_frame_index,
                rotation_deg=int(rotation_deg) % 360,
                roi_x=roi.x,
                roi_y=roi.y,
                roi_w=roi.w,
                roi_h=roi.h,
            )

        return None

    def _resolve_event_overlay_timestamp(
        self,
        *,
        ocr: TimestampOCR,
        video_path: str,
        rotation_deg: int,
        anchor_frame_index: int,
        anchor_video_time_ms: int,
        anchor_frame_bgr,
        fallback_frame_index: int,
        fallback_video_time_ms: int,
        fallback_frame_bgr,
        logger: logging.Logger,
    ) -> tuple[int | None, OverlayTSStatus, float | None, str, int]:
        anchor_overlay, anchor_status, anchor_conf, anchor_raw = self._resolve_overlay_timestamp(
            ocr=ocr,
            frame_bgr=anchor_frame_bgr,
            video_path=video_path,
            frame_index=anchor_frame_index,
            video_time_ms=anchor_video_time_ms,
            rotation_deg=rotation_deg,
            logger=logger,
            allow_interpolation=False,
        )
        if anchor_status == OverlayTSStatus.OK and anchor_overlay is not None:
            self._last_overlay_ts = int(anchor_overlay)
            self._last_overlay_video_time_ms = int(anchor_video_time_ms)
            return anchor_overlay, anchor_status, anchor_conf, anchor_raw, int(anchor_frame_index)
        return anchor_overlay, anchor_status, anchor_conf, anchor_raw, int(anchor_frame_index)

    def _resolve_overlay_timestamp(
        self,
        *,
        ocr: TimestampOCR,
        frame_bgr,
        video_path: str,
        frame_index: int,
        video_time_ms: int,
        rotation_deg: int,
        logger: logging.Logger,
        allow_interpolation: bool = True,
    ) -> tuple[int | None, OverlayTSStatus, float | None, str]:
        def run_ocr_once(fast_mode: bool):
            if frame_bgr is None:
                return ocr._extract_timestamp_for_video_frame(
                    video_path=video_path,
                    frame_index=frame_index,
                    rotation_deg=rotation_deg,
                    fast=fast_mode,
                )
            result = ocr.extract_timestamp(frame_bgr, fast=fast_mode)
            ocr.cache_video_frame_result(
                video_path=video_path,
                frame_index=frame_index,
                rotation_deg=rotation_deg,
                fast=fast_mode,
                result=result,
            )
            return result

        ocr_result = run_ocr_once(True)
        candidates = [
            ts
            for ts in ocr.extract_timestamp_candidates(ocr_result.raw_text)
            if ocr.is_valid_timestamp(ts)
        ]
        primary = int(ocr_result.timestamp_ms) if ocr.is_valid_timestamp(ocr_result.timestamp_ms) else None
        if primary is not None:
            picked = int(primary)
            self._last_overlay_ts = picked
            self._last_overlay_video_time_ms = video_time_ms
            return picked, OverlayTSStatus.OK, ocr_result.confidence, ocr_result.raw_text

        if candidates:
            picked = int(candidates[0])
            self._last_overlay_ts = picked
            self._last_overlay_video_time_ms = video_time_ms
            return picked, OverlayTSStatus.OK, ocr_result.confidence, ocr_result.raw_text

        # Fast OCR miss: retry once with richer preprocessing to reduce empty raw_text events.
        ocr_result_slow = run_ocr_once(False)
        slow_candidates = [
            ts
            for ts in ocr.extract_timestamp_candidates(ocr_result_slow.raw_text)
            if ocr.is_valid_timestamp(ts)
        ]
        slow_primary = (
            int(ocr_result_slow.timestamp_ms)
            if ocr.is_valid_timestamp(ocr_result_slow.timestamp_ms)
            else None
        )
        if slow_primary is not None:
            picked = int(slow_primary)
            self._last_overlay_ts = picked
            self._last_overlay_video_time_ms = video_time_ms
            return picked, OverlayTSStatus.OK, ocr_result_slow.confidence, ocr_result_slow.raw_text
        if slow_candidates:
            picked = int(slow_candidates[0])
            self._last_overlay_ts = picked
            self._last_overlay_video_time_ms = video_time_ms
            return picked, OverlayTSStatus.OK, ocr_result_slow.confidence, ocr_result_slow.raw_text
        if (not ocr_result.raw_text) and ocr_result_slow.raw_text:
            ocr_result = ocr_result_slow

        return None, OverlayTSStatus.FAILED, ocr_result.confidence, ocr_result.raw_text

    def _expected_overlay_ts(self, video_time_ms: int) -> tuple[int | None, int]:
        if self._last_overlay_ts is None or self._last_overlay_video_time_ms is None:
            return None, 0
        delta_video = int(video_time_ms) - int(self._last_overlay_video_time_ms)
        if delta_video < 0:
            return int(self._last_overlay_ts), 1500
        expected = int(self._last_overlay_ts) + delta_video
        tolerance = max(1500, int(delta_video * 4 + 500))
        return expected, tolerance

    def _is_plausible_timestamp(self, candidate_ts: int, video_time_ms: int) -> bool:
        if self._last_overlay_ts is None or self._last_overlay_video_time_ms is None:
            return True
        delta_video = int(video_time_ms) - int(self._last_overlay_video_time_ms)
        # Out-of-order frame timestamp in stream; don't over-filter.
        if delta_video < 0:
            return True
        expected = int(self._last_overlay_ts) + delta_video
        # Allow generous drift (clock jitter/OCR noise), but reject gross digit-shift errors.
        tolerance = max(1500, int(delta_video * 4 + 500))
        return abs(int(candidate_ts) - expected) <= tolerance

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
