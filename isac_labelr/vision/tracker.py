from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isac_labelr.models import Detection, Track


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    missed: int = 0
    prev_center: tuple[float, float] | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h

    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union


class ByteTrackLite:
    """간단한 IoU 기반 ByteTrack-like 추적기."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def update(self, detections: list[Detection]) -> list[Track]:
        track_ids = list(self._tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_det = set(range(len(detections)))

        pairs: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            t = self._tracks[track_id]
            for d_idx, det in enumerate(detections):
                pairs.append((_iou(t.bbox, det.bbox), track_id, d_idx))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for iou_score, track_id, d_idx in pairs:
            if iou_score < self.iou_threshold:
                continue
            if track_id not in unmatched_tracks or d_idx not in unmatched_det:
                continue
            t = self._tracks[track_id]
            t.prev_center = t.center
            t.bbox = detections[d_idx].bbox
            t.confidence = detections[d_idx].confidence
            t.missed = 0
            unmatched_tracks.remove(track_id)
            unmatched_det.remove(d_idx)

        for track_id in list(unmatched_tracks):
            t = self._tracks[track_id]
            t.missed += 1
            if t.missed > self.max_missed:
                del self._tracks[track_id]

        for d_idx in unmatched_det:
            det = detections[d_idx]
            self._tracks[self._next_id] = _TrackState(
                track_id=self._next_id,
                bbox=det.bbox,
                confidence=det.confidence,
                missed=0,
                prev_center=None,
            )
            self._next_id += 1

        tracks: list[Track] = []
        for t in self._tracks.values():
            if t.missed > 0:
                continue
            tracks.append(
                Track(
                    track_id=t.track_id,
                    bbox=t.bbox,
                    confidence=t.confidence,
                    prev_center=t.prev_center,
                )
            )
        return tracks
