from __future__ import annotations

from dataclasses import dataclass

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

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 30,
        max_tracks: int = 4000,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_tracks = max_tracks
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def update(self, detections: list[Detection]) -> list[Track]:
        if self.max_tracks > 0 and len(self._tracks) > self.max_tracks:
            # Drop stalest/lowest-confidence tracks first to cap memory.
            victims = sorted(
                self._tracks.items(),
                key=lambda kv: (kv[1].missed, -kv[1].confidence),
                reverse=True,
            )[: max(0, len(self._tracks) - self.max_tracks)]
            for track_id, _ in victims:
                self._tracks.pop(track_id, None)

        track_ids = list(self._tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_det = set(range(len(detections)))

        # Greedy matching without allocating a giant global (track x det) pair list.
        for track_id in track_ids:
            if track_id not in unmatched_tracks:
                continue
            if not unmatched_det:
                break

            t = self._tracks[track_id]
            best_det = None
            best_iou = self.iou_threshold
            for d_idx in unmatched_det:
                iou_score = _iou(t.bbox, detections[d_idx].bbox)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_det = d_idx

            if best_det is None:
                continue

            t.prev_center = t.center
            t.bbox = detections[best_det].bbox
            t.confidence = detections[best_det].confidence
            t.missed = 0
            unmatched_tracks.remove(track_id)
            unmatched_det.remove(best_det)

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
