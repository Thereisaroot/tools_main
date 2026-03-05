from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class AnalysisMode(StrEnum):
    ENTRY_ONLY = "entry_only"
    ENTRY_EXIT_DIRECTION = "entry_exit_direction"


class OCRRoiMode(StrEnum):
    AUTO_WITH_MANUAL_FALLBACK = "auto_with_manual_fallback"


class OverlayTSStatus(StrEnum):
    OK = "ok"
    INTERPOLATED_PREV = "interpolated_prev"
    FAILED = "failed"


@dataclass(slots=True)
class ROI:
    roi_id: str
    x: int
    y: int
    w: int
    h: int

    def normalized(self) -> "ROI":
        x = self.x
        y = self.y
        w = self.w
        h = self.h
        if w < 0:
            x = x + w
            w = -w
        if h < 0:
            y = y + h
            h = -h
        return ROI(roi_id=self.roi_id, x=x, y=y, w=w, h=h)

    def contains(self, px: float, py: float) -> bool:
        n = self.normalized()
        return n.x <= px <= n.x + n.w and n.y <= py <= n.y + n.h

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Vector2:
    dx: float
    dy: float

    def to_dict(self) -> dict[str, float]:
        return {"dx": self.dx, "dy": self.dy}


@dataclass(slots=True)
class AnalysisRequest:
    video_path: str
    mode: AnalysisMode | str
    rotation_deg: int
    rois: list[ROI]
    direction_vectors: dict[str, Vector2]
    analyze_full: bool
    start_ms: int = 0
    duration_ms: int | None = None
    timestamp_correction_ms: int = 0
    ocr_roi_mode: OCRRoiMode = OCRRoiMode.AUTO_WITH_MANUAL_FALLBACK
    ocr_manual_roi: ROI | None = None
    detector_model_path: str = "models/person_detector.onnx"
    detector_confidence: float = 0.25
    detector_iou: float = 0.5
    chunk_seconds: int = 60


@dataclass(slots=True)
class EventRecord:
    event_id: str
    video_name: str
    roi_id: str
    person_id: int
    event_type: str
    direction_label: str | None
    frame_index: int
    video_time_ms: int
    overlay_ts_ms: int | None
    overlay_ts_status: OverlayTSStatus
    correction_ms: int
    corrected_ts_ms: int | None
    det_conf: float
    ocr_conf: float | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["overlay_ts_status"] = self.overlay_ts_status.value
        return data


@dataclass(slots=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    prev_center: tuple[float, float] | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(slots=True)
class FramePacket:
    frame_index: int
    time_ms: int
    image_bgr: Any


@dataclass(slots=True)
class AnalysisProgress:
    frame_index: int
    total_frames: int
    video_time_ms: int
    processed_events: int


@dataclass(slots=True)
class AnalysisResult:
    total_events: int
    output_dir: Path
    aborted: bool = False


@dataclass(slots=True)
class AppPreferences:
    detector_model_path: str = "models/person_detector.onnx"
    detector_confidence: float = 0.25
    detector_iou: float = 0.5
    chunk_seconds: int = 60

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppPreferences":
        return cls(
            detector_model_path=str(data.get("detector_model_path", cls.detector_model_path)),
            detector_confidence=float(data.get("detector_confidence", cls.detector_confidence)),
            detector_iou=float(data.get("detector_iou", cls.detector_iou)),
            chunk_seconds=int(data.get("chunk_seconds", cls.chunk_seconds)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SessionConfig:
    video_path: str
    rotation_deg: int
    mode: AnalysisMode | str
    rois: list[ROI] = field(default_factory=list)
    direction_vectors: dict[str, Vector2] = field(default_factory=dict)
    analyze_full: bool = True
    start_ms: int = 0
    duration_ms: int | None = None
    timestamp_correction_ms: int = 0
    ocr_manual_roi: ROI | None = None

    def to_dict(self) -> dict[str, Any]:
        mode_value = self.mode.value if isinstance(self.mode, AnalysisMode) else str(self.mode)
        return {
            "video_path": self.video_path,
            "rotation_deg": self.rotation_deg,
            "mode": mode_value,
            "rois": [roi.to_dict() for roi in self.rois],
            "direction_vectors": {
                roi_id: vec.to_dict() for roi_id, vec in self.direction_vectors.items()
            },
            "analyze_full": self.analyze_full,
            "start_ms": self.start_ms,
            "duration_ms": self.duration_ms,
            "timestamp_correction_ms": self.timestamp_correction_ms,
            "ocr_manual_roi": self.ocr_manual_roi.to_dict() if self.ocr_manual_roi else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionConfig":
        rois = [ROI(**roi) for roi in data.get("rois", [])]
        direction_vectors = {
            roi_id: Vector2(**vec) for roi_id, vec in data.get("direction_vectors", {}).items()
        }
        ocr_raw = data.get("ocr_manual_roi")
        ocr_roi = ROI(**ocr_raw) if ocr_raw else None
        return cls(
            video_path=str(data.get("video_path", "")),
            rotation_deg=int(data.get("rotation_deg", 0)),
            mode=AnalysisMode(data.get("mode", AnalysisMode.ENTRY_ONLY.value)),
            rois=rois,
            direction_vectors=direction_vectors,
            analyze_full=bool(data.get("analyze_full", True)),
            start_ms=int(data.get("start_ms", 0)),
            duration_ms=(
                int(data["duration_ms"]) if data.get("duration_ms") is not None else None
            ),
            timestamp_correction_ms=int(data.get("timestamp_correction_ms", 0)),
            ocr_manual_roi=ocr_roi,
        )
