from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from isac_labelr.models import Detection

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional runtime import
    ort = None


@dataclass(slots=True)
class DetectorConfig:
    model_path: str
    confidence: float = 0.4
    iou: float = 0.5


class DetectorInitializationError(RuntimeError):
    pass


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(1e-6, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = np.maximum(1e-6, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou_xyxy(boxes[[i]], boxes[order[1:]]).reshape(-1)
        inds = np.where(ious <= iou_thr)[0]
        order = order[inds + 1]
    return keep


class OnnxYoloPersonDetector:
    """YOLO ONNX 추론기 (person class=0 전용)."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        if ort is None:
            raise DetectorInitializationError(
                "onnxruntime is not installed. Install requirements first."
            )

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise DetectorInitializationError(f"Model not found: {model_path}")

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # Typical: [1, 3, 640, 640]
        self.input_h = int(shape[2]) if isinstance(shape[2], int) else 640
        self.input_w = int(shape[3]) if isinstance(shape[3], int) else 640

    def _preprocess(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)

        pad_x = (self.input_w - nw) // 2
        pad_y = (self.input_h - nh) // 2
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor, scale, pad_x, pad_y, w, h

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        tensor, scale, pad_x, pad_y, src_w, src_h = self._preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        raw = outputs[0]

        boxes, scores = self._decode(raw)
        if boxes.size == 0:
            return []

        # Undo letterbox
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= max(scale, 1e-6)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, src_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, src_h - 1)

        keep = _nms(boxes, scores, self.config.iou)
        detections: list[Detection] = []
        for idx in keep:
            detections.append(
                Detection(
                    bbox=(
                        float(boxes[idx, 0]),
                        float(boxes[idx, 1]),
                        float(boxes[idx, 2]),
                        float(boxes[idx, 3]),
                    ),
                    confidence=float(scores[idx]),
                )
            )
        return detections

    def _decode(self, output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Supports common YOLO ONNX outputs:
        - [1, N, 6] = x1,y1,x2,y2,score,class
        - [1, N, 85] or [1, 85, N] = x,y,w,h,obj,cls...
        - [1, N, 84] or [1, 84, N] = x,y,w,h,cls... (YOLOv8/11 no objectness)
        """
        out = np.array(output)
        if out.ndim == 3:
            out = out[0]
            if out.shape[0] < out.shape[1] and out.shape[0] <= 128:
                out = out.T

        if out.ndim != 2 or out.shape[1] < 6:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        if out.shape[1] == 6:
            # x1,y1,x2,y2,score,class
            cls = out[:, 5]
            person = cls == 0
            scores = out[:, 4]
            mask = person & (scores >= self.config.confidence)
            return out[mask, :4].astype(np.float32), scores[mask].astype(np.float32)

        # Try both formats and choose the one producing stronger person detections.
        xywh = out[:, 0:4]
        num_cols = out.shape[1]

        def build_candidates(conf: np.ndarray, cls_idx: np.ndarray):
            person_mask = cls_idx == 0
            score_mask = conf >= self.config.confidence
            mask = person_mask & score_mask
            if not np.any(mask):
                return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

            selected = xywh[mask]
            scores = conf[mask]
            x, y, w, h = (
                selected[:, 0],
                selected[:, 1],
                selected[:, 2],
                selected[:, 3],
            )
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
            return boxes, scores.astype(np.float32)

        candidates: list[tuple[np.ndarray, np.ndarray]] = []

        # Format A: x,y,w,h,cls... (no objectness)
        if num_cols >= 5:
            cls_scores_a = out[:, 4:]
            if cls_scores_a.shape[1] > 0:
                cls_idx_a = np.argmax(cls_scores_a, axis=1)
                conf_a = cls_scores_a[np.arange(len(out)), cls_idx_a]
                candidates.append(build_candidates(conf_a, cls_idx_a))

        # Format B: x,y,w,h,obj,cls...
        if num_cols >= 6:
            obj = out[:, 4]
            cls_scores_b = out[:, 5:]
            if cls_scores_b.shape[1] > 0:
                cls_idx_b = np.argmax(cls_scores_b, axis=1)
                cls_conf_b = cls_scores_b[np.arange(len(out)), cls_idx_b]
                conf_b = obj * cls_conf_b
                candidates.append(build_candidates(conf_b, cls_idx_b))

        if not candidates:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Pick the richer candidate set first, then stronger max score.
        best_boxes = np.empty((0, 4), dtype=np.float32)
        best_scores = np.empty((0,), dtype=np.float32)
        for boxes, scores in candidates:
            if len(scores) > len(best_scores):
                best_boxes, best_scores = boxes, scores
            elif len(scores) == len(best_scores) and len(scores) > 0:
                if float(np.max(scores)) > (float(np.max(best_scores)) if len(best_scores) else -1.0):
                    best_boxes, best_scores = boxes, scores

        return best_boxes, best_scores
