from __future__ import annotations

import re
from shutil import which
from dataclasses import dataclass

import cv2
import numpy as np

from isac_labelr.models import ROI

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover - optional runtime import
    RapidOCR = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional runtime import
    pytesseract = None


TS_REGEX = re.compile(r"(\d{13})")


@dataclass(slots=True)
class OCRResult:
    timestamp_ms: int | None
    confidence: float | None
    success: bool
    roi: ROI | None
    raw_text: str


class TimestampOCR:
    def __init__(self) -> None:
        self._backend = "none"
        self._engine = None

        if RapidOCR is not None:
            self._engine = RapidOCR()
            self._backend = "rapidocr"
        elif pytesseract is not None and which("tesseract"):
            self._backend = "pytesseract"

        self._manual_roi: ROI | None = None
        self._auto_roi: ROI | None = None

    @property
    def manual_roi(self) -> ROI | None:
        return self._manual_roi

    @property
    def backend(self) -> str:
        return self._backend

    def set_manual_roi(self, roi: ROI | None) -> None:
        self._manual_roi = roi

    def clear_auto_roi(self) -> None:
        self._auto_roi = None

    def detect_auto_roi(self, frame_bgr: np.ndarray) -> ROI | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bright = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_roi: ROI | None = None
        best_score = -1.0

        h, w = gray.shape[:2]
        min_area = max(200.0, (w * h) * 0.001)

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < min_area:
                continue
            if ch <= 0 or cw <= 0:
                continue
            aspect = cw / max(1, ch)
            if aspect < 2.0 or aspect > 20.0:
                continue

            patch = gray[y : y + ch, x : x + cw]
            dark_ratio = float((patch < 120).sum()) / float(max(1, patch.size))
            score = dark_ratio * area
            if score > best_score:
                best_score = score
                best_roi = ROI("timestamp_auto", int(x), int(y), int(cw), int(ch))

        self._auto_roi = best_roi
        return best_roi

    def _crop_roi(self, frame_bgr: np.ndarray, roi: ROI | None) -> tuple[np.ndarray, ROI | None]:
        if roi is None:
            h, w = frame_bgr.shape[:2]
            top_band_h = max(40, h // 8)
            default_roi = ROI("timestamp_default", 0, 0, w, top_band_h)
            return frame_bgr[0:top_band_h, :], default_roi

        n = roi.normalized()
        x1 = max(0, n.x)
        y1 = max(0, n.y)
        x2 = min(frame_bgr.shape[1], n.x + n.w)
        y2 = min(frame_bgr.shape[0], n.y + n.h)
        if x2 <= x1 or y2 <= y1:
            return frame_bgr, None
        return frame_bgr[y1:y2, x1:x2], ROI(n.roi_id, x1, y1, x2 - x1, y2 - y1)

    def _try_extract(self, image_bgr: np.ndarray) -> OCRResult:
        if self._backend == "none":
            return OCRResult(None, None, False, None, "")

        preprocess_candidates = []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        preprocess_candidates.append(gray)
        preprocess_candidates.append(cv2.equalizeHist(gray))
        preprocess_candidates.append(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])

        best_text = ""
        best_conf = 0.0

        for candidate in preprocess_candidates:
            for rot in [0, 1, 2, 3]:
                c = np.rot90(candidate, k=rot)
                text, conf = self._ocr_once(c)
                if conf > best_conf:
                    best_conf = conf
                    best_text = text

        match = TS_REGEX.search(best_text)
        if not match:
            return OCRResult(None, best_conf if best_conf > 0 else None, False, None, best_text)
        return OCRResult(int(match.group(1)), best_conf, True, None, best_text)

    def _ocr_once(self, image_gray: np.ndarray) -> tuple[str, float]:
        if self._backend == "rapidocr" and self._engine is not None:
            result, _ = self._engine(image_gray)
            if not result:
                return "", 0.0
            text = " ".join([str(r[1]) for r in result])
            conf = float(np.mean([float(r[2]) for r in result])) if result else 0.0
            return text, conf

        if self._backend == "pytesseract" and pytesseract is not None:
            config = "--psm 7 -c tessedit_char_whitelist=0123456789"
            data = pytesseract.image_to_data(
                image_gray,
                output_type=pytesseract.Output.DICT,
                config=config,
            )
            if not data or "text" not in data:
                return "", 0.0

            texts = []
            confs = []
            for txt, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
                t = str(txt).strip()
                if not t:
                    continue
                texts.append(t)
                try:
                    c = float(conf)
                    if c >= 0:
                        confs.append(c / 100.0)
                except Exception:
                    continue

            if not texts:
                return "", 0.0
            merged = " ".join(texts)
            score = float(np.mean(confs)) if confs else 0.0
            return merged, score

        return "", 0.0

    def extract_timestamp(self, frame_bgr: np.ndarray) -> OCRResult:
        roi_pref = self._manual_roi or self._auto_roi
        crop, used_roi = self._crop_roi(frame_bgr, roi_pref)
        result = self._try_extract(crop)
        result.roi = used_roi

        if result.success:
            return result

        if self._manual_roi is None:
            auto = self.detect_auto_roi(frame_bgr)
            crop, used_roi = self._crop_roi(frame_bgr, auto)
            result = self._try_extract(crop)
            result.roi = used_roi
            return result

        return result
