from __future__ import annotations

import re
from shutil import which
from dataclasses import dataclass

import cv2
import numpy as np

from isac_labelr.io.video_stream import rotate_bgr
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
EPOCH_MIN_MS = 946684800000      # 2000-01-01
EPOCH_MAX_MS = 4102444800000     # 2100-01-01
EXPECTED_TS_PREFIX = "177"
EXPECTED_TS_DIGITS = 13


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

    @property
    def auto_roi(self) -> ROI | None:
        return self._auto_roi

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

    @staticmethod
    def _expand_roi(roi: ROI, frame_shape: tuple[int, int, int], ratio: float = 0.15) -> ROI:
        h, w = frame_shape[:2]
        n = roi.normalized()
        pad_x = int(n.w * ratio)
        pad_y = int(n.h * ratio)
        x1 = max(0, n.x - pad_x)
        y1 = max(0, n.y - pad_y)
        x2 = min(w, n.x + n.w + pad_x)
        y2 = min(h, n.y + n.h + pad_y)
        return ROI(n.roi_id + "_expanded", x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    def _build_candidate_rois(self, frame_bgr: np.ndarray, *, fast: bool = False) -> list[ROI]:
        h, w = frame_bgr.shape[:2]
        band_h = max(40, h // 8)

        candidates: list[ROI] = []
        if self._manual_roi is not None:
            manual = self._manual_roi.normalized()
            candidates.append(manual)
            # Manual ROI 지정 시에는 주변만 추가 검사해 속도를 보장한다.
            if not fast:
                candidates.append(self._expand_roi(manual, frame_bgr.shape))
        else:
            if self._auto_roi is not None:
                candidates.append(self._auto_roi.normalized())

            if not fast:
                detected = self.detect_auto_roi(frame_bgr)
                if detected is not None:
                    candidates.append(detected.normalized())

            # Common overlay positions
            candidates.append(ROI("timestamp_top", 0, 0, w, band_h))
            if not fast:
                candidates.append(ROI("timestamp_bottom", 0, h - band_h, w, band_h))

        deduped: list[ROI] = []
        seen: set[tuple[int, int, int, int]] = set()
        for roi in candidates:
            n = roi.normalized()
            key = (n.x, n.y, n.w, n.h)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(n)
        return deduped

    def _try_extract(self, image_bgr: np.ndarray, *, fast: bool = False) -> OCRResult:
        if self._backend == "none":
            return OCRResult(None, None, False, None, "")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if fast:
            preprocess_candidates = [gray]
            if min(gray.shape[:2]) < 200:
                preprocess_candidates.append(
                    cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                )
        else:
            preprocess_candidates = [
                gray,
                cv2.equalizeHist(gray),
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            ]
            if min(gray.shape[:2]) < 200:
                preprocess_candidates.append(
                    cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                )

        best_text = ""
        best_conf = 0.0
        best_score = -1.0

        for candidate in preprocess_candidates:
            for rot in [0, 1, 2, 3]:
                c = np.rot90(candidate, k=rot)
                text, conf = self._ocr_once(c, fast=fast)
                score = sum(ch.isdigit() for ch in text) + conf
                if score > best_score:
                    best_score = score
                    best_conf = conf
                    best_text = text

        ts = self._extract_unix_ms(best_text)
        if ts is None:
            return OCRResult(None, best_conf if best_conf > 0 else None, False, None, best_text)
        return OCRResult(ts, best_conf, True, None, best_text)

    @staticmethod
    def _extract_unix_ms(text: str) -> int | None:
        # 1) direct 13-digit match
        match = TS_REGEX.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

        # 2) OCR가 숫자를 띄워서 반환하는 경우를 복원
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) < 13:
            return None

        candidates: list[int] = []
        for i in range(0, len(digits) - 12):
            chunk = digits[i : i + 13]
            try:
                val = int(chunk)
            except ValueError:
                continue
            candidates.append(val)

        if not candidates:
            return None

        plausible = [v for v in candidates if EPOCH_MIN_MS <= v <= EPOCH_MAX_MS]
        if plausible:
            return plausible[0]
        return candidates[0]

    def _ocr_once(self, image_gray: np.ndarray, *, fast: bool = False) -> tuple[str, float]:
        if self._backend == "rapidocr" and self._engine is not None:
            result, _ = self._engine(image_gray)
            if not result:
                return "", 0.0
            text = " ".join([str(r[1]) for r in result])
            conf = float(np.mean([float(r[2]) for r in result])) if result else 0.0
            return text, conf

        if self._backend == "pytesseract" and pytesseract is not None:
            if fast:
                configs = ["--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"]
                timeout_sec = 0.6
            else:
                configs = [
                    "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
                    "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789",
                ]
                timeout_sec = 1.5
            best_txt = ""
            best_conf = 0.0
            best_score = -1.0

            for config in configs:
                try:
                    data = pytesseract.image_to_data(
                        image_gray,
                        output_type=pytesseract.Output.DICT,
                        config=config,
                        timeout=timeout_sec,
                    )
                except Exception:
                    continue

                if not data or "text" not in data:
                    continue

                texts = []
                confs = []
                for txt, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
                    t = str(txt).strip()
                    if t:
                        texts.append(t)
                    try:
                        c = float(conf)
                        if c >= 0:
                            confs.append(c / 100.0)
                    except Exception:
                        continue

                text = " ".join(texts)
                conf_score = float(np.mean(confs)) if confs else 0.0
                digits_len = sum(ch.isdigit() for ch in text)
                score = digits_len + conf_score

                if score > best_score:
                    best_score = score
                    best_conf = conf_score
                    best_txt = text

            if not best_txt.strip():
                return "", 0.0
            return best_txt, best_conf

        return "", 0.0

    @staticmethod
    def is_valid_timestamp(timestamp_ms: int | None) -> bool:
        if timestamp_ms is None:
            return False
        txt = str(int(timestamp_ms))
        return len(txt) == EXPECTED_TS_DIGITS and txt.startswith(EXPECTED_TS_PREFIX)

    @staticmethod
    def _read_rotated_frame(video_path: str, frame_index: int, rotation_deg: int) -> np.ndarray | None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        try:
            if frame_index < 0:
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            return rotate_bgr(frame, rotation_deg)
        finally:
            cap.release()

    def interpolate_timestamp_from_neighbors(
        self,
        *,
        video_path: str,
        frame_index: int,
        rotation_deg: int,
        fast: bool = True,
    ) -> tuple[int | None, float | None]:
        if frame_index <= 0:
            return None, None

        prev_frame = self._read_rotated_frame(video_path, frame_index - 1, rotation_deg)
        next_frame = self._read_rotated_frame(video_path, frame_index + 1, rotation_deg)
        if prev_frame is None or next_frame is None:
            return None, None

        prev_result = self.extract_timestamp(prev_frame, fast=fast)
        next_result = self.extract_timestamp(next_frame, fast=fast)
        prev_ts = prev_result.timestamp_ms if self.is_valid_timestamp(prev_result.timestamp_ms) else None
        next_ts = next_result.timestamp_ms if self.is_valid_timestamp(next_result.timestamp_ms) else None
        if prev_ts is None or next_ts is None:
            return None, None

        interpolated = int(round((prev_ts + next_ts) / 2.0))
        conf_values = [
            c
            for c in [prev_result.confidence, next_result.confidence]
            if c is not None
        ]
        conf = float(np.mean(conf_values)) if conf_values else None
        return interpolated, conf

    def extract_timestamp(self, frame_bgr: np.ndarray, *, fast: bool = False) -> OCRResult:
        candidates = self._build_candidate_rois(frame_bgr, fast=fast)

        success_results: list[OCRResult] = []
        best_fail: OCRResult | None = None

        for roi in candidates:
            crop, used_roi = self._crop_roi(frame_bgr, roi)
            result = self._try_extract(crop, fast=fast)
            result.roi = used_roi

            if result.success:
                success_results.append(result)
                continue

            if best_fail is None:
                best_fail = result
            else:
                curr = result.confidence if result.confidence is not None else -1.0
                prev = best_fail.confidence if best_fail.confidence is not None else -1.0
                if curr > prev:
                    best_fail = result

        if success_results:
            success_results.sort(
                key=lambda r: (r.confidence if r.confidence is not None else -1.0),
                reverse=True,
            )
            return success_results[0]

        if best_fail is not None:
            return best_fail
        return OCRResult(None, None, False, None, "")
