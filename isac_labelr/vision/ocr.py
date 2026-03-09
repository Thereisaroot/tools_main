from __future__ import annotations

import os
import re
import sys
from collections import OrderedDict
import logging
from shutil import which
from dataclasses import dataclass

import cv2
import numpy as np

from isac_labelr.io.video_stream import rotate_bgr
from isac_labelr.models import ROI

try:
    from rapidocr_onnxruntime import RapidOCR as RapidOCROnnxRuntime
except Exception:  # pragma: no cover - optional runtime import
    RapidOCROnnxRuntime = None

try:
    from rapidocr import RapidOCR as RapidOCRUnified
except Exception:  # pragma: no cover - optional runtime import
    RapidOCRUnified = None

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
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)
        logging.getLogger("rapidocr").setLevel(logging.ERROR)

        self._manual_roi: ROI | None = None
        self._auto_roi: ROI | None = None
        self._neighbor_cap_path: str | None = None
        self._neighbor_cap: cv2.VideoCapture | None = None
        self._neighbor_next_index: int | None = None
        self._frame_ocr_cache: OrderedDict[tuple[str, int, int], OCRResult] = OrderedDict()
        self._frame_ocr_cache_max = 2048
        self._single_rotation_ocr = str(
            os.getenv("ISAC_OCR_SINGLE_ROTATION", "1")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._fast_full_rotation = str(
            os.getenv("ISAC_OCR_FAST_FULL_ROTATION", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._init_backend()

    def _init_backend(self) -> None:
        self._backend = "none"
        self._engine = None
        forced = os.getenv("ISAC_OCR_BACKEND", "auto").strip().lower()
        has_tesseract = pytesseract is not None and which("tesseract")
        default_prefer_tesseract = "1" if sys.platform == "darwin" else "0"
        prefer_tesseract = str(
            os.getenv("ISAC_OCR_PREFER_PYTESSERACT", default_prefer_tesseract)
        ).strip().lower() in {"1", "true", "yes", "on"}

        if forced in {"pytesseract", "tesseract"}:
            if has_tesseract:
                self._backend = "pytesseract"
            return

        if forced in {"rapidocr-onnxruntime", "rapidocr_onnxruntime", "rapidocr-ort"}:
            if RapidOCROnnxRuntime is not None:
                self._engine = RapidOCROnnxRuntime()
                self._backend = "rapidocr-onnxruntime"
            return

        if forced in {"rapidocr", "rapidocr-unified"}:
            if RapidOCRUnified is not None:
                self._engine = RapidOCRUnified()
                self._backend = "rapidocr"
            elif RapidOCROnnxRuntime is not None:
                self._engine = RapidOCROnnxRuntime()
                self._backend = "rapidocr-onnxruntime"
            return

        if prefer_tesseract and has_tesseract:
            self._backend = "pytesseract"
        elif RapidOCROnnxRuntime is not None:
            self._engine = RapidOCROnnxRuntime()
            self._backend = "rapidocr-onnxruntime"
        elif RapidOCRUnified is not None:
            self._engine = RapidOCRUnified()
            self._backend = "rapidocr"
        elif has_tesseract:
            self._backend = "pytesseract"

    def reset(self, *, preserve_rois: bool = True) -> None:
        manual = self._manual_roi if preserve_rois else None
        auto = self._auto_roi if preserve_rois else None
        self.close()
        self._frame_ocr_cache.clear()
        self._init_backend()
        self._manual_roi = manual
        self._auto_roi = auto

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

    def _cache_get(self, key: tuple[str, int, int]) -> OCRResult | None:
        value = self._frame_ocr_cache.get(key)
        if value is None:
            return None
        self._frame_ocr_cache.move_to_end(key)
        return value

    def _cache_put(self, key: tuple[str, int, int], value: OCRResult) -> None:
        self._frame_ocr_cache[key] = value
        self._frame_ocr_cache.move_to_end(key)
        while len(self._frame_ocr_cache) > self._frame_ocr_cache_max:
            self._frame_ocr_cache.popitem(last=False)

    def cache_video_frame_result(
        self,
        *,
        video_path: str,
        frame_index: int,
        rotation_deg: int,
        result: OCRResult,
    ) -> None:
        key = (str(video_path), int(frame_index), int(rotation_deg) % 360)
        self._cache_put(key, result)

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

    @staticmethod
    def _rotate_roi(roi: ROI, frame_w: int, frame_h: int, rotation_deg: int) -> tuple[ROI, tuple[int, int]]:
        n = roi.normalized()
        rot = rotation_deg % 360
        if rot == 0:
            return n, (frame_w, frame_h)
        if rot == 90:
            # clockwise
            new_x = frame_h - (n.y + n.h)
            new_y = n.x
            return ROI(n.roi_id, int(new_x), int(new_y), int(n.h), int(n.w)).normalized(), (
                frame_h,
                frame_w,
            )
        if rot == 180:
            new_x = frame_w - (n.x + n.w)
            new_y = frame_h - (n.y + n.h)
            return ROI(n.roi_id, int(new_x), int(new_y), int(n.w), int(n.h)).normalized(), (
                frame_w,
                frame_h,
            )
        if rot == 270:
            # counter-clockwise 90
            new_x = n.y
            new_y = frame_w - (n.x + n.w)
            return ROI(n.roi_id, int(new_x), int(new_y), int(n.h), int(n.w)).normalized(), (
                frame_h,
                frame_w,
            )
        return n, (frame_w, frame_h)

    @staticmethod
    def _orientation_score(roi: ROI, frame_w: int, frame_h: int) -> float:
        n = roi.normalized()
        cx = n.x + n.w / 2.0
        # Expect timestamp ROI: wide + near top + near horizontal center.
        aspect = n.w / max(1.0, float(n.h))
        width_ratio = n.w / max(1.0, float(frame_w))
        top_score = 1.0 - min(1.0, n.y / max(1.0, frame_h * 0.5))
        center_score = 1.0 - min(1.0, abs(cx - frame_w / 2.0) / max(1.0, frame_w / 2.0))
        return (aspect * 2.5) + (width_ratio * 1.5) + (top_score * 1.5) + (center_score * 1.0)

    def _canonicalize_roi_orientation(
        self, frame_bgr: np.ndarray, roi: ROI | None
    ) -> tuple[np.ndarray, ROI | None]:
        if roi is None:
            return frame_bgr, None

        h, w = frame_bgr.shape[:2]
        n = roi.normalized()

        # User rule: if ROI is tall and located on the left, rotate +90 first.
        cx = n.x + n.w / 2.0
        if n.h > n.w and cx < (w * 0.4):
            rotated_frame = rotate_bgr(frame_bgr, 90)
            rotated_roi, _ = self._rotate_roi(n, w, h, 90)
            return rotated_frame, rotated_roi

        best_rot = 0
        best_score = float("-inf")
        for rot in (0, 90, 180, 270):
            r_roi, (rw, rh) = self._rotate_roi(n, w, h, rot)
            score = self._orientation_score(r_roi, rw, rh)
            if score > best_score:
                best_score = score
                best_rot = rot

        if best_rot == 0:
            return frame_bgr, n
        rotated_frame = rotate_bgr(frame_bgr, best_rot)
        rotated_roi, _ = self._rotate_roi(n, w, h, best_rot)
        return rotated_frame, rotated_roi

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
        if self._single_rotation_ocr:
            rotations = [0]
        elif fast and not self._fast_full_rotation:
            rotations = [0, 2]
        else:
            rotations = [0, 2, 1, 3] if fast else [0, 1, 2, 3]

        for candidate in preprocess_candidates:
            for rot in rotations:
                c = np.rot90(candidate, k=rot)
                text, conf = self._ocr_once(c, fast=fast)
                if fast:
                    quick_ts = self._extract_unix_ms(text)
                    if self.is_valid_timestamp(quick_ts):
                        return OCRResult(quick_ts, conf, True, None, text)
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
        candidates = TimestampOCR.extract_timestamp_candidates(text)
        return candidates[0] if candidates else None

    @staticmethod
    def extract_timestamp_candidates(text: str) -> list[int]:
        source_quality: dict[int, int] = {}

        def add_candidate(token: str, *, quality: int) -> None:
            try:
                val = int(token)
            except ValueError:
                return
            prev = source_quality.get(val, -1)
            if quality > prev:
                source_quality[val] = quality

        # 1) direct contiguous 13-digit chunks from OCR text
        for match in TS_REGEX.finditer(text):
            token = match.group(1)
            add_candidate(token, quality=3)

        # 1.5) repair duplicated adjacent digit chunks inside a single OCR block
        # Example: "177252627575383" -> remove duplicated "75" -> "1772526275383"
        def _add_dedup_candidates(block: str) -> None:
            if len(block) <= 13:
                return
            max_chunk = min(4, len(block) // 2)
            for chunk_len in range(1, max_chunk + 1):
                end = len(block) - (2 * chunk_len) + 1
                for i in range(0, max(0, end)):
                    left = block[i : i + chunk_len]
                    right = block[i + chunk_len : i + (2 * chunk_len)]
                    if left != right:
                        continue
                    repaired = block[: i + chunk_len] + block[i + (2 * chunk_len) :]
                    if len(repaired) < 13:
                        continue
                    if len(repaired) == 13:
                        add_candidate(repaired, quality=4)
                        continue
                    for j in range(0, len(repaired) - 12):
                        add_candidate(repaired[j : j + 13], quality=4)

        for blk in re.findall(r"\d+", text):
            _add_dedup_candidates(blk)

        # 2) merge split digit blocks with suffix/prefix overlap
        # Example: "1772526275 75383" -> overlap "75" -> "1772526275383"
        digit_blocks = [blk for blk in re.findall(r"\d+", text) if blk]
        if len(digit_blocks) >= 2:
            merged = digit_blocks[0]
            for blk in digit_blocks[1:]:
                max_ov = min(4, len(merged), len(blk))
                ov = 0
                for k in range(max_ov, 0, -1):
                    if merged[-k:] == blk[:k]:
                        ov = k
                        break
                merged += blk[ov:]
            if len(merged) >= 13:
                for i in range(0, len(merged) - 12):
                    add_candidate(merged[i : i + 13], quality=2)

        # 3) OCR가 숫자를 띄워서 반환하는 경우를 복원 (fallback)
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 13:
            for i in range(0, len(digits) - 12):
                token = digits[i : i + 13]
                add_candidate(token, quality=1)

        if not source_quality:
            return []

        # Prefer realistic unix-ms range and expected prefix.
        def score(v: int) -> tuple[int, int]:
            txt = str(v)
            in_range = EPOCH_MIN_MS <= v <= EPOCH_MAX_MS
            has_prefix = len(txt) == EXPECTED_TS_DIGITS and txt.startswith(EXPECTED_TS_PREFIX)
            non_repetitive = (
                len(txt) == EXPECTED_TS_DIGITS
                and not TimestampOCR._looks_repetitive_timestamp_text(txt)
            )
            return (
                int(in_range),
                int(has_prefix),
                int(non_repetitive),
                int(source_quality.get(v, 0)),
            )

        ordered = sorted(source_quality.keys(), key=score, reverse=True)
        return ordered

    def _ocr_once(self, image_gray: np.ndarray, *, fast: bool = False) -> tuple[str, float]:
        if self._backend in {"rapidocr-onnxruntime", "rapidocr"} and self._engine is not None:
            output = self._engine(image_gray)
            if self._backend == "rapidocr-onnxruntime":
                result = output[0] if isinstance(output, tuple) else output
                if not result:
                    return "", 0.0
                text = " ".join([str(r[1]) for r in result])
                conf = float(np.mean([float(r[2]) for r in result])) if result else 0.0
                return text, conf

            txts = list(getattr(output, "txts", ()) or ())
            if not txts:
                return "", 0.0
            text = " ".join([str(t) for t in txts if str(t).strip()])
            scores = [float(s) for s in (getattr(output, "scores", ()) or ())]
            conf = float(np.mean(scores)) if scores else 0.0
            return text, conf

        if self._backend == "pytesseract" and pytesseract is not None:
            if fast:
                config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
                timeout_sec = 0.5
                try:
                    text = pytesseract.image_to_string(
                        image_gray,
                        config=config,
                        timeout=timeout_sec,
                    )
                except Exception:
                    return "", 0.0
                text = str(text).strip()
                # pytesseract fast path does not expose stable confidence cheaply.
                conf_score = min(1.0, sum(ch.isdigit() for ch in text) / 13.0) if text else 0.0
                return text, conf_score
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
    def _looks_repetitive_timestamp_text(text: str) -> bool:
        tail = text[len(EXPECTED_TS_PREFIX) :]
        if re.search(r"(\d)\1{4,}", tail):
            return True
        pair_tokens = [tail[i : i + 2] for i in range(0, len(tail) - 1, 2)]
        if len(pair_tokens) >= 4 and len(set(pair_tokens)) <= 2:
            return True
        return False

    @staticmethod
    def is_valid_timestamp(timestamp_ms: int | None) -> bool:
        if timestamp_ms is None:
            return False
        txt = str(int(timestamp_ms))
        if len(txt) != EXPECTED_TS_DIGITS or not txt.startswith(EXPECTED_TS_PREFIX):
            return False
        return not TimestampOCR._looks_repetitive_timestamp_text(txt)

    def _get_neighbor_cap(self, video_path: str) -> cv2.VideoCapture | None:
        norm_path = str(video_path)
        if self._neighbor_cap is not None and self._neighbor_cap_path == norm_path:
            return self._neighbor_cap
        if self._neighbor_cap is not None:
            self._neighbor_cap.release()
            self._neighbor_cap = None
            self._neighbor_cap_path = None
            self._neighbor_next_index = None

        cap = cv2.VideoCapture(norm_path)
        if not cap.isOpened():
            return None
        self._neighbor_cap = cap
        self._neighbor_cap_path = norm_path
        self._neighbor_next_index = None
        return cap

    def _read_rotated_frame_cached(self, video_path: str, frame_index: int, rotation_deg: int) -> np.ndarray | None:
        cap = self._get_neighbor_cap(video_path)
        if cap is None:
            return None
        if frame_index < 0:
            return None
        if self._neighbor_next_index != frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self._neighbor_next_index = frame_index
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        self._neighbor_next_index = frame_index + 1
        return rotate_bgr(frame, rotation_deg)

    def _extract_timestamp_for_video_frame(
        self,
        *,
        video_path: str,
        frame_index: int,
        rotation_deg: int,
        fast: bool,
    ) -> OCRResult:
        key = (str(video_path), int(frame_index), int(rotation_deg) % 360)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        frame = self._read_rotated_frame_cached(video_path, frame_index, rotation_deg)
        if frame is None:
            result = OCRResult(None, None, False, None, "")
            self._cache_put(key, result)
            return result

        result = self.extract_timestamp(frame, fast=fast)
        self._cache_put(key, result)
        return result

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

        prev_result = self._extract_timestamp_for_video_frame(
            video_path=video_path,
            frame_index=frame_index - 1,
            rotation_deg=rotation_deg,
            fast=fast,
        )
        next_result = self._extract_timestamp_for_video_frame(
            video_path=video_path,
            frame_index=frame_index + 1,
            rotation_deg=rotation_deg,
            fast=fast,
        )
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

    def close(self) -> None:
        if self._neighbor_cap is not None:
            self._neighbor_cap.release()
            self._neighbor_cap = None
            self._neighbor_cap_path = None
            self._neighbor_next_index = None
        self._frame_ocr_cache.clear()
        engine = self._engine
        self._engine = None
        self._backend = "none"
        if engine is not None:
            close_fn = getattr(engine, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            release_fn = getattr(engine, "release", None)
            if callable(release_fn):
                try:
                    release_fn()
                except Exception:
                    pass

    def extract_timestamp(self, frame_bgr: np.ndarray, *, fast: bool = False) -> OCRResult:
        candidates = self._build_candidate_rois(frame_bgr, fast=fast)

        success_results: list[OCRResult] = []
        best_fail: OCRResult | None = None

        for roi in candidates:
            canonical_frame, canonical_roi = self._canonicalize_roi_orientation(frame_bgr, roi)
            crop, used_roi = self._crop_roi(canonical_frame, canonical_roi)
            result = self._try_extract(crop, fast=fast)
            # Keep ROI in current UI coordinate system for display.
            result.roi = roi.normalized() if roi is not None else used_roi

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
