from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2

from isac_labelr.models import FramePacket


def _env_truthy(value: str | None) -> bool | None:
    if value is None:
        return None
    val = value.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return None


def _should_use_pyav() -> bool:
    # Optional override:
    # - ISAC_USE_PYAV=1  -> force enable
    # - ISAC_USE_PYAV=0  -> force disable
    env = _env_truthy(os.getenv("ISAC_USE_PYAV"))
    if env is not None:
        return env

    # macOS에서는 cv2/av가 FFmpeg dylib를 중복 로드해 충돌 경고가 발생할 수 있어 기본 비활성.
    if platform.system() == "Darwin":
        return False
    return True


def _load_av():
    try:
        import av as av_mod
    except Exception:  # pragma: no cover - optional runtime import
        return None
    return av_mod


@dataclass(slots=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_ms: int


def rotate_bgr(image, rotation_deg: int):
    rotation_deg = rotation_deg % 360
    if rotation_deg == 0:
        return image
    if rotation_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotation_deg}")


class VideoStream:
    def __init__(self, video_path: str) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        self._av = _load_av() if _should_use_pyav() else None
        self._use_pyav = self._av is not None
        self._info = self._probe()

    @property
    def info(self) -> VideoInfo:
        return self._info

    def _probe(self) -> VideoInfo:
        if not self._use_pyav:
            return self._probe_with_cv2()

        with self._av.open(str(self.video_path)) as container:  # type: ignore[union-attr]
            stream = container.streams.video[0]
            width = int(stream.width or 0)
            height = int(stream.height or 0)

            fps = 30.0
            if stream.average_rate is not None:
                fps = float(stream.average_rate)
            elif stream.base_rate is not None:
                fps = float(stream.base_rate)

            duration_ms = 0
            if stream.duration is not None and stream.time_base is not None:
                duration_ms = int(float(stream.duration * stream.time_base) * 1000)
            elif container.duration is not None:
                duration_ms = int(container.duration / 1000)

            total_frames = int(stream.frames or 0)
            if total_frames <= 0 and duration_ms > 0:
                total_frames = max(1, int((duration_ms / 1000.0) * fps))

            return VideoInfo(
                path=self.video_path,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_ms=duration_ms,
            )

    def _probe_with_cv2(self) -> VideoInfo:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_ms = int((total_frames / max(1.0, fps)) * 1000)
        cap.release()

        return VideoInfo(
            path=self.video_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_ms=duration_ms,
        )

    def iter_frames(
        self,
        *,
        rotation_deg: int,
        start_ms: int = 0,
        duration_ms: int | None = None,
        chunk_seconds: int = 60,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> Iterator[FramePacket]:
        if not self._use_pyav:
            yield from self._iter_frames_cv2(
                rotation_deg=rotation_deg,
                start_ms=start_ms,
                duration_ms=duration_ms,
                chunk_seconds=chunk_seconds,
                on_chunk=on_chunk,
            )
            return

        end_ms = (start_ms + duration_ms) if duration_ms is not None else None
        chunk_ms = max(1, chunk_seconds) * 1000
        last_chunk_tick = start_ms
        chunk_idx = 0

        with self._av.open(str(self.video_path)) as container:  # type: ignore[union-attr]
            stream = container.streams.video[0]

            if start_ms > 0:
                container.seek(start_ms * 1000, backward=True, any_frame=False)

            for frame in container.decode(video=stream.index):
                if frame.time is not None:
                    time_ms = int(frame.time * 1000)
                elif frame.pts is not None and frame.time_base is not None:
                    time_ms = int(float(frame.pts * frame.time_base) * 1000)
                else:
                    # fallback based on decode order
                    index = getattr(frame, "index", 0) or 0
                    time_ms = int(index / self._info.fps * 1000)

                if time_ms < start_ms:
                    continue
                if end_ms is not None and time_ms > end_ms:
                    break

                image = frame.to_ndarray(format="bgr24")
                image = rotate_bgr(image, rotation_deg)

                frame_index = int((time_ms / 1000.0) * self._info.fps)
                yield FramePacket(frame_index=frame_index, time_ms=time_ms, image_bgr=image)

                if on_chunk and (time_ms - last_chunk_tick) >= chunk_ms:
                    chunk_idx += 1
                    last_chunk_tick = time_ms
                    on_chunk(chunk_idx, time_ms)

    def _iter_frames_cv2(
        self,
        *,
        rotation_deg: int,
        start_ms: int,
        duration_ms: int | None,
        chunk_seconds: int,
        on_chunk: Callable[[int, int], None] | None,
    ) -> Iterator[FramePacket]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        end_ms = (start_ms + duration_ms) if duration_ms is not None else None
        chunk_ms = max(1, chunk_seconds) * 1000
        last_chunk_tick = start_ms
        chunk_idx = 0

        if start_ms > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(start_ms))

        fps = max(1.0, self._info.fps)

        try:
            while True:
                ok, image = cap.read()
                if not ok or image is None:
                    break

                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                time_ms = int(pos_msec) if pos_msec > 0 else int((frame_no / fps) * 1000)

                if time_ms < start_ms:
                    continue
                if end_ms is not None and time_ms > end_ms:
                    break

                rotated = rotate_bgr(image, rotation_deg)
                frame_index = int((time_ms / 1000.0) * fps)
                yield FramePacket(frame_index=frame_index, time_ms=time_ms, image_bgr=rotated)

                if on_chunk and (time_ms - last_chunk_tick) >= chunk_ms:
                    chunk_idx += 1
                    last_chunk_tick = time_ms
                    on_chunk(chunk_idx, time_ms)
        finally:
            cap.release()
