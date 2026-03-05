from __future__ import annotations

from pathlib import Path
from threading import Event as ThreadEvent

from PySide6.QtCore import QObject, Signal, Slot

from isac_labelr.analysis.engine import AnalysisEngine
from isac_labelr.io.metadata_writer import MetadataWriter
from isac_labelr.logger import build_run_logger
from isac_labelr.models import AnalysisProgress, AnalysisRequest, AnalysisResult, EventRecord, SessionConfig


class AnalysisWorker(QObject):
    progress = Signal(object)
    event = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, request: AnalysisRequest) -> None:
        super().__init__()
        self.request = request
        self._stop_event = ThreadEvent()
        self._ui_events_emitted = 0
        self._max_ui_events = 5000
        self._progress_emit_stride = 5
        self._last_progress_frame = -1

    @Slot()
    def run(self) -> None:
        output_dir = MetadataWriter.default_output_dir(self.request.video_path)
        writer = MetadataWriter(output_dir)
        logger = build_run_logger(output_dir / "run_log.txt")

        session = SessionConfig(
            video_path=self.request.video_path,
            rotation_deg=self.request.rotation_deg,
            mode=self.request.mode,
            rois=self.request.rois,
            direction_vectors=self.request.direction_vectors,
            analyze_full=self.request.analyze_full,
            start_ms=self.request.start_ms,
            duration_ms=self.request.duration_ms,
            timestamp_correction_ms=self.request.timestamp_correction_ms,
            ocr_manual_roi=self.request.ocr_manual_roi,
        )
        writer.write_session_config(session.to_dict())

        engine = AnalysisEngine()
        try:
            result = engine.run(
                self.request,
                stop_event=self._stop_event,
                on_progress=self._emit_progress,
                on_event=lambda e: self._handle_event(e, writer),
                logger=logger,
            )
            result.output_dir = Path(output_dir)
            writer.flush()
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            writer.close()

    def stop(self) -> None:
        self._stop_event.set()

    def _emit_progress(self, progress: AnalysisProgress) -> None:
        if progress.frame_index - self._last_progress_frame >= self._progress_emit_stride:
            self.progress.emit(progress)
            self._last_progress_frame = progress.frame_index

    def _handle_event(self, event: EventRecord, writer: MetadataWriter) -> None:
        writer.write_event(event)
        if self._ui_events_emitted < self._max_ui_events:
            self.event.emit(event)
            self._ui_events_emitted += 1
