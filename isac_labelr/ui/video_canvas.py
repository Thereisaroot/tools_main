from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from isac_labelr.models import ROI, Track, Vector2


@dataclass(slots=True)
class _DrawState:
    drawing: bool = False
    start: tuple[int, int] | None = None
    end: tuple[int, int] | None = None


class VideoCanvas(QWidget):
    roi_created = Signal(object)
    roi_selected = Signal(str)
    timestamp_roi_created = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)

        self._frame_bgr: np.ndarray | None = None
        self._image_size: tuple[int, int] = (0, 0)
        self._target_rect = QRectF()

        self._rois: dict[str, ROI] = {}
        self._direction_vectors: dict[str, Vector2] = {}
        self._selected_roi_id: str | None = None

        self._timestamp_roi: ROI | None = None
        self._tracks: list[Track] = []
        self._active_roi_ids: set[str] = set()

        self._add_roi_mode = False
        self._manual_timestamp_mode = False
        self._show_overlays = True
        self._zoom = 1.0

        self._draw = _DrawState()

    def set_frame(self, frame_bgr: np.ndarray | None) -> None:
        self._frame_bgr = frame_bgr
        if frame_bgr is not None:
            self._image_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        self.update()

    def set_rois(self, rois: dict[str, ROI]) -> None:
        self._rois = dict(rois)
        self._active_roi_ids.intersection_update(self._rois.keys())
        self.update()

    def set_direction_vectors(self, vectors: dict[str, Vector2]) -> None:
        self._direction_vectors = dict(vectors)
        self.update()

    def set_selected_roi(self, roi_id: str | None) -> None:
        self._selected_roi_id = roi_id
        self.update()

    def set_timestamp_roi(self, roi: ROI | None) -> None:
        self._timestamp_roi = roi
        self.update()

    def set_tracks(self, tracks: list[Track]) -> None:
        self._tracks = tracks
        self.update()

    def set_active_roi_ids(self, roi_ids: set[str]) -> None:
        self._active_roi_ids = {str(roi_id) for roi_id in roi_ids}
        self.update()

    def set_show_overlays(self, show: bool) -> None:
        self._show_overlays = show
        self.update()

    def begin_add_roi(self) -> None:
        self._add_roi_mode = True
        self._manual_timestamp_mode = False

    def begin_manual_timestamp_roi(self) -> None:
        self._manual_timestamp_mode = True
        self._add_roi_mode = False

    def zoom_in(self) -> None:
        self._zoom = min(3.0, self._zoom + 0.1)
        self.update()

    def zoom_out(self) -> None:
        self._zoom = max(0.25, self._zoom - 0.1)
        self.update()

    def zoom_reset(self) -> None:
        self._zoom = 1.0
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(15, 15, 15))

        if self._frame_bgr is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a video to start")
            return

        image = cv2.cvtColor(self._frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        qimg = QImage(image.data, w, h, image.strides[0], QImage.Format_RGB888)

        self._target_rect = self._compute_target_rect(w, h)
        painter.drawImage(self._target_rect, qimg)

        if not self._show_overlays:
            return

        self._draw_rois(painter)
        self._draw_tracks(painter)
        self._draw_timestamp_roi(painter)
        self._draw_in_progress_rect(painter)

    def _compute_target_rect(self, img_w: int, img_h: int) -> QRectF:
        if img_w <= 0 or img_h <= 0:
            return QRectF()
        scale = min(self.width() / img_w, self.height() / img_h) * self._zoom
        draw_w = img_w * scale
        draw_h = img_h * scale
        x = (self.width() - draw_w) / 2.0
        y = (self.height() - draw_h) / 2.0
        return QRectF(x, y, draw_w, draw_h)

    def _frame_to_widget(self, x: float, y: float) -> QPointF:
        iw, ih = self._image_size
        if iw <= 0 or ih <= 0 or self._target_rect.width() <= 0:
            return QPointF(0, 0)
        px = self._target_rect.x() + (x / iw) * self._target_rect.width()
        py = self._target_rect.y() + (y / ih) * self._target_rect.height()
        return QPointF(px, py)

    def _widget_to_frame(self, point: QPointF) -> tuple[int, int] | None:
        iw, ih = self._image_size
        if iw <= 0 or ih <= 0:
            return None
        if not self._target_rect.contains(point):
            return None

        x = (point.x() - self._target_rect.x()) / max(1.0, self._target_rect.width())
        y = (point.y() - self._target_rect.y()) / max(1.0, self._target_rect.height())
        fx = int(np.clip(x * iw, 0, iw - 1))
        fy = int(np.clip(y * ih, 0, ih - 1))
        return fx, fy

    def _draw_rois(self, painter: QPainter) -> None:
        for roi_id, roi in self._rois.items():
            n = roi.normalized()
            p1 = self._frame_to_widget(n.x, n.y)
            p2 = self._frame_to_widget(n.x + n.w, n.y + n.h)
            rect = QRectF(p1, p2)

            selected = roi_id == self._selected_roi_id
            active = roi_id in self._active_roi_ids
            if active:
                painter.fillRect(rect, QColor(255, 80, 80, 48))

            if selected and active:
                color = QColor(255, 220, 90)
            elif selected:
                color = QColor(255, 165, 0)
            elif active:
                color = QColor(255, 80, 80)
            else:
                color = QColor(70, 220, 90)

            pen = QPen(color, 3 if active else 2)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.drawText(rect.topLeft() + QPointF(4, 14), roi_id)

            direction = self._direction_vectors.get(roi_id)
            if direction is not None:
                center = self._frame_to_widget(n.x + n.w / 2, n.y + n.h / 2)
                mag = np.hypot(direction.dx, direction.dy)
                if mag > 0:
                    dx = (direction.dx / mag) * 30
                    dy = (direction.dy / mag) * 30
                    end = QPointF(center.x() + dx, center.y() + dy)
                    painter.drawLine(center, end)

    def _draw_tracks(self, painter: QPainter) -> None:
        pen = QPen(QColor(0, 180, 255), 2)
        painter.setPen(pen)
        for track in self._tracks:
            x1, y1, x2, y2 = track.bbox
            p1 = self._frame_to_widget(x1, y1)
            p2 = self._frame_to_widget(x2, y2)
            rect = QRectF(p1, p2)
            painter.drawRect(rect)
            painter.drawText(rect.topLeft() + QPointF(3, 13), f"ID {track.track_id}")

    def _draw_timestamp_roi(self, painter: QPainter) -> None:
        if self._timestamp_roi is None:
            return
        n = self._timestamp_roi.normalized()
        p1 = self._frame_to_widget(n.x, n.y)
        p2 = self._frame_to_widget(n.x + n.w, n.y + n.h)
        rect = QRectF(p1, p2)
        pen = QPen(QColor(240, 230, 70), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.drawText(rect.bottomLeft() + QPointF(2, -2), "TS ROI")

    def _draw_in_progress_rect(self, painter: QPainter) -> None:
        if not self._draw.drawing or self._draw.start is None or self._draw.end is None:
            return
        sx, sy = self._draw.start
        ex, ey = self._draw.end
        p1 = self._frame_to_widget(sx, sy)
        p2 = self._frame_to_widget(ex, ey)
        pen = QPen(QColor(255, 255, 255), 1, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(QRectF(p1, p2))

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return

        fpt = self._widget_to_frame(event.position())
        if fpt is None:
            return

        if self._add_roi_mode or self._manual_timestamp_mode:
            self._draw.drawing = True
            self._draw.start = fpt
            self._draw.end = fpt
            self.update()
            return

        selected = None
        for roi_id, roi in self._rois.items():
            if roi.normalized().contains(*fpt):
                selected = roi_id
                break

        self._selected_roi_id = selected
        if selected is not None:
            self.roi_selected.emit(selected)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self._draw.drawing or self._draw.start is None:
            return
        fpt = self._widget_to_frame(event.position())
        if fpt is None:
            return
        self._draw.end = fpt
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        if not self._draw.drawing or self._draw.start is None or self._draw.end is None:
            return

        sx, sy = self._draw.start
        ex, ey = self._draw.end
        roi = ROI(roi_id="", x=sx, y=sy, w=ex - sx, h=ey - sy).normalized()

        self._draw = _DrawState()

        if roi.w < 8 or roi.h < 8:
            self.update()
            return

        if self._manual_timestamp_mode:
            roi.roi_id = "timestamp_manual"
            self.timestamp_roi_created.emit(roi)
            self._manual_timestamp_mode = False
        elif self._add_roi_mode:
            self.roi_created.emit(roi)
            self._add_roi_mode = False

        self.update()
