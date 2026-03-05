from __future__ import annotations

import json
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from isac_labelr.analysis.engine import AnalysisEngine, PresenceState
from isac_labelr.analysis.worker import AnalysisWorker
from isac_labelr.commands import AppCommand, MenuActionSpec, build_menu_action_map
from isac_labelr.io.metadata_writer import MetadataWriter
from isac_labelr.io.video_stream import rotate_bgr
from isac_labelr.models import (
    AnalysisMode,
    AnalysisProgress,
    AnalysisRequest,
    AnalysisResult,
    AppPreferences,
    EventRecord,
    OverlayTSStatus,
    ROI,
    SessionConfig,
    Vector2,
)
from isac_labelr.settings import (
    load_preferences,
    load_recent_videos,
    push_recent_video,
    save_preferences,
)
from isac_labelr.ui.video_canvas import VideoCanvas
from isac_labelr.vision.detector import (
    DetectorConfig,
    DetectorInitializationError,
    OnnxYoloPersonDetector,
)
from isac_labelr.vision.ocr import TimestampOCR
from isac_labelr.vision.tracker import ByteTrackLite


class PreferencesDialog(QDialog):
    def __init__(self, pref: AppPreferences, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self._pref = pref

        self.model_path = QLineEdit(pref.detector_model_path)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_model)

        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.model_path, 1)
        model_layout.addWidget(browse_btn)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(pref.detector_confidence)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(pref.detector_iou)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(10, 3600)
        self.chunk_spin.setValue(pref.chunk_seconds)

        form = QFormLayout()
        form.addRow("Detector model", model_row)
        form.addRow("Confidence", self.conf_spin)
        form.addRow("IoU threshold", self.iou_spin)
        form.addRow("Chunk seconds", self.chunk_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select ONNX model", "", "ONNX (*.onnx)")
        if path:
            self.model_path.setText(path)

    def value(self) -> AppPreferences:
        return AppPreferences(
            detector_model_path=self.model_path.text().strip() or "models/person_detector.onnx",
            detector_confidence=float(self.conf_spin.value()),
            detector_iou=float(self.iou_spin.value()),
            chunk_seconds=int(self.chunk_spin.value()),
        )


class OCRDebugWorker(QObject):
    finished = Signal(object, str)
    failed = Signal(str)

    def __init__(self, frame_bgr, manual_roi: ROI | None) -> None:
        super().__init__()
        self.frame_bgr = frame_bgr
        self.manual_roi = manual_roi

    @Slot()
    def run(self) -> None:
        try:
            ocr = TimestampOCR()
            if self.manual_roi is not None:
                ocr.set_manual_roi(self.manual_roi)
            # Debug button should return quickly; use fast OCR path.
            result = ocr.extract_timestamp(self.frame_bgr, fast=True)
            self.finished.emit(result, ocr.backend)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ISAC Labelr")
        self.resize(1400, 900)

        self.preferences = load_preferences()

        self.video_path: str | None = None
        self.capture: cv2.VideoCapture | None = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame_index = 0
        self.current_video_time_ms = 0
        self.current_frame_raw = None
        self.current_frame_rotated = None

        self.rotation_deg = 0
        self.rois: dict[str, ROI] = {}
        self.direction_vectors: dict[str, Vector2] = {}
        self._next_roi_num = 1
        self.ocr_manual_roi: ROI | None = None

        self.events: list[EventRecord] = []
        self.output_dir: Path | None = None

        self.analysis_thread: QThread | None = None
        self.analysis_worker: AnalysisWorker | None = None
        self.ocr_debug_thread: QThread | None = None
        self.ocr_debug_worker: OCRDebugWorker | None = None
        self.preview_detector: OnnxYoloPersonDetector | None = None
        self.preview_tracker: ByteTrackLite | None = None
        self.preview_event_engine = AnalysisEngine()
        self.preview_presence: dict[tuple[str, int], PresenceState] = {}
        self.preview_last_overlay_ts: int | None = None

        self.timestamp_ocr = TimestampOCR()

        self.menu_actions: dict[AppCommand, QAction] = {}
        self.command_buttons: dict[AppCommand, QPushButton] = {}

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)

        self._build_ui()
        self._build_menubar()
        self._bind_signals()
        self._refresh_recent_menu()
        self.status_label.setText(f"Idle | OCR backend: {self.timestamp_ocr.backend}")
        self._set_notice(self._default_notice_text())
        self._refresh_action_states()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        self.splitter = splitter

        left = QWidget()
        left_layout = QVBoxLayout(left)

        quick_actions = QWidget()
        quick_actions_layout = QHBoxLayout(quick_actions)
        quick_actions_layout.setContentsMargins(0, 0, 0, 0)

        self.open_file_button = QPushButton("Open File")
        self.add_roi_button = QPushButton("Add ROI")
        self.add_ts_roi_button = QPushButton("Add TS ROI")
        self.rotate_left_button = QPushButton("Rotate -90")
        self.rotate_right_button = QPushButton("Rotate +90")
        self.start_full_button = QPushButton("Start Full Analysis")
        self.start_full_button.setToolTip("Always analyze full video, ignoring start/duration.")
        self.start_partial_button = QPushButton("Start Partial Analysis")
        self.start_partial_button.setToolTip("Analyze only Start ms ~ Duration ms range.")
        self.stop_analysis_button = QPushButton("Stop Analysis")

        self.command_buttons = {
            AppCommand.OPEN_VIDEO: self.open_file_button,
            AppCommand.ADD_ROI: self.add_roi_button,
            AppCommand.MANUAL_OCR_ROI: self.add_ts_roi_button,
            AppCommand.ROTATE_PREV: self.rotate_left_button,
            AppCommand.ROTATE_NEXT: self.rotate_right_button,
            AppCommand.RUN_FULL: self.start_full_button,
            AppCommand.RUN_PARTIAL: self.start_partial_button,
            AppCommand.STOP_ANALYSIS: self.stop_analysis_button,
        }

        for button in [
            self.open_file_button,
            self.add_roi_button,
            self.add_ts_roi_button,
            self.rotate_left_button,
            self.rotate_right_button,
            self.start_full_button,
            self.start_partial_button,
            self.stop_analysis_button,
        ]:
            quick_actions_layout.addWidget(button)
        quick_actions_layout.addStretch(1)

        left_layout.addWidget(quick_actions)

        self.canvas = VideoCanvas()
        left_layout.addWidget(self.canvas, 1)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self.play_button = QPushButton("Play Preview")
        self.play_button.setToolTip("Preview playback only. Analysis starts with Start Full/Partial Analysis buttons.")
        self.prev_button = QPushButton("Prev")
        self.next_button = QPushButton("Next")
        self.preview_detection_checkbox = QCheckBox("Detection On")
        self.preview_detection_checkbox.setToolTip(
            "When enabled, Play Preview runs person detection/tracking and draws person IDs."
        )
        self.frame_slider = QSpinBox()
        self.frame_slider.setRange(0, 0)
        self.frame_label = QLabel("frame: 0 / 0")
        self.preview_hint_label = QLabel("Play = preview only (analysis not started)")
        self.preview_hint_label.setStyleSheet("color: #999;")

        controls_layout.addWidget(self.prev_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.preview_detection_checkbox)
        controls_layout.addWidget(QLabel("Frame"))
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(self.preview_hint_label)
        left_layout.addWidget(controls)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        settings_box = QGroupBox("Analysis Settings")
        settings_form = QFormLayout(settings_box)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Mode A (Entry)", AnalysisMode.ENTRY_ONLY)
        self.mode_combo.addItem("Mode B (Entry+Exit)", AnalysisMode.ENTRY_EXIT_DIRECTION)

        self.full_checkbox = QCheckBox("Analyze full video")
        self.full_checkbox.setChecked(True)

        self.start_ms_spin = QSpinBox()
        self.start_ms_spin.setRange(0, 2_000_000_000)

        self.duration_ms_spin = QSpinBox()
        self.duration_ms_spin.setRange(1, 2_000_000_000)
        self.duration_ms_spin.setValue(60_000)

        self.correction_edit = QLineEdit("0")
        self.correction_edit.setValidator(QIntValidator(-2_000_000_000, 2_000_000_000, self))

        settings_form.addRow("Mode", self.mode_combo)
        settings_form.addRow("Scope", self.full_checkbox)
        settings_form.addRow("Start ms", self.start_ms_spin)
        settings_form.addRow("Duration ms", self.duration_ms_spin)
        settings_form.addRow("TS correction ms", self.correction_edit)

        right_layout.addWidget(settings_box)

        notice_box = QGroupBox("TEXT NOTICE")
        notice_layout = QVBoxLayout(notice_box)
        self.notice_text = QTextEdit()
        self.notice_text.setReadOnly(True)
        self.notice_text.setFixedHeight(36)
        notice_layout.addWidget(self.notice_text)
        right_layout.addWidget(notice_box)

        roi_box = QGroupBox("ROIs")
        roi_layout = QVBoxLayout(roi_box)
        self.roi_list = QListWidget()

        dir_widget = QWidget()
        dir_layout = QHBoxLayout(dir_widget)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(-1000, 1000)
        self.dx_spin.setSingleStep(0.1)
        self.dx_spin.setValue(1.0)
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(-1000, 1000)
        self.dy_spin.setSingleStep(0.1)
        self.dy_spin.setValue(0.0)
        dir_layout.addWidget(QLabel("dx"))
        dir_layout.addWidget(self.dx_spin)
        dir_layout.addWidget(QLabel("dy"))
        dir_layout.addWidget(self.dy_spin)

        roi_layout.addWidget(self.roi_list)
        roi_layout.addWidget(dir_widget)
        right_layout.addWidget(roi_box, 1)

        event_box = QGroupBox("Detected Events")
        event_layout = QVBoxLayout(event_box)
        self.event_list = QListWidget()
        event_layout.addWidget(self.event_list)
        right_layout.addWidget(event_box, 2)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([900, 500])

        self.setCentralWidget(splitter)

        self.status_label = QLabel("Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress_bar)

        self._toggle_partial_fields()

    def _build_menubar(self) -> None:
        action_map = build_menu_action_map()

        for menu_name, specs in action_map.items():
            menu = self.menuBar().addMenu(menu_name)

            if menu_name == "File":
                open_recent = QMenu("Open Recent", self)
                self.recent_menu = open_recent

                for spec in specs:
                    action = self._create_action(spec)
                    menu.addAction(action)
                    if spec.command == AppCommand.OPEN_VIDEO:
                        menu.addMenu(open_recent)
                        menu.addSeparator()
                continue

            for spec in specs:
                action = self._create_action(spec)
                menu.addAction(action)

        self.menu_actions[AppCommand.TOGGLE_OVERLAYS].setChecked(True)

        self.menu_actions[AppCommand.SET_MODE_A].setCheckable(True)
        self.menu_actions[AppCommand.SET_MODE_B].setCheckable(True)
        self.menu_actions[AppCommand.SET_MODE_A].setChecked(True)

    def _create_action(self, spec: MenuActionSpec) -> QAction:
        action = QAction(spec.text, self)
        if spec.shortcut:
            action.setShortcut(spec.shortcut)
        action.setCheckable(spec.checkable)
        action.triggered.connect(lambda _checked=False, c=spec.command: self._dispatch_command(c))
        self.menu_actions[spec.command] = action
        return action

    def _bind_signals(self) -> None:
        self.open_file_button.clicked.connect(lambda: self._dispatch_command(AppCommand.OPEN_VIDEO))
        self.add_roi_button.clicked.connect(lambda: self._dispatch_command(AppCommand.ADD_ROI))
        self.add_ts_roi_button.clicked.connect(lambda: self._dispatch_command(AppCommand.MANUAL_OCR_ROI))
        self.rotate_left_button.clicked.connect(lambda: self._dispatch_command(AppCommand.ROTATE_PREV))
        self.rotate_right_button.clicked.connect(lambda: self._dispatch_command(AppCommand.ROTATE_NEXT))
        self.start_full_button.clicked.connect(lambda: self._dispatch_command(AppCommand.RUN_FULL))
        self.start_partial_button.clicked.connect(lambda: self._dispatch_command(AppCommand.RUN_PARTIAL))
        self.stop_analysis_button.clicked.connect(lambda: self._dispatch_command(AppCommand.STOP_ANALYSIS))

        self.play_button.clicked.connect(self._toggle_play)
        self.prev_button.clicked.connect(lambda: self._seek_frame(self.current_frame_index - 1))
        self.next_button.clicked.connect(lambda: self._seek_frame(self.current_frame_index + 1))
        self.preview_detection_checkbox.toggled.connect(self._on_preview_detection_toggled)
        self.frame_slider.valueChanged.connect(self._seek_frame)

        self.full_checkbox.stateChanged.connect(self._toggle_partial_fields)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.roi_list.currentItemChanged.connect(self._on_roi_list_selection_changed)
        self.dx_spin.valueChanged.connect(self._on_direction_changed)
        self.dy_spin.valueChanged.connect(self._on_direction_changed)
        self.event_list.itemClicked.connect(self._on_event_item_clicked)

        self.canvas.roi_created.connect(self._on_canvas_roi_created)
        self.canvas.roi_selected.connect(self._on_canvas_roi_selected)
        self.canvas.timestamp_roi_created.connect(self._on_manual_timestamp_roi_created)

    def _dispatch_command(self, command: AppCommand) -> None:
        if command == AppCommand.OPEN_VIDEO:
            self._open_video_dialog()
        elif command == AppCommand.SAVE_SESSION:
            self._save_session_dialog()
        elif command == AppCommand.LOAD_SESSION:
            self._load_session_dialog()
        elif command == AppCommand.EXPORT_METADATA:
            self._export_metadata_dialog()
        elif command == AppCommand.EXIT:
            self.close()
        elif command == AppCommand.ADD_ROI:
            self.canvas.begin_add_roi()
            self.status_label.setText("Drag on preview to create ROI")
            self._set_notice("Add ROI mode is on. Drag a rectangle in preview.")
        elif command == AppCommand.DELETE_SELECTED_ROI:
            self._delete_selected_roi()
        elif command == AppCommand.CLEAR_ALL_ROIS:
            self._clear_all_rois()
        elif command == AppCommand.PREFERENCES:
            self._open_preferences()
        elif command == AppCommand.ROTATE_PREV:
            self._rotate(-90)
        elif command == AppCommand.ROTATE_NEXT:
            self._rotate(90)
        elif command == AppCommand.ZOOM_IN:
            self.canvas.zoom_in()
        elif command == AppCommand.ZOOM_OUT:
            self.canvas.zoom_out()
        elif command == AppCommand.ZOOM_RESET:
            self.canvas.zoom_reset()
        elif command == AppCommand.TOGGLE_OVERLAYS:
            checked = self.menu_actions[AppCommand.TOGGLE_OVERLAYS].isChecked()
            self.canvas.set_show_overlays(checked)
        elif command == AppCommand.RUN_FULL:
            self._start_analysis(force_full=True)
        elif command == AppCommand.RUN_PARTIAL:
            self._start_analysis(force_full=False)
        elif command == AppCommand.STOP_ANALYSIS:
            self._stop_analysis()
        elif command == AppCommand.CLEAR_DETECTED_EVENTS:
            self._clear_detected_events(confirm=True)
        elif command == AppCommand.SET_MODE_A:
            self.mode_combo.setCurrentIndex(0)
        elif command == AppCommand.SET_MODE_B:
            self.mode_combo.setCurrentIndex(1)
        elif command == AppCommand.AUTO_OCR_ROI:
            self._auto_detect_timestamp_roi()
        elif command == AppCommand.MANUAL_OCR_ROI:
            self.canvas.begin_manual_timestamp_roi()
            self.status_label.setText("Drag on preview to set timestamp ROI")
            self._set_notice("Add TS ROI mode is on. Drag a rectangle over the timestamp text area.")
        elif command == AppCommand.SET_CORRECTION_MS:
            self._set_correction_dialog()
        elif command == AppCommand.RERUN_OCR_SELECTED_EVENT:
            self._rerun_ocr_on_selected_event()
        elif command == AppCommand.RESET_LAYOUT:
            self._reset_layout()
        elif command == AppCommand.OPEN_OUTPUT_FOLDER:
            self._open_output_folder()
        elif command == AppCommand.VIEW_LOGS:
            self._view_logs_dialog()
        elif command == AppCommand.ABOUT:
            self._about_dialog()
        elif command == AppCommand.DEBUG_OCR_CURRENT_FRAME:
            self._debug_ocr_current_frame()

        self._refresh_action_states()

    def _open_video_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            "",
            "Videos (*.mp4 *.avi *.mkv *.mov *.m4v *.ts *.mpeg *.mpg);;All files (*.*)",
        )
        if path:
            self._open_video(path)

    def _open_video(self, path: str) -> None:
        self._release_capture()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", f"Cannot open video:\n{path}")
            return

        self.capture = cap
        self.video_path = path
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.total_frames < 1:
            self.total_frames = 1

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, self.total_frames - 1))
        self.frame_slider.blockSignals(False)

        self._seek_frame(0)

        push_recent_video(path)
        self._refresh_recent_menu()

        self.rois.clear()
        self.direction_vectors.clear()
        self._next_roi_num = 1
        self._refresh_roi_list()
        self.ocr_manual_roi = None
        self.timestamp_ocr.set_manual_roi(None)
        self.timestamp_ocr.clear_auto_roi()
        self.canvas.set_timestamp_roi(None)

        self._clear_detected_events(confirm=False)
        self._reset_preview_inference(clear_detector=False, clear_overlay=True)

        self.output_dir = MetadataWriter.default_output_dir(path)
        self.setWindowTitle(f"ISAC Labelr - {Path(path).name}")
        self.status_label.setText(f"Loaded {Path(path).name}")
        self._set_notice(self._default_notice_text())
        self._refresh_action_states()

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _seek_frame(self, frame_index: int) -> None:
        if self.capture is None:
            return

        frame_index = max(0, min(frame_index, self.total_frames - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return

        self.current_frame_index = frame_index
        self.current_video_time_ms = int(round((frame_index / max(self.fps, 1.0)) * 1000.0))
        self.current_frame_raw = frame
        self.current_frame_rotated = rotate_bgr(frame, self.rotation_deg)
        self.canvas.set_frame(self.current_frame_rotated)
        if not (self.play_timer.isActive() and self.preview_detection_checkbox.isChecked()):
            self._clear_preview_overlays()

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_index)
        self.frame_slider.blockSignals(False)

        self.frame_label.setText(f"frame: {frame_index} / {self.total_frames - 1}")

    def _toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("Play Preview")
            self.status_label.setText("Preview paused (analysis is separate)")
            return

        if self.preview_detection_checkbox.isChecked():
            if not self._ensure_preview_detector():
                self.preview_detection_checkbox.blockSignals(True)
                self.preview_detection_checkbox.setChecked(False)
                self.preview_detection_checkbox.blockSignals(False)
                self._clear_preview_overlays()
                self._refresh_action_states()
                return
            self._reset_preview_tracker()
            self._reset_preview_event_state()

        interval = int(1000 / max(1.0, self.fps))
        self.play_timer.start(max(1, interval))
        self.play_button.setText("Pause Preview")
        self.status_label.setText("Preview playing (analysis not started)")

    def _play_tick(self) -> None:
        next_frame = self.current_frame_index + 1
        if next_frame >= self.total_frames:
            self.play_timer.stop()
            self.play_button.setText("Play Preview")
            return
        self._seek_frame(next_frame)
        if self.preview_detection_checkbox.isChecked():
            self._run_preview_detection_on_current_frame()

    def _on_preview_detection_toggled(self, checked: bool) -> None:
        if not checked:
            self._reset_preview_inference(clear_detector=False, clear_overlay=True)
            self.status_label.setText("Preview detection off")
            self._refresh_action_states()
            return

        if self.video_path is None:
            self.preview_detection_checkbox.blockSignals(True)
            self.preview_detection_checkbox.setChecked(False)
            self.preview_detection_checkbox.blockSignals(False)
            self.status_label.setText("Open a video first")
            self._refresh_action_states()
            return

        if not self._ensure_preview_detector():
            self.preview_detection_checkbox.blockSignals(True)
            self.preview_detection_checkbox.setChecked(False)
            self.preview_detection_checkbox.blockSignals(False)
            self._clear_preview_overlays()
            self._refresh_action_states()
            return

        self._reset_preview_tracker()
        self._reset_preview_event_state()
        self.status_label.setText("Preview detection on")
        self._refresh_action_states()

    def _ensure_preview_detector(self) -> bool:
        if self.preview_detector is not None and self.preview_tracker is not None:
            return True
        try:
            self.preview_detector = OnnxYoloPersonDetector(
                DetectorConfig(
                    model_path=self.preferences.detector_model_path,
                    confidence=self.preferences.detector_confidence,
                    iou=self.preferences.detector_iou,
                )
            )
            self.preview_tracker = ByteTrackLite(iou_threshold=self.preferences.detector_iou)
            return True
        except DetectorInitializationError as exc:
            QMessageBox.warning(self, "Preview Detection", f"Cannot start preview detection:\n{exc}")
        except Exception as exc:
            QMessageBox.warning(self, "Preview Detection", f"Failed to initialize detector:\n{exc}")

        self.preview_detector = None
        self.preview_tracker = None
        return False

    def _reset_preview_tracker(self) -> None:
        if self.preview_tracker is None:
            self.preview_tracker = ByteTrackLite(iou_threshold=self.preferences.detector_iou)
        else:
            self.preview_tracker.reset()

    def _clear_preview_overlays(self) -> None:
        self.canvas.set_tracks([])
        self.canvas.set_active_roi_ids(set())

    def _reset_preview_event_state(self) -> None:
        self.preview_presence.clear()
        self.preview_last_overlay_ts = None

    def _reset_preview_inference(self, *, clear_detector: bool, clear_overlay: bool) -> None:
        if self.preview_tracker is not None:
            self.preview_tracker.reset()
        self._reset_preview_event_state()
        if clear_detector:
            self.preview_detector = None
            self.preview_tracker = None
        if clear_overlay:
            self._clear_preview_overlays()

    def _run_preview_detection_on_current_frame(self) -> None:
        if self.current_frame_rotated is None:
            return
        if not self._ensure_preview_detector():
            self.preview_detection_checkbox.blockSignals(True)
            self.preview_detection_checkbox.setChecked(False)
            self.preview_detection_checkbox.blockSignals(False)
            self._clear_preview_overlays()
            self._refresh_action_states()
            return

        assert self.preview_detector is not None
        assert self.preview_tracker is not None

        try:
            detections = self.preview_detector.detect(self.current_frame_rotated)
            tracks = self.preview_tracker.update(detections)
        except Exception as exc:
            self.preview_detection_checkbox.blockSignals(True)
            self.preview_detection_checkbox.setChecked(False)
            self.preview_detection_checkbox.blockSignals(False)
            self._clear_preview_overlays()
            self.status_label.setText(f"Preview detection failed: {exc}")
            self._refresh_action_states()
            return

        active_roi_ids: set[str] = set()
        for roi_id, roi in self.rois.items():
            n = roi.normalized()
            for track in tracks:
                cx, cy = track.center
                if n.contains(cx, cy):
                    active_roi_ids.add(roi_id)
                    break

        self.canvas.set_tracks(tracks)
        self.canvas.set_active_roi_ids(active_roi_ids)

        if self.video_path is None:
            return

        preview_events = self.preview_event_engine._update_presence(
            rois=[roi.normalized() for roi in self.rois.values()],
            direction_vectors=dict(self.direction_vectors),
            mode=self._current_mode(),
            frame_index=self.current_frame_index,
            video_time_ms=self.current_video_time_ms,
            video_name=Path(self.video_path).name,
            tracks=tracks,
            presence=self.preview_presence,
        )
        if not preview_events:
            return

        overlay_ts_ms, overlay_status, ocr_conf = self._resolve_overlay_timestamp_for_frame(
            frame_bgr=self.current_frame_rotated,
            frame_index=self.current_frame_index,
            last_valid_ts=self.preview_last_overlay_ts,
        )
        if overlay_status in {OverlayTSStatus.OK, OverlayTSStatus.INTERPOLATED_NEIGHBOR}:
            self.preview_last_overlay_ts = overlay_ts_ms

        correction = int(self.correction_edit.text() or 0)
        corrected_ts = overlay_ts_ms + correction if overlay_ts_ms is not None else None

        for event in preview_events:
            event.overlay_ts_ms = overlay_ts_ms
            event.overlay_ts_status = overlay_status
            event.correction_ms = correction
            event.corrected_ts_ms = corrected_ts
            event.ocr_conf = ocr_conf
            self._on_analysis_event(event)

    def _resolve_overlay_timestamp_for_frame(
        self,
        *,
        frame_bgr,
        frame_index: int,
        last_valid_ts: int | None,
    ) -> tuple[int | None, OverlayTSStatus, float | None]:
        ocr_result = self.timestamp_ocr.extract_timestamp(frame_bgr)
        candidate_ts = (
            ocr_result.timestamp_ms
            if self.timestamp_ocr.is_valid_timestamp(ocr_result.timestamp_ms)
            else None
        )
        if candidate_ts is not None:
            return candidate_ts, OverlayTSStatus.OK, ocr_result.confidence

        if self.video_path is not None:
            neighbor_ts, neighbor_conf = self.timestamp_ocr.interpolate_timestamp_from_neighbors(
                video_path=self.video_path,
                frame_index=frame_index,
                rotation_deg=self.rotation_deg,
                fast=True,
            )
            if neighbor_ts is not None:
                conf = neighbor_conf if neighbor_conf is not None else ocr_result.confidence
                return neighbor_ts, OverlayTSStatus.INTERPOLATED_NEIGHBOR, conf

        if last_valid_ts is not None and self.timestamp_ocr.is_valid_timestamp(last_valid_ts):
            return last_valid_ts, OverlayTSStatus.INTERPOLATED_PREV, ocr_result.confidence
        return None, OverlayTSStatus.FAILED, ocr_result.confidence

    def _rotate(self, delta: int) -> None:
        if self.rois:
            answer = QMessageBox.question(
                self,
                "Rotate with existing ROI",
                (
                    "ROI was created on the current orientation.\n"
                    "If you rotate now, ROI coordinates can be misaligned.\n\n"
                    "Clear all ROIs and rotate?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return
            self.rois.clear()
            self.direction_vectors.clear()
            self._refresh_roi_list()
            self.canvas.set_rois(self.rois)
            self.canvas.set_direction_vectors(self.direction_vectors)

        self.rotation_deg = (self.rotation_deg + delta) % 360
        if self.current_frame_raw is not None:
            self.current_frame_rotated = rotate_bgr(self.current_frame_raw, self.rotation_deg)
            self.canvas.set_frame(self.current_frame_rotated)
        self.ocr_manual_roi = None
        self.timestamp_ocr.set_manual_roi(None)
        self.timestamp_ocr.clear_auto_roi()
        self.canvas.set_timestamp_roi(None)
        self._reset_preview_inference(clear_detector=False, clear_overlay=True)
        self.status_label.setText(f"Rotation: {self.rotation_deg} deg")
        self._set_notice("Rotation changed. Set TS ROI again before analysis.")
        self._refresh_action_states()

    def _toggle_partial_fields(self) -> None:
        full = self.full_checkbox.isChecked()
        self.start_ms_spin.setEnabled(not full)
        self.duration_ms_spin.setEnabled(not full)

    def _on_canvas_roi_created(self, roi: ROI) -> None:
        roi_id = f"roi_{self._next_roi_num}"
        self._next_roi_num += 1

        new_roi = ROI(roi_id=roi_id, x=roi.x, y=roi.y, w=roi.w, h=roi.h).normalized()
        self.rois[roi_id] = new_roi
        self.direction_vectors[roi_id] = Vector2(1.0, 0.0)

        self._refresh_roi_list(selected_roi_id=roi_id)
        self.canvas.set_rois(self.rois)
        self.canvas.set_direction_vectors(self.direction_vectors)
        self._reset_preview_event_state()
        self.status_label.setText(f"Added {roi_id}")
        self._refresh_action_states()

    def _on_canvas_roi_selected(self, roi_id: str) -> None:
        self._select_roi_in_list(roi_id)

    def _refresh_roi_list(self, selected_roi_id: str | None = None) -> None:
        self.roi_list.clear()
        for roi_id in sorted(self.rois.keys()):
            roi = self.rois[roi_id].normalized()
            item = QListWidgetItem()
            item.setData(Qt.UserRole, roi_id)
            self.roi_list.addItem(item)
            row_widget = self._build_roi_item_widget(roi_id, roi)
            item.setSizeHint(row_widget.sizeHint())
            self.roi_list.setItemWidget(item, row_widget)
            if selected_roi_id == roi_id:
                self.roi_list.setCurrentItem(item)

        if selected_roi_id is None and self.roi_list.count() > 0 and self.roi_list.currentItem() is None:
            self.roi_list.setCurrentRow(0)

    def _select_roi_in_list(self, roi_id: str) -> None:
        for i in range(self.roi_list.count()):
            item = self.roi_list.item(i)
            if item.data(Qt.UserRole) == roi_id:
                self.roi_list.setCurrentItem(item)
                break

    def _build_roi_item_widget(self, roi_id: str, roi: ROI) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(8)

        label_btn = QPushButton(f"{roi_id} ({roi.x},{roi.y},{roi.w},{roi.h})")
        label_btn.setFlat(True)
        label_btn.setStyleSheet("text-align: left;")
        label_btn.clicked.connect(lambda _=False, rid=roi_id: self._select_roi_in_list(rid))
        delete_btn = QPushButton("x")
        delete_btn.setFixedWidth(24)
        delete_btn.setToolTip(f"Delete {roi_id}")
        delete_btn.clicked.connect(lambda _=False, rid=roi_id: self._delete_roi_by_id(rid))

        layout.addWidget(label_btn, 1)
        layout.addWidget(delete_btn, 0)
        return widget

    def _on_roi_list_selection_changed(self, current: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        if current is None:
            self.canvas.set_selected_roi(None)
            self._refresh_action_states()
            return

        roi_id = str(current.data(Qt.UserRole))
        self.canvas.set_selected_roi(roi_id)
        vec = self.direction_vectors.get(roi_id, Vector2(1.0, 0.0))
        self.dx_spin.blockSignals(True)
        self.dy_spin.blockSignals(True)
        self.dx_spin.setValue(vec.dx)
        self.dy_spin.setValue(vec.dy)
        self.dx_spin.blockSignals(False)
        self.dy_spin.blockSignals(False)
        self._refresh_action_states()

    def _on_direction_changed(self) -> None:
        item = self.roi_list.currentItem()
        if item is None:
            return
        roi_id = str(item.data(Qt.UserRole))
        self.direction_vectors[roi_id] = Vector2(float(self.dx_spin.value()), float(self.dy_spin.value()))
        self.canvas.set_direction_vectors(self.direction_vectors)

    def _delete_selected_roi(self) -> None:
        item = self.roi_list.currentItem()
        if item is None:
            return
        roi_id = str(item.data(Qt.UserRole))
        self._delete_roi_by_id(roi_id)

    def _delete_roi_by_id(self, roi_id: str) -> None:
        if roi_id not in self.rois:
            return
        self.rois.pop(roi_id, None)
        self.direction_vectors.pop(roi_id, None)
        self._refresh_roi_list()
        self.canvas.set_rois(self.rois)
        self.canvas.set_direction_vectors(self.direction_vectors)
        self._reset_preview_event_state()
        self._refresh_action_states()

    def _clear_all_rois(self) -> None:
        if not self.rois:
            return
        answer = QMessageBox.question(self, "Clear ROIs", "Delete all ROIs?")
        if answer != QMessageBox.Yes:
            return
        self.rois.clear()
        self.direction_vectors.clear()
        self._refresh_roi_list()
        self.canvas.set_rois(self.rois)
        self.canvas.set_direction_vectors(self.direction_vectors)
        self._reset_preview_event_state()
        self._refresh_action_states()

    def _on_manual_timestamp_roi_created(self, roi: ROI) -> None:
        self.ocr_manual_roi = roi.normalized()
        self.timestamp_ocr.set_manual_roi(self.ocr_manual_roi)
        self.canvas.set_timestamp_roi(self.ocr_manual_roi)
        self.status_label.setText("Manual timestamp ROI set")
        self._set_notice("Timestamp ROI is set. You can start analysis.")

    def _auto_detect_timestamp_roi(self) -> None:
        if self.current_frame_rotated is None:
            return
        roi = self.timestamp_ocr.detect_auto_roi(self.current_frame_rotated)
        if roi is None:
            QMessageBox.information(self, "OCR ROI", "Failed to auto-detect timestamp ROI")
            self._set_notice("Auto TS ROI detect failed. Use Add TS ROI and draw it manually.")
            return
        self.canvas.set_timestamp_roi(roi)
        self.status_label.setText("Auto timestamp ROI detected")
        self._set_notice("Auto TS ROI detected. You can start analysis.")

    def _set_correction_dialog(self) -> None:
        current = int(self.correction_edit.text() or 0)
        value, ok = QInputDialog.getInt(
            self,
            "Timestamp correction",
            "Correction (ms)",
            value=current,
            minValue=-2_000_000_000,
            maxValue=2_000_000_000,
        )
        if ok:
            self.correction_edit.setText(str(value))

    def _debug_ocr_current_frame(self) -> None:
        if self.current_frame_rotated is None:
            QMessageBox.information(self, "OCR Debug", "No frame loaded")
            return

        if self.ocr_debug_worker is not None:
            QMessageBox.information(self, "OCR Debug", "OCR debug is already running")
            return

        frame = self.current_frame_rotated.copy()
        manual_roi = self.ocr_manual_roi.normalized() if self.ocr_manual_roi else None

        self.ocr_debug_thread = QThread(self)
        self.ocr_debug_worker = OCRDebugWorker(frame, manual_roi)
        self.ocr_debug_worker.moveToThread(self.ocr_debug_thread)

        self.ocr_debug_thread.started.connect(self.ocr_debug_worker.run)
        self.ocr_debug_worker.finished.connect(self._on_ocr_debug_finished)
        self.ocr_debug_worker.failed.connect(self._on_ocr_debug_failed)
        self.ocr_debug_worker.finished.connect(self.ocr_debug_thread.quit)
        self.ocr_debug_worker.failed.connect(self.ocr_debug_thread.quit)
        self.ocr_debug_thread.finished.connect(self._cleanup_ocr_debug_worker)

        self.status_label.setText("OCR debug running on current frame...")
        self._refresh_action_states()
        self.ocr_debug_thread.start()

    def _on_ocr_debug_finished(self, result, backend: str) -> None:
        ts_display = str(result.timestamp_ms) if result.timestamp_ms is not None else "-"
        conf_display = f"{result.confidence:.3f}" if result.confidence is not None else "-"
        valid_display = str(self.timestamp_ocr.is_valid_timestamp(result.timestamp_ms))
        roi_display = (
            f"{result.roi.roi_id} ({result.roi.x},{result.roi.y},{result.roi.w},{result.roi.h})"
            if result.roi is not None
            else "-"
        )

        if result.roi is not None:
            self.canvas.set_timestamp_roi(result.roi)

        QMessageBox.information(
            self,
            "OCR Debug (Current Frame)",
            "\n".join(
                [
                    f"backend: {backend}",
                    f"success: {result.success}",
                    f"timestamp_ms: {ts_display}",
                    f"valid_177_13digits: {valid_display}",
                    f"confidence: {conf_display}",
                    f"roi: {roi_display}",
                    f"raw_text: {result.raw_text!r}",
                ]
            ),
        )
        self.status_label.setText("OCR debug finished")

    def _on_ocr_debug_failed(self, message: str) -> None:
        QMessageBox.critical(self, "OCR Debug failed", message)
        self.status_label.setText("OCR debug failed")

    def _cleanup_ocr_debug_worker(self) -> None:
        if self.ocr_debug_worker is not None:
            self.ocr_debug_worker.deleteLater()
        if self.ocr_debug_thread is not None:
            self.ocr_debug_thread.deleteLater()
        self.ocr_debug_worker = None
        self.ocr_debug_thread = None
        self._refresh_action_states()

    def _current_mode(self) -> AnalysisMode:
        raw = self.mode_combo.currentData()
        if isinstance(raw, AnalysisMode):
            return raw
        if isinstance(raw, str):
            try:
                return AnalysisMode(raw)
            except ValueError:
                return AnalysisMode.ENTRY_ONLY
        return AnalysisMode.ENTRY_ONLY

    def _default_notice_text(self) -> str:
        return (
            "Instructions:\n"
            "1) Open video.\n"
            "2) Add ROI (required).\n"
            "3) Add TS ROI or Auto Detect Timestamp ROI (required).\n"
            "4) Choose Full/Partial and start analysis."
        )

    def _set_notice(self, message: str) -> None:
        self.notice_text.setPlainText(message.strip())

    def _has_timestamp_roi(self) -> bool:
        return (self.ocr_manual_roi is not None) or (self.timestamp_ocr.auto_roi is not None)

    def _validate_analysis_request(self, force_full: bool) -> str | None:
        if self.video_path is None:
            return "Open a video first."
        if not self.rois:
            return "Add at least one ROI before starting analysis."
        if not self._has_timestamp_roi():
            return "Set timestamp ROI first. Use Add TS ROI or Tools > Auto Detect Timestamp ROI."
        if (not force_full) and self.duration_ms_spin.value() <= 0:
            return "Duration must be greater than 0 for partial analysis."
        return None

    def _start_analysis(self, force_full: bool) -> None:
        if self.analysis_worker is not None:
            QMessageBox.warning(self, "Busy", "Analysis is already running")
            self._set_notice("Analysis is already running.")
            return

        validation_error = self._validate_analysis_request(force_full=force_full)
        if validation_error is not None:
            self._set_notice(validation_error)
            self.status_label.setText("Analysis blocked. Check TEXT NOTICE.")
            return

        analyze_full = force_full

        mode = self._current_mode()
        correction = int(self.correction_edit.text() or 0)
        analysis_ocr_roi = self.ocr_manual_roi
        if analysis_ocr_roi is None:
            analysis_ocr_roi = self.timestamp_ocr.auto_roi

        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("Play Preview")

        request = AnalysisRequest(
            video_path=self.video_path,
            mode=mode,
            rotation_deg=self.rotation_deg,
            rois=[roi.normalized() for roi in self.rois.values()],
            direction_vectors=dict(self.direction_vectors),
            analyze_full=analyze_full,
            start_ms=0 if analyze_full else int(self.start_ms_spin.value()),
            duration_ms=None if analyze_full else int(self.duration_ms_spin.value()),
            timestamp_correction_ms=correction,
            ocr_manual_roi=analysis_ocr_roi,
            detector_model_path=self.preferences.detector_model_path,
            detector_confidence=self.preferences.detector_confidence,
            detector_iou=self.preferences.detector_iou,
            chunk_seconds=self.preferences.chunk_seconds,
        )

        self._clear_detected_events(confirm=False)
        self.progress_bar.setValue(0)

        self.analysis_thread = QThread(self)
        self.analysis_worker = AnalysisWorker(request)
        self.analysis_worker.moveToThread(self.analysis_thread)

        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.progress.connect(self._on_analysis_progress)
        self.analysis_worker.event.connect(self._on_analysis_event)
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.failed.connect(self._on_analysis_failed)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_worker.failed.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(self._cleanup_analysis_worker)

        self.analysis_thread.start()
        scope = "full" if analyze_full else "partial"
        self.status_label.setText(f"Analysis started ({scope})")
        self._set_notice(f"Analysis started ({scope}).")
        self._refresh_action_states()

    def _stop_analysis(self) -> None:
        if self.analysis_worker is not None:
            self.analysis_worker.stop()
            self.status_label.setText("Stopping analysis...")

    def _cleanup_analysis_worker(self) -> None:
        if self.analysis_worker is not None:
            self.analysis_worker.deleteLater()
        if self.analysis_thread is not None:
            self.analysis_thread.deleteLater()
        self.analysis_worker = None
        self.analysis_thread = None
        self._refresh_action_states()

    def _on_analysis_progress(self, progress: AnalysisProgress) -> None:
        denom = max(1, self.total_frames - 1)
        pct = int((progress.frame_index / denom) * 100)
        pct = max(0, min(100, pct))
        self.progress_bar.setValue(pct)
        self.status_label.setText(
            f"frame={progress.frame_index} time_ms={progress.video_time_ms} events={progress.processed_events}"
        )

    def _on_analysis_event(self, event: EventRecord) -> None:
        index = len(self.events)
        self.events.append(event)

        item = QListWidgetItem(self._event_text(event))
        item.setData(Qt.UserRole, index)
        self.event_list.addItem(item)

    def _on_analysis_finished(self, result: AnalysisResult) -> None:
        self.progress_bar.setValue(100 if not result.aborted else self.progress_bar.value())
        self.status_label.setText(
            f"Analysis {'aborted' if result.aborted else 'finished'} | events={result.total_events}"
        )

        if result.output_dir:
            self.output_dir = result.output_dir
        elif self.video_path:
            self.output_dir = MetadataWriter.default_output_dir(self.video_path)

        self._refresh_action_states()

    def _on_analysis_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Analysis failed", message)
        self.status_label.setText("Analysis failed")
        self._refresh_action_states()

    def _event_text(self, event: EventRecord) -> str:
        corrected = event.corrected_ts_ms if event.corrected_ts_ms is not None else "-"
        direction = event.direction_label if event.direction_label is not None else "-"
        return (
            f"[{event.roi_id}] {event.event_type} id={event.person_id} "
            f"dir={direction} ts={corrected} frame={event.frame_index} "
            f"all={event.visible_person_count} roi={event.roi_person_count}"
        )

    def _on_event_item_clicked(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.UserRole)
        if idx is None:
            return
        event = self.events[int(idx)]
        self._seek_frame(event.frame_index)
        self._refresh_action_states()

    def _clear_detected_events(self, *, confirm: bool) -> None:
        if not self.events:
            return
        if confirm:
            answer = QMessageBox.question(self, "Clear Events", "Delete all detected events?")
            if answer != QMessageBox.Yes:
                return

        self.events.clear()
        self.event_list.clear()
        self._reset_preview_event_state()
        self._refresh_action_states()

    def _rerun_ocr_on_selected_event(self) -> None:
        item = self.event_list.currentItem()
        if item is None:
            return
        if self.video_path is None:
            return

        idx = int(item.data(Qt.UserRole))
        event = self.events[idx]

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, event.frame_index)
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            QMessageBox.warning(self, "OCR", "Unable to read target frame")
            return

        rotated = rotate_bgr(frame, self.rotation_deg)
        overlay, status, ocr_conf = self._resolve_overlay_timestamp_for_frame(
            frame_bgr=rotated,
            frame_index=event.frame_index,
            last_valid_ts=event.overlay_ts_ms,
        )

        correction = int(self.correction_edit.text() or 0)
        corrected = overlay + correction if overlay is not None else None

        event.overlay_ts_ms = overlay
        event.overlay_ts_status = status
        event.correction_ms = correction
        event.corrected_ts_ms = corrected
        event.ocr_conf = ocr_conf
        item.setText(self._event_text(event))

        if self.output_dir is not None:
            self._write_metadata(self.output_dir)

    def _write_metadata(self, output_dir: Path) -> None:
        writer = MetadataWriter(output_dir)
        try:
            for event in self.events:
                writer.write_event(event)
            writer.write_session_config(self._session_config().to_dict())
        finally:
            writer.close()

    def _export_metadata_dialog(self) -> None:
        if not self.events:
            QMessageBox.information(self, "Export", "No events to export")
            return

        default_dir = str(self.output_dir) if self.output_dir else ""
        selected = QFileDialog.getExistingDirectory(self, "Select output directory", default_dir)
        if not selected:
            return
        out = Path(selected)
        self._write_metadata(out)
        QMessageBox.information(self, "Export", f"Exported metadata to:\n{out}")

    def _save_session_dialog(self) -> None:
        if self.video_path is None:
            QMessageBox.information(self, "Save session", "Open a video first")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save session", "session.json", "JSON (*.json)")
        if not path:
            return

        config = self._session_config().to_dict()
        Path(path).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_session_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load session", "", "JSON (*.json)")
        if not path:
            return

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        config = SessionConfig.from_dict(raw)

        if not Path(config.video_path).exists():
            QMessageBox.critical(
                self,
                "Session",
                f"Video not found:\n{config.video_path}",
            )
            return

        self._open_video(config.video_path)
        self.rotation_deg = config.rotation_deg % 360
        self.mode_combo.setCurrentIndex(0 if config.mode == AnalysisMode.ENTRY_ONLY else 1)
        self.full_checkbox.setChecked(config.analyze_full)
        self.start_ms_spin.setValue(config.start_ms)
        self.duration_ms_spin.setValue(config.duration_ms or self.duration_ms_spin.value())
        self.correction_edit.setText(str(config.timestamp_correction_ms))

        self.rois = {roi.roi_id: roi for roi in config.rois}
        self.direction_vectors = dict(config.direction_vectors)
        for roi_id in self.rois:
            self.direction_vectors.setdefault(roi_id, Vector2(1.0, 0.0))

        self.ocr_manual_roi = config.ocr_manual_roi
        self.timestamp_ocr.set_manual_roi(self.ocr_manual_roi)
        self.canvas.set_timestamp_roi(self.ocr_manual_roi)

        self.canvas.set_rois(self.rois)
        self.canvas.set_direction_vectors(self.direction_vectors)
        self._refresh_roi_list()

        if self.current_frame_raw is not None:
            self.current_frame_rotated = rotate_bgr(self.current_frame_raw, self.rotation_deg)
            self.canvas.set_frame(self.current_frame_rotated)

        self._refresh_action_states()

    def _open_preferences(self) -> None:
        dialog = PreferencesDialog(self.preferences, self)
        if dialog.exec() == QDialog.Accepted:
            self.preferences = dialog.value()
            save_preferences(self.preferences)
            self._reset_preview_inference(clear_detector=True, clear_overlay=True)

    def _session_config(self) -> SessionConfig:
        mode = self._current_mode()
        return SessionConfig(
            video_path=self.video_path or "",
            rotation_deg=self.rotation_deg,
            mode=mode,
            rois=[roi.normalized() for roi in self.rois.values()],
            direction_vectors=dict(self.direction_vectors),
            analyze_full=self.full_checkbox.isChecked(),
            start_ms=int(self.start_ms_spin.value()),
            duration_ms=None if self.full_checkbox.isChecked() else int(self.duration_ms_spin.value()),
            timestamp_correction_ms=int(self.correction_edit.text() or 0),
            ocr_manual_roi=self.ocr_manual_roi,
        )

    def _refresh_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = load_recent_videos(limit=10)
        if not recent:
            empty = QAction("(empty)", self)
            empty.setEnabled(False)
            self.recent_menu.addAction(empty)
            return

        for path in recent:
            action = QAction(path, self)
            action.triggered.connect(lambda _checked=False, p=path: self._open_video(p))
            self.recent_menu.addAction(action)

    def _reset_layout(self) -> None:
        self.splitter.setSizes([900, 500])
        self.canvas.zoom_reset()

    def _open_output_folder(self) -> None:
        if self.output_dir is None:
            QMessageBox.information(self, "Output", "No output directory available")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))

    def _view_logs_dialog(self) -> None:
        if self.output_dir is None:
            QMessageBox.information(self, "Logs", "No output directory available")
            return

        path = self.output_dir / "run_log.txt"
        if not path.exists():
            QMessageBox.information(self, "Logs", "run_log.txt not found")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Run Logs")
        dlg.resize(900, 600)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
        layout.addWidget(text)
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.rejected.connect(dlg.reject)
        btn.accepted.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec()

    def _about_dialog(self) -> None:
        QMessageBox.information(
            self,
            "About",
            "ISAC Labelr\n"
            "Cross-platform ROI entry/exit metadata tool\n"
            f"Stack: PySide6 + ONNX Runtime | OCR: {self.timestamp_ocr.backend}",
        )

    def _refresh_action_states(self) -> None:
        has_video = self.video_path is not None
        running = self.analysis_worker is not None
        ocr_debug_running = self.ocr_debug_worker is not None
        has_roi = bool(self.rois)
        has_selected_roi = self.roi_list.currentItem() is not None
        has_events = bool(self.events)
        has_selected_event = self.event_list.currentItem() is not None

        enable_map = {
            AppCommand.SAVE_SESSION: has_video,
            AppCommand.LOAD_SESSION: True,
            AppCommand.EXPORT_METADATA: has_events,
            AppCommand.ADD_ROI: has_video and not running,
            AppCommand.DELETE_SELECTED_ROI: has_selected_roi and not running,
            AppCommand.CLEAR_ALL_ROIS: has_roi and not running,
            AppCommand.ROTATE_PREV: has_video and not running,
            AppCommand.ROTATE_NEXT: has_video and not running,
            AppCommand.ZOOM_IN: has_video,
            AppCommand.ZOOM_OUT: has_video,
            AppCommand.ZOOM_RESET: has_video,
            AppCommand.TOGGLE_OVERLAYS: has_video,
            AppCommand.RUN_FULL: has_video and has_roi and not running,
            AppCommand.RUN_PARTIAL: has_video and has_roi and not running,
            AppCommand.STOP_ANALYSIS: running,
            AppCommand.CLEAR_DETECTED_EVENTS: has_events,
            AppCommand.SET_MODE_A: has_video and not running,
            AppCommand.SET_MODE_B: has_video and not running,
            AppCommand.AUTO_OCR_ROI: has_video,
            AppCommand.MANUAL_OCR_ROI: has_video,
            AppCommand.SET_CORRECTION_MS: True,
            AppCommand.RERUN_OCR_SELECTED_EVENT: has_selected_event and has_video,
            AppCommand.OPEN_OUTPUT_FOLDER: self.output_dir is not None,
            AppCommand.VIEW_LOGS: self.output_dir is not None,
            AppCommand.RESET_LAYOUT: True,
            AppCommand.PREFERENCES: not running,
            AppCommand.DEBUG_OCR_CURRENT_FRAME: has_video and not ocr_debug_running,
        }

        for command, enabled in enable_map.items():
            action = self.menu_actions.get(command)
            if action is not None:
                action.setEnabled(enabled)

        for command, button in self.command_buttons.items():
            button.setEnabled(enable_map.get(command, True))

        preview_enabled = has_video and not running
        self.play_button.setEnabled(preview_enabled)
        self.prev_button.setEnabled(preview_enabled)
        self.next_button.setEnabled(preview_enabled)
        self.preview_detection_checkbox.setEnabled(preview_enabled)
        self.frame_slider.setEnabled(has_video)

        self._sync_mode_actions()

    def _sync_mode_actions(self) -> None:
        mode = self._current_mode()
        self.menu_actions[AppCommand.SET_MODE_A].setChecked(mode == AnalysisMode.ENTRY_ONLY)
        self.menu_actions[AppCommand.SET_MODE_B].setChecked(
            mode == AnalysisMode.ENTRY_EXIT_DIRECTION
        )

    def _on_mode_changed(self, _index: int) -> None:
        self._reset_preview_event_state()
        self._sync_mode_actions()

    def closeEvent(self, event) -> None:
        self.play_timer.stop()
        self._stop_analysis()
        if self.analysis_thread is not None:
            self.analysis_thread.quit()
            self.analysis_thread.wait(2000)
        if self.ocr_debug_thread is not None:
            self.ocr_debug_thread.quit()
            self.ocr_debug_thread.wait(2000)
        self._release_capture()
        super().closeEvent(event)
