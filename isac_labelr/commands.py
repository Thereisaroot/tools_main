from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AppCommand(Enum):
    OPEN_VIDEO = "open_video"
    SAVE_SESSION = "save_session"
    LOAD_SESSION = "load_session"
    EXPORT_METADATA = "export_metadata"
    EXIT = "exit"

    ADD_ROI = "add_roi"
    DELETE_SELECTED_ROI = "delete_selected_roi"
    CLEAR_ALL_ROIS = "clear_all_rois"
    PREFERENCES = "preferences"

    ROTATE_PREV = "rotate_prev"
    ROTATE_NEXT = "rotate_next"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    ZOOM_RESET = "zoom_reset"
    TOGGLE_OVERLAYS = "toggle_overlays"

    RUN_FULL = "run_full"
    RUN_PARTIAL = "run_partial"
    STOP_ANALYSIS = "stop_analysis"
    SET_MODE_A = "set_mode_a"
    SET_MODE_B = "set_mode_b"

    AUTO_OCR_ROI = "auto_ocr_roi"
    MANUAL_OCR_ROI = "manual_ocr_roi"
    SET_CORRECTION_MS = "set_correction_ms"
    RERUN_OCR_SELECTED_EVENT = "rerun_ocr_selected_event"

    RESET_LAYOUT = "reset_layout"

    OPEN_OUTPUT_FOLDER = "open_output_folder"
    VIEW_LOGS = "view_logs"
    ABOUT = "about"
    SEEK_EVENT = "seek_event"


@dataclass(slots=True)
class MenuActionSpec:
    command: AppCommand
    text: str
    shortcut: str | None = None
    checkable: bool = False


MenuActionMap = dict[str, list[MenuActionSpec]]


def build_menu_action_map() -> MenuActionMap:
    return {
        "File": [
            MenuActionSpec(AppCommand.OPEN_VIDEO, "Open Video", "Ctrl+O"),
            MenuActionSpec(AppCommand.SAVE_SESSION, "Save Session", "Ctrl+S"),
            MenuActionSpec(AppCommand.LOAD_SESSION, "Load Session", "Ctrl+Shift+O"),
            MenuActionSpec(AppCommand.EXPORT_METADATA, "Export Metadata", "Ctrl+E"),
            MenuActionSpec(AppCommand.EXIT, "Exit", "Ctrl+Q"),
        ],
        "Edit": [
            MenuActionSpec(AppCommand.ADD_ROI, "Add ROI", "R"),
            MenuActionSpec(AppCommand.DELETE_SELECTED_ROI, "Delete Selected ROI", "Del"),
            MenuActionSpec(
                AppCommand.CLEAR_ALL_ROIS,
                "Clear All ROIs",
                "Ctrl+Shift+Del",
            ),
            MenuActionSpec(AppCommand.PREFERENCES, "Preferences", "Ctrl+,"),
        ],
        "View": [
            MenuActionSpec(AppCommand.ROTATE_PREV, "Rotate -90", "["),
            MenuActionSpec(AppCommand.ROTATE_NEXT, "Rotate +90", "]"),
            MenuActionSpec(AppCommand.ZOOM_IN, "Zoom In", "Ctrl++"),
            MenuActionSpec(AppCommand.ZOOM_OUT, "Zoom Out", "Ctrl+-"),
            MenuActionSpec(AppCommand.ZOOM_RESET, "Zoom Reset", "Ctrl+0"),
            MenuActionSpec(
                AppCommand.TOGGLE_OVERLAYS,
                "Toggle Overlays",
                "T",
                checkable=True,
            ),
        ],
        "Analyze": [
            MenuActionSpec(AppCommand.RUN_FULL, "Full Analysis", "F5"),
            MenuActionSpec(AppCommand.RUN_PARTIAL, "Partial Analysis", "F6"),
            MenuActionSpec(AppCommand.STOP_ANALYSIS, "Stop", "Shift+F5"),
            MenuActionSpec(AppCommand.SET_MODE_A, "Mode A (Entry)", "Ctrl+1"),
            MenuActionSpec(AppCommand.SET_MODE_B, "Mode B (Entry+Exit)", "Ctrl+2"),
        ],
        "Tools": [
            MenuActionSpec(AppCommand.AUTO_OCR_ROI, "Auto Detect Timestamp ROI", "Ctrl+D"),
            MenuActionSpec(AppCommand.MANUAL_OCR_ROI, "Manual Timestamp ROI", "Ctrl+M"),
            MenuActionSpec(AppCommand.SET_CORRECTION_MS, "Set Timestamp Correction (ms)", "Ctrl+K"),
            MenuActionSpec(
                AppCommand.RERUN_OCR_SELECTED_EVENT,
                "Re-run OCR on Selected Event",
                "Ctrl+R",
            ),
        ],
        "Window": [
            MenuActionSpec(AppCommand.RESET_LAYOUT, "Reset Layout", "Ctrl+Shift+L"),
        ],
        "Help": [
            MenuActionSpec(AppCommand.OPEN_OUTPUT_FOLDER, "Open Output Folder", "Ctrl+Shift+E"),
            MenuActionSpec(AppCommand.VIEW_LOGS, "View Logs"),
            MenuActionSpec(AppCommand.ABOUT, "About"),
        ],
    }
