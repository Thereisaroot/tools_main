from __future__ import annotations

import sys

import cv2
from PySide6.QtWidgets import QApplication

from isac_labelr.ui.main_window import MainWindow


def main() -> int:
    # Keep OpenCV thread/buffer usage predictable on long runs.
    try:
        cv2.setNumThreads(1)
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("ISAC Labelr")
    app.setOrganizationName("ISAC")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
