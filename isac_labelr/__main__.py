from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from isac_labelr.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ISAC Labelr")
    app.setOrganizationName("ISAC")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
