"""Application bootstrap for the ShookLink desktop client."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from shooklink.chat.service import ChatService
from shooklink.protocol.messages import Message
from shooklink.ui.main_window import MainWindow


class _DisconnectedBus:
    trusted = False

    def send(self, _message: Message, *, secure: bool = False) -> None:
        raise RuntimeError("connect a serial peer first")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShookLink serial peer client")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable protocol diagnostics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    qt_app = QApplication.instance() or QApplication(sys.argv[:1])
    qt_app.setApplicationName("ShookLink")
    qt_app.setOrganizationName("ShookLink")
    if arguments.debug:
        qt_app.setProperty("shooklinkDebug", True)
    window = MainWindow(ChatService(_DisconnectedBus()))
    window.show()
    return qt_app.exec()


__all__ = ["build_argument_parser", "main"]
