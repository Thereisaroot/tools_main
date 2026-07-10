"""Application bootstrap for the ShookLink desktop client."""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from shooklink.chat.service import ChatService
from shooklink.input.service import InputService
from shooklink.protocol.messages import Message
from shooklink.transport.multiplexer import Priority
from shooklink.ui.main_window import MainWindow


class _DisconnectedBus:
    trusted = False

    def send(
        self,
        _message: Message,
        *,
        secure: bool = False,
        priority: Priority = Priority.NORMAL,
    ) -> None:
        raise RuntimeError("connect a serial peer first")

    def decrypt_secure(self, _message: Message) -> bytes:
        raise RuntimeError("connect a trusted serial peer first")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShookLink serial peer client")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable protocol diagnostics",
    )
    return parser


def _build_input_backend():
    if sys.platform == "darwin":
        from shooklink.input.macos_backend import MacOSInputBackend

        return MacOSInputBackend()
    if sys.platform == "win32":
        from shooklink.input.windows_backend import WindowsInputBackend

        return WindowsInputBackend()
    return None


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    qt_app = QApplication.instance() or QApplication(sys.argv[:1])
    qt_app.setApplicationName("ShookLink")
    qt_app.setOrganizationName("ShookLink")
    if arguments.debug:
        qt_app.setProperty("shooklinkDebug", True)
    bus = _DisconnectedBus()
    input_backend = _build_input_backend()
    input_service = (
        None
        if input_backend is None
        else InputService(
            bus,
            input_backend,
            local_peer_id="local-disconnected",
            peer_id="remote-disconnected",
        )
    )
    window = MainWindow(ChatService(bus), input_service=input_service)
    if input_service is not None:
        qt_app.aboutToQuit.connect(input_service.close)
    window.show()
    return qt_app.exec()


__all__ = ["build_argument_parser", "main"]
