"""Application bootstrap and Qt-safe lifecycle wiring for ShookLink."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
import uuid
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, QStandardPaths, QTimer, Signal
from PySide6.QtWidgets import QApplication

from shooklink.core import CoreSnapshot, ShookLinkCore
from shooklink.input.topology import Side
from shooklink.protocol.crypto import IdentityStore, TrustStore
from shooklink.settings import AppSettings, SettingsStore
from shooklink.ui.main_window import MainWindow

logger = logging.getLogger(__name__)
_PEER_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShookLink serial peer client")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable protocol diagnostics without logging secure payloads",
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


def _load_or_create_peer_id(path: str | Path) -> str:
    peer_id_path = Path(path)
    try:
        stored = peer_id_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        stored = ""
    if _PEER_ID_PATTERN.fullmatch(stored):
        return stored

    peer_id = uuid.uuid4().hex
    peer_id_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=peer_id_path.parent,
            prefix=f".{peer_id_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(peer_id + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if os.name == "posix":
            temporary_path.chmod(0o600)
        os.replace(temporary_path, peer_id_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return peer_id


def _application_data_dir() -> Path:
    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    if location:
        return Path(location)
    return Path.home() / ".shooklink"


class _ControllerBridge(QObject):
    snapshot_received = Signal(object)
    operation_finished = Signal(int, str, object)


class ApplicationController(QObject):
    """Serializes blocking core operations away from the Qt GUI thread."""

    def __init__(
        self,
        core,
        window,
        settings_store: SettingsStore,
        settings: AppSettings,
        *,
        executor: Executor | None = None,
    ) -> None:
        super().__init__()
        self._core = core
        self._window = window
        self._settings_store = settings_store
        self._settings = settings
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="shooklink-control",
        )
        self._bridge = _ControllerBridge(self)
        self._bridge.snapshot_received.connect(self._apply_snapshot)
        self._bridge.operation_finished.connect(self._operation_finished)
        self._next_operation_id = 1
        self._active_operations: set[int] = set()
        self._closed = False
        self._started = False
        self._core_listener = self._bridge.snapshot_received.emit

        self._window.connect_requested.connect(self._connect_requested)
        self._window.disconnect_requested.connect(self._disconnect_requested)
        self._window.trust_requested.connect(self._trust_requested)
        self._window.preferences_changed.connect(self._save_preferences)
        self._core.add_state_listener(self._core_listener)
        self._window.set_serial_defaults(
            self._settings.last_port,
            self._settings.baud_rate,
        )
        self._window.apply_core_snapshot(self._core.snapshot)

    def start(self) -> None:
        if self._closed or self._started:
            return
        self._started = True
        if (
            self._settings.last_port
            and self._settings.last_port in self._window.available_ports()
        ):
            QTimer.singleShot(
                0,
                lambda: self._connect_requested(
                    self._settings.last_port,
                    self._settings.baud_rate,
                ),
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._core.remove_state_listener(self._core_listener)
        except (AttributeError, ValueError):
            pass
        self._save_preferences()
        self._core.close()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _connect_requested(self, port: str, baud_rate: int) -> None:
        if self._closed:
            return
        self._window.set_connection_pending(port)
        self._settings = AppSettings(
            last_port=port,
            baud_rate=baud_rate,
            peer_side=self._settings.peer_side,
            auto_edge_enabled=self._settings.auto_edge_enabled,
            download_dir=self._settings.download_dir,
            allow_remote_shell=self._settings.allow_remote_shell,
            allow_input=self._settings.allow_input,
        )
        self._submit("connect", self._core.connect_port, port, baud_rate)

    def _disconnect_requested(self) -> None:
        if not self._closed:
            self._submit("disconnect", self._core.disconnect)

    def _trust_requested(self, connection_id: int, fingerprint: str) -> None:
        if not self._closed:
            self._submit(
                "trust",
                self._core.approve_peer,
                connection_id,
                fingerprint,
            )

    def _submit(self, kind: str, callback, *args) -> None:
        operation_id = self._next_operation_id
        self._next_operation_id += 1
        self._active_operations.add(operation_id)
        future = self._executor.submit(callback, *args)
        future.add_done_callback(
            lambda completed, oid=operation_id, operation=kind: (
                self._bridge.operation_finished.emit(
                    oid,
                    operation,
                    _future_error(completed),
                )
            )
        )

    def _operation_finished(
        self,
        operation_id: int,
        kind: str,
        error: BaseException | None,
    ) -> None:
        if operation_id not in self._active_operations:
            return
        self._active_operations.remove(operation_id)
        if self._closed or error is None:
            return
        logger.debug("%s operation failed", kind, exc_info=error)
        if kind == "connect":
            self._window.show_connection_error(str(error))
            return
        self._window.apply_core_snapshot(self._core.snapshot)
        self._window.show_operation_error(str(error))

    def _apply_snapshot(self, snapshot: CoreSnapshot) -> None:
        if not self._closed:
            self._window.apply_core_snapshot(snapshot)

    def _save_preferences(self) -> None:
        try:
            (
                port,
                baud_rate,
                peer_side,
                auto_edge_enabled,
                allow_remote_shell,
                allow_input,
            ) = (
                self._window.persistent_preferences()
            )
            settings = AppSettings(
                last_port=port,
                baud_rate=baud_rate,
                peer_side=peer_side,
                auto_edge_enabled=auto_edge_enabled,
                download_dir=self._settings.download_dir,
                allow_remote_shell=allow_remote_shell,
                allow_input=allow_input,
            )
            self._settings_store.save(settings)
            self._settings = settings
        except Exception:
            logger.exception("could not save application settings")


def _apply_persisted_authorizations(core, settings: AppSettings) -> None:
    core.shell.set_allow_remote_shell(settings.allow_remote_shell)
    if core.input is not None:
        core.input.set_allow_remote_input(settings.allow_input)


def _future_error(future: Future) -> BaseException | None:
    try:
        return future.exception()
    except BaseException as error:
        return error


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    qt_app = QApplication.instance() or QApplication(["shooklink"])
    qt_app.setApplicationName("ShookLink")
    qt_app.setOrganizationName("ShookLink")
    if arguments.debug:
        qt_app.setProperty("shooklinkDebug", True)

    data_dir = _application_data_dir()
    settings_store = SettingsStore(data_dir / "settings.json")
    settings = settings_store.load()
    identity = IdentityStore(data_dir / "identity.key").load_or_create()
    core = ShookLinkCore(
        identity=identity,
        trust_store=TrustStore(data_dir / "trusted-peers.json"),
        download_dir=settings.download_dir,
        local_peer_id=_load_or_create_peer_id(data_dir / "installation-id"),
        input_backend=_build_input_backend(),
        peer_side=Side(settings.peer_side),
        auto_edge_enabled=settings.auto_edge_enabled,
        debug=arguments.debug,
    )
    _apply_persisted_authorizations(core, settings)
    window = MainWindow(
        core.chat,
        file_service=core.files,
        shell_service=core.shell,
        input_service=core.input,
    )
    controller = ApplicationController(
        core,
        window,
        settings_store,
        settings,
    )
    qt_app.aboutToQuit.connect(controller.close)
    window.emergency_exit_requested.connect(qt_app.quit)
    window.show()
    controller.start()
    try:
        return qt_app.exec()
    finally:
        controller.close()


__all__ = [
    "ApplicationController",
    "build_argument_parser",
    "main",
]
