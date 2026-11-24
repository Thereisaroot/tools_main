"""Offline xterm.js renderer. The only input destination is its owning shell session."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QKeySequence
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import (
    QWebEnginePage, QWebEngineProfile, QWebEngineSettings, QWebEngineUrlRequestInterceptor,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication
from shiboken6 import delete as delete_qobject

from shooklink.live_update.release import TERMINAL_ASSETS

ASSETS = Path(__file__).resolve().with_name("terminal_assets")
INDEX = QUrl.fromLocalFile(str(ASSETS / "index.html"))
_URLS = {QUrl.fromLocalFile(str(ASSETS / Path(name).name)).toString()
         for name in TERMINAL_ASSETS if not name.endswith(".txt")}
_URLS.add("qrc:///qtwebchannel/qwebchannel.js")


def asset_url_allowed(url: QUrl) -> bool:
    return url.toString() in _URLS


class OutputQueue:
    def __init__(self, limit=2 * 1024 * 1024, batch=48 * 1024):
        self.limit, self.batch = limit, batch
        self.waiting = bytearray()
        self.inflight = b""

    def append(self, data: bytes):
        if len(self.inflight) + len(self.waiting) + len(data) > self.limit:
            raise BufferError("terminal renderer fell behind")
        self.waiting.extend(data)

    def take(self) -> bytes:
        if self.inflight or not self.waiting:
            return b""
        self.inflight = bytes(self.waiting[:self.batch])
        del self.waiting[:self.batch]
        return self.inflight

    def ack(self):
        self.inflight = b""

    def pending(self):
        return self.inflight + self.waiting


class _Gate(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, request):
        request.block(request.requestMethod() != b"GET" or not asset_url_allowed(request.requestUrl()))


class _Page(QWebEnginePage):
    def acceptNavigationRequest(self, url, _kind, main_frame):
        return main_frame and url == INDEX

    def createWindow(self, _kind):
        return None


def _profile():
    profile = QWebEngineProfile()  # Off the record; no server, cookies or disk cache.
    profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.NoCache)
    profile._gate = _Gate(profile)
    profile.setUrlRequestInterceptor(profile._gate)
    profile.downloadRequested.connect(lambda download: download.cancel())
    return profile


class _Bridge(QObject):
    output = Signal(str)
    pasteRequested = Signal(str)
    findRequested = Signal(str, bool)
    fontRequested = Signal(int)

    def __init__(self, view):
        super().__init__(view)
        self.view = view

    @Slot()
    def started(self):
        self.view._started()

    @Slot()
    def drained(self):
        self.view._queue.ack()
        self.view._send_next()

    @Slot(str)
    def keys(self, encoded):
        if self.view.is_ready and 0 < len(encoded) <= 21848:
            try:
                data = base64.b64decode(encoded, validate=True)
            except ValueError:
                return
            if len(data) <= 16384:
                self.view.input_bytes.emit(data)

    @Slot(int, int)
    def dimensions(self, columns, rows):
        if 2 <= columns <= 500 and 2 <= rows <= 500:
            self.view.terminal_resized.emit(columns, rows)

    @Slot(bool)
    def found(self, matched):
        self.view.search_result.emit(matched)


class WebTerminal(QWebEngineView):
    input_bytes = Signal(bytes)
    terminal_resized = Signal(int, int)
    ready = Signal()
    failed = Signal(str, bytes)
    search_result = Signal(bool)
    find_shortcut = Signal()
    copy_shortcut = Signal()
    paste_shortcut = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        if not all((ASSETS / Path(name).name).is_file() for name in TERMINAL_ASSETS):
            raise OSError("offline terminal files are missing")
        self.is_ready = False
        self._broken = False
        self._queue = OutputQueue()
        self._bridge = _Bridge(self)
        self._owned_profile = _profile()
        page = self._owned_page = _Page(self._owned_profile, self)
        self._owned_profile.setParent(self)
        self.setPage(page)
        attr = QWebEngineSettings.WebAttribute
        for option in (attr.LocalContentCanAccessRemoteUrls, attr.JavascriptCanAccessClipboard,
                       attr.JavascriptCanPaste, attr.JavascriptCanOpenWindows, attr.LocalStorageEnabled,
                       attr.PluginsEnabled, attr.FullScreenSupportEnabled):
            page.settings().setAttribute(option, False)
        self.setAcceptDrops(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        page.permissionRequested.connect(lambda permission: permission.deny())
        self._channel = QWebChannel(page)
        self._channel.registerObject("terminalIO", self._bridge)
        page.setWebChannel(self._channel)
        self._watchdog = QTimer(self)
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(lambda: self._fail("terminal renderer did not start"))
        self._watchdog.start(8000)
        self.loadFinished.connect(lambda ok: None if ok else self._fail("terminal page could not load"))
        self.renderProcessTerminated.connect(lambda *_: self._fail("terminal renderer stopped"))
        QApplication.instance().aboutToQuit.connect(self.shutdown)
        self.load(INDEX)

    @property
    def output_pending(self):
        return bool(self._queue.inflight or self._queue.waiting)

    def _started(self):
        if self._broken or self.is_ready:
            return
        self._watchdog.stop()
        self.is_ready = True
        if self.focusProxy() is not None:
            self.focusProxy().installEventFilter(self)
        self.ready.emit()
        self._send_next()

    def eventFilter(self, target, event):
        if self.is_ready and event.type() in (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride):
            mods = event.modifiers()
            action = None
            if (event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab)
                    and mods in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier)):
                reverse = event.key() == Qt.Key.Key_Backtab or bool(mods & Qt.KeyboardModifier.ShiftModifier)
                action = lambda: self.input_bytes.emit(b"\x1b[Z" if reverse else b"\t")
            elif mods == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
                action = {Qt.Key.Key_F: self.find_shortcut.emit, Qt.Key.Key_C: self.copy_shortcut.emit,
                          Qt.Key.Key_V: self.paste_shortcut.emit}.get(event.key())
            elif sys.platform == "darwin":
                for key, signal in ((QKeySequence.StandardKey.Copy, self.copy_shortcut),
                                    (QKeySequence.StandardKey.Paste, self.paste_shortcut),
                                    (QKeySequence.StandardKey.Find, self.find_shortcut)):
                    if event.matches(key):
                        action = signal.emit
                        break
            if action is None:
                return super().eventFilter(target, event)
            if event.type() == QEvent.Type.KeyPress:
                action()
            event.accept()
            return True
        return super().eventFilter(target, event)

    def _fail(self, reason, extra=b""):
        if self._broken:
            return
        self._broken, self.is_ready = True, False
        self._watchdog.stop()
        pending = self._queue.pending() + extra
        self._queue = OutputQueue()
        self.failed.emit(reason, pending)

    def feed_output(self, data):
        if self._broken:
            return
        try:
            self._queue.append(data)
        except BufferError as error:
            self._fail(str(error), data)
            return
        self._send_next()

    def _send_next(self):
        if self.is_ready and (part := self._queue.take()):
            self._bridge.output.emit(base64.b64encode(part).decode("ascii"))

    def focus_terminal(self):
        self.setFocus()
        if self.is_ready:
            self.page().runJavaScript("window.shookTerm.focus()")

    def find_text(self, query, previous=False):
        if self.is_ready:
            self._bridge.findRequested.emit(query, previous)

    def paste_text(self, text):
        if self.is_ready and text:
            self._bridge.pasteRequested.emit(text)

    def set_font_size(self, pixels):
        if self.is_ready:
            self._bridge.fontRequested.emit(max(9, min(32, pixels)))

    def copy_selection(self):
        def copy(text):
            if isinstance(text, str) and text:
                QApplication.clipboard().setText(text)
        if self.is_ready:
            self.page().runJavaScript("window.shookTerm.selection()", 0, copy)

    def shutdown(self):
        if self._owned_page is None:
            return
        self.is_ready, self._broken = False, True
        self._watchdog.stop()
        self.hide()
        page, self._owned_page = self._owned_page, None
        page.setWebChannel(None)
        # setPage(None) can create a replacement page. Explicitly dispose of our
        # page on the GUI thread before its private profile and the Qt event loop.
        delete_qobject(page)
        delete_qobject(self._owned_profile)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
