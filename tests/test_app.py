from shooklink import app
from shooklink.input.macos_backend import MacOSInputBackend
from shooklink.input.windows_backend import WindowsInputBackend


def test_input_backend_factory_selects_supported_platform(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "darwin")
    assert isinstance(app._build_input_backend(), MacOSInputBackend)

    monkeypatch.setattr(app.sys, "platform", "win32")
    assert isinstance(app._build_input_backend(), WindowsInputBackend)


def test_input_backend_factory_disables_unsupported_platform(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "linux")
    assert app._build_input_backend() is None
