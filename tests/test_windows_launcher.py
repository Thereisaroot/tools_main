from pathlib import Path


def test_windows_launcher_pulls_before_dependency_check_and_startup():
    launcher = (
        Path(__file__).resolve().parents[1] / "run_serial_text_chat.bat"
    ).read_text(encoding="utf-8")

    pull = launcher.index("git pull --ff-only")
    dependency_check = launcher.index('py -3 -c "import shooklink')
    startup = launcher.index("py -3 -m shooklink %*")

    assert pull < dependency_check < startup
    assert "Failed to update ShookLink." in launcher
