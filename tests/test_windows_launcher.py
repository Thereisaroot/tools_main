from pathlib import Path


def test_windows_launcher_skips_git_update_and_starts_normally():
    launcher = (
        Path(__file__).resolve().parents[1] / "run_serial_text_chat.bat"
    ).read_text(encoding="utf-8")

    dependency_check = launcher.index('py -3 -c "import shooklink')
    startup = launcher.index("py -3 -m shooklink %*")

    assert "git pull" not in launcher
    assert "Updating ShookLink" not in launcher
    assert dependency_check < startup
