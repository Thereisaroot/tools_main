import json
from dataclasses import FrozenInstanceError

import pytest

from shooklink.settings import AppSettings, SettingsStore


def test_settings_round_trip(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    expected = AppSettings(last_port="COM7", baud_rate=460800, peer_side="left")

    store.save(expected)

    assert store.load() == expected


def test_remote_permissions_never_restore_enabled(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"allow_remote_shell": True, "allow_input": True}),
        encoding="utf-8",
    )

    settings = SettingsStore(path).load()

    assert settings.allow_remote_shell is False
    assert settings.allow_input is False


def test_active_remote_permissions_are_never_persisted(tmp_path):
    path = tmp_path / "settings.json"
    store = SettingsStore(path)

    store.save(AppSettings(allow_remote_shell=True, allow_input=True))

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["allow_remote_shell"] is False
    assert persisted["allow_input"] is False


def test_invalid_peer_side_falls_back_to_default(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"peer_side": "diagonal"}), encoding="utf-8")

    settings = SettingsStore(path).load()

    assert settings.peer_side == AppSettings().peer_side


@pytest.mark.parametrize("invalid_baud", [0, -1, "460800", True, 100_000_000])
def test_invalid_baud_rate_falls_back_to_default(tmp_path, invalid_baud):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"baud_rate": invalid_baud}), encoding="utf-8")

    settings = SettingsStore(path).load()

    assert settings.baud_rate == AppSettings().baud_rate


def test_unknown_json_keys_are_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"last_port": "COM9", "future_setting": {"enabled": True}}),
        encoding="utf-8",
    )

    settings = SettingsStore(path).load()

    assert settings.last_port == "COM9"
    assert not hasattr(settings, "future_setting")


@pytest.mark.parametrize("contents", [None, "not json", "[]"])
def test_missing_or_corrupt_settings_return_defaults(tmp_path, contents):
    path = tmp_path / "settings.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")

    assert SettingsStore(path).load() == AppSettings()


def test_app_settings_is_frozen():
    settings = AppSettings()

    with pytest.raises(FrozenInstanceError):
        settings.last_port = "COM8"
