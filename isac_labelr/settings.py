from __future__ import annotations

import json
from pathlib import Path

from isac_labelr.models import AppPreferences


CONFIG_DIR = Path.home() / ".isac_labelr"
PREFERENCES_PATH = CONFIG_DIR / "preferences.json"
RECENT_PATH = CONFIG_DIR / "recent_videos.json"


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_preferences() -> AppPreferences:
    ensure_config_dir()
    if not PREFERENCES_PATH.exists():
        return AppPreferences()
    try:
        data = json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
        return AppPreferences.from_dict(data)
    except Exception:
        return AppPreferences()


def save_preferences(pref: AppPreferences) -> None:
    ensure_config_dir()
    PREFERENCES_PATH.write_text(json.dumps(pref.to_dict(), indent=2), encoding="utf-8")


def load_recent_videos(limit: int = 10) -> list[str]:
    ensure_config_dir()
    if not RECENT_PATH.exists():
        return []
    try:
        data = json.loads(RECENT_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return [str(p) for p in data[:limit]]
    except Exception:
        return []


def push_recent_video(path: str, limit: int = 10) -> list[str]:
    ensure_config_dir()
    history = [path] + [p for p in load_recent_videos(limit=limit) if p != path]
    history = history[:limit]
    RECENT_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
