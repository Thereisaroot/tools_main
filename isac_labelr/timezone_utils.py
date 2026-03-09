from __future__ import annotations

from datetime import timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def get_kst_tz() -> tzinfo:
    """Return Asia/Seoul timezone, with fixed-offset fallback on systems without tzdata."""
    try:
        return ZoneInfo("Asia/Seoul")
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=9), name="KST")
