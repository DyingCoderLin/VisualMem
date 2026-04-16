"""
Timezone conversion helpers for the Data Platform API.

Strategy (per docs/data_platform_api_design.md § 〇):
  - DB stores UTC (frames/sub_frames timestamps).
  - API accepts local time (with or without offset).
  - API responds with local time + timezone offset.
"""

from datetime import datetime, timezone
from typing import Optional


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Parse a datetime and guarantee it is UTC-aware.

    - If *dt* already carries a tzinfo → convert to UTC.
    - If *dt* is naive → treat as **server-local** time, then convert to UTC.
      (Server-local == user-local because VisualMem is a local-deploy app.)
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        # naive → assume local, astimezone infers the system tz first
        return dt.astimezone(timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_dt(raw: str) -> datetime:
    """Parse an ISO-format string into a datetime.

    Supports both offset-aware ("2026-04-02T09:00:00+08:00") and naive
    ("2026-04-02T09:00:00") inputs.
    """
    return datetime.fromisoformat(raw)


def parse_utc(raw: str) -> datetime:
    """Convenience: parse *and* convert to UTC in one call."""
    return ensure_utc(parse_dt(raw))


def to_local_iso(dt: datetime) -> str:
    """UTC datetime → local-time ISO string **with** timezone offset suffix.

    Example:
        DB value  : 2026-04-02T01:00:00   (UTC)
        Returned  : 2026-04-02T09:00:00+08:00  (for an Asia/Shanghai server)
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().isoformat()


def db_ts_to_local(ts_str: str) -> str:
    """Convert a raw DB timestamp string (assumed UTC) to local ISO with offset."""
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().isoformat()
