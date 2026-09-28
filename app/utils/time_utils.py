"""Time formatting helpers."""

from __future__ import annotations

from datetime import datetime


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.s``."""
    seconds = max(0.0, seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:04.1f}"


def frame_to_timestamp(frame_index: int, fps: float) -> str:
    """Convert a frame index to a video timestamp."""
    return format_timestamp(frame_index / fps if fps > 0 else 0.0)


def now_iso() -> str:
    """Return the local time in ISO 8601 without microseconds."""
    return datetime.now().replace(microsecond=0).isoformat()
