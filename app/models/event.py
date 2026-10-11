"""Region events: an object entered, left, was lost or stayed too long."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Kinds of region events."""

    ENTER = "ENTER"
    EXIT = "EXIT"
    LOST = "LOST"          # the track disappeared while inside (occlusion, left the frame)
    OVERSTAY = "OVERSTAY"  # dwell time exceeded the region's max_dwell_seconds


@dataclass(frozen=True)
class RegionEvent:
    """One event. ``timestamp`` is media seconds; ``dwell_seconds`` is set for EXIT, LOST and OVERSTAY."""

    event_id: str
    track_id: int
    roi_id: str
    event_type: EventType
    timestamp: float
    class_name: str
    dwell_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape from the technical document (section 5.3)."""
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "roi_id": self.roi_id,
            "event_type": self.event_type.value,
            "timestamp": round(self.timestamp, 2),
            "class_name": self.class_name,
            "dwell_seconds": None if self.dwell_seconds is None else round(self.dwell_seconds, 2),
        }
