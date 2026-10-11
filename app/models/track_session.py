"""One visit of a tracked object inside one region."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    """Lifecycle of a visit."""

    INSIDE = "inside"
    EXITED = "exited"
    LOST = "lost"


@dataclass(frozen=True)
class TrackSession:
    """Snapshot of a visit. Times are media seconds (frame index / fps)."""

    session_id: int
    track_id: int
    class_name: str
    roi_id: str
    entry_time: float
    last_seen_time: float
    exit_time: float | None
    dwell_seconds: float
    status: SessionStatus
    overstay: bool = False

    @property
    def is_open(self) -> bool:
        """True while the object is still considered inside."""
        return self.status is SessionStatus.INSIDE

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON shape from the technical document (section 5.2)."""
        return {
            "session_id": self.session_id,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "roi_id": self.roi_id,
            "entry_time": round(self.entry_time, 2),
            "last_seen_time": round(self.last_seen_time, 2),
            "exit_time": None if self.exit_time is None else round(self.exit_time, 2),
            "dwell_seconds": round(self.dwell_seconds, 2),
            "status": self.status.value,
            "overstay": self.overstay,
        }
