"""Per-region metrics shown on the dashboard and written to the report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.track_session import TrackSession


@dataclass(frozen=True)
class RegionStatistics:
    """Snapshot of one region after a frame.

    ``occupancy`` is the number of objects inside now; ``entries`` is the
    cumulative number of ENTER events. They answer different questions and are
    never substituted for each other.
    """

    roi_id: str
    name: str
    occupancy: int = 0
    peak_occupancy: int = 0
    average_occupancy: float = 0.0
    entries: int = 0
    exits: int = 0
    lost: int = 0
    overstays: int = 0
    average_dwell_seconds: float = 0.0
    max_dwell_seconds: float = 0.0
    dwell_limit_seconds: float = 0.0
    active_sessions: tuple[TrackSession, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (active sessions excluded)."""
        return {
            "roi_id": self.roi_id,
            "name": self.name,
            "occupancy": self.occupancy,
            "peak_occupancy": self.peak_occupancy,
            "average_occupancy": round(self.average_occupancy, 2),
            "entries": self.entries,
            "exits": self.exits,
            "lost": self.lost,
            "overstays": self.overstays,
            "average_dwell_seconds": round(self.average_dwell_seconds, 2),
            "max_dwell_seconds": round(self.max_dwell_seconds, 2),
            "dwell_limit_seconds": self.dwell_limit_seconds,
        }
