"""Aggregated statistics for the dashboard, overlay and report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.event import RegionEvent
from app.models.region_statistics import RegionStatistics
from app.models.violation import Violation


@dataclass(frozen=True)
class TrafficStatistics:
    """Snapshot of counters and speed metrics after a frame."""

    total_in: int = 0
    total_out: int = 0
    by_class: dict[str, dict[str, int]] = field(default_factory=dict)
    violations_count: int = 0
    violations: tuple[Violation, ...] = ()
    average_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    active_tracks: int = 0
    fps: float = 0.0
    processing_ms: float = 0.0
    frame_index: int = 0
    total_frames: int = 0
    media_time: float = 0.0
    regions: tuple[RegionStatistics, ...] = ()
    recent_events: tuple[RegionEvent, ...] = ()
    events_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "by_class": {name: dict(counts) for name, counts in self.by_class.items()},
            "violations_count": self.violations_count,
            "violations": [violation.to_dict() for violation in self.violations],
            "average_speed_kmh": round(self.average_speed_kmh, 1),
            "max_speed_kmh": round(self.max_speed_kmh, 1),
            "active_tracks": self.active_tracks,
            "fps": round(self.fps, 2),
            "processing_ms": round(self.processing_ms, 2),
            "frame_index": self.frame_index,
            "total_frames": self.total_frames,
            "media_time": round(self.media_time, 2),
            "regions": [region.to_dict() for region in self.regions],
            "events_total": self.events_total,
        }

    @property
    def overstay_alerts(self) -> int:
        """Number of OVERSTAY alerts over all regions."""
        return sum(region.overstays for region in self.regions)
