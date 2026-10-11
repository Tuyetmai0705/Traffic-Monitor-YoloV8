"""Result of processing one frame through the core pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import supervision as sv

from app.models.event import RegionEvent
from app.models.traffic_statistics import TrafficStatistics


@dataclass
class DetectionResult:
    """Tracked detections of one frame with speeds, violation flags and new region events."""

    frame_index: int
    detections: sv.Detections
    speeds: dict[int, float | None] = field(default_factory=dict)
    violating_ids: set[int] = field(default_factory=set)
    statistics: TrafficStatistics = field(default_factory=TrafficStatistics)
    region_events: list[RegionEvent] = field(default_factory=list)
