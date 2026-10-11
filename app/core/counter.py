"""Vehicle counting across a line with ``sv.LineZone``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import supervision as sv

from app.constants.classes import OBJECT_CLASSES


class VehicleCounter:
    """Count IN/OUT crossings in total and per vehicle class."""

    def __init__(self, start: Sequence[int], end: Sequence[int]) -> None:
        """Create the line zone anchored at the bottom centre of each box."""
        self._start = sv.Point(int(start[0]), int(start[1]))
        self._end = sv.Point(int(end[0]), int(end[1]))
        self._line_zone = self._create_line_zone()
        self._class_counts = self._empty_counts()

    @property
    def line_zone(self) -> sv.LineZone:
        """Underlying line zone, used by the annotator."""
        return self._line_zone

    def count(self, detections: sv.Detections) -> tuple[np.ndarray, np.ndarray]:
        """Trigger the line zone exactly once and accumulate per-class counts."""
        if detections.tracker_id is None:
            raise ValueError("detections must be tracked before counting")
        crossed_in, crossed_out = self._line_zone.trigger(detections)
        self._accumulate(detections.class_id[crossed_in], "in")
        self._accumulate(detections.class_id[crossed_out], "out")
        return crossed_in, crossed_out

    def get_total_in(self) -> int:
        """Total vehicles crossing in the IN direction."""
        return int(self._line_zone.in_count)

    def get_total_out(self) -> int:
        """Total vehicles crossing in the OUT direction."""
        return int(self._line_zone.out_count)

    def get_class_counts(self) -> dict[str, dict[str, int]]:
        """Per-class counts for every supported class, e.g. ``{"car": {"in": 3, "out": 1}}``."""
        return {name: dict(counts) for name, counts in self._class_counts.items()}

    def reset(self) -> None:
        """Reset all counters."""
        self._line_zone = self._create_line_zone()
        self._class_counts = self._empty_counts()

    def _create_line_zone(self) -> sv.LineZone:
        return sv.LineZone(start=self._start, end=self._end, triggering_anchors=[sv.Position.BOTTOM_CENTER])

    def _accumulate(self, class_ids: np.ndarray, direction: str) -> None:
        for class_id in class_ids:
            name = OBJECT_CLASSES.get(int(class_id))
            if name is not None:
                self._class_counts[name][direction] += 1

    @staticmethod
    def _empty_counts() -> dict[str, dict[str, int]]:
        return {name: {"in": 0, "out": 0} for name in OBJECT_CLASSES.values()}
