"""Current, peak and average number of objects inside each region."""

from __future__ import annotations

from collections.abc import Iterable


class OccupancyCounter:
    """Occupancy(ROI, t) = number of unique track ids inside the ROI at frame t.

    This is a current count, not a cumulative number of entries.
    """

    def __init__(self, roi_ids: Iterable[str]) -> None:
        """Create zeroed counters for ``roi_ids``."""
        self._roi_ids = list(roi_ids)
        self.reset()

    def update(self, inside_tracks: dict[str, set[int]]) -> dict[str, int]:
        """Record one frame and return the current occupancy per region."""
        self._frames += 1
        for roi_id in self._roi_ids:
            count = len(inside_tracks.get(roi_id, set()))
            self._current[roi_id] = count
            self._peak[roi_id] = max(self._peak[roi_id], count)
            self._sum[roi_id] += count
        return dict(self._current)

    def get_occupancy(self, roi_id: str) -> int:
        """Objects inside ``roi_id`` at the last frame."""
        return self._current[roi_id]

    def get_peak(self, roi_id: str) -> int:
        """Highest occupancy seen in ``roi_id``."""
        return self._peak[roi_id]

    def get_average(self, roi_id: str) -> float:
        """Mean occupancy over all processed frames."""
        return self._sum[roi_id] / self._frames if self._frames else 0.0

    def reset(self) -> None:
        """Zero every counter."""
        self._frames = 0
        self._current = {roi_id: 0 for roi_id in self._roi_ids}
        self._peak = {roi_id: 0 for roi_id in self._roi_ids}
        self._sum = {roi_id: 0 for roi_id in self._roi_ids}
