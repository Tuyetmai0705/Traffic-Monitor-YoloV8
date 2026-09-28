"""Multi-object tracking with ByteTrack plus box smoothing."""

from __future__ import annotations

import warnings

import numpy as np
import supervision as sv

from app.config.settings import TrackerConfig


class VehicleTracker:
    """Assign stable ``tracker_id`` values and smooth boxes to reduce jitter."""

    def __init__(self, config: TrackerConfig, frame_rate: float) -> None:
        """Create ByteTrack with the real video frame rate."""
        self._config = config
        self._frame_rate = frame_rate
        self._tracker = self._create_tracker()
        self._smoother = sv.DetectionsSmoother(length=config.smoother_length)
        self._last_ids: list[int] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Track then smooth detections. The result always has a ``tracker_id`` array."""
        tracked = self._tracker.update_with_detections(detections)
        if len(tracked) > 0:
            tracked = self._smoother.update_with_detections(tracked)
        if tracked.tracker_id is None:
            tracked.tracker_id = np.empty(len(tracked), dtype=int)
        self._last_ids = [int(tracker_id) for tracker_id in tracked.tracker_id]
        return tracked

    def reset(self) -> None:
        """Forget every track, e.g. before re-running a video."""
        self._tracker.reset()
        self._smoother = sv.DetectionsSmoother(length=self._config.smoother_length)
        self._last_ids = []

    def get_tracker_ids(self) -> list[int]:
        """Tracker ids present in the latest frame."""
        return list(self._last_ids)

    def _create_tracker(self) -> sv.ByteTrack:
        # supervision 0.28+ marks ByteTrack as deprecated but it is still the reference tracker.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return sv.ByteTrack(
                track_activation_threshold=self._config.track_activation_threshold,
                lost_track_buffer=self._config.lost_track_buffer,
                minimum_matching_threshold=self._config.minimum_matching_threshold,
                frame_rate=self._frame_rate,
                minimum_consecutive_frames=self._config.minimum_consecutive_frames,
            )
