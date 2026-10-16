"""Multi-object tracking with ByteTrack, box smoothing and per-track class voting."""

from __future__ import annotations

import warnings

import numpy as np
import supervision as sv

from app.config.settings import TrackerConfig
from app.core.box_smoother import COASTING_KEY, CONFIRMED_KEY, KalmanBoxSmoother
from app.core.class_voter import ClassVoter

SMOOTHING_MODES = ("kalman", "moving_average", "none")


class VehicleTracker:
    """Assign stable ``tracker_id`` values, steady the boxes and keep one class per track."""

    def __init__(self, config: TrackerConfig, frame_rate: float) -> None:
        """Create ByteTrack with the real video frame rate."""
        if config.smoothing not in SMOOTHING_MODES:
            raise ValueError(f"smoothing phải là một trong {SMOOTHING_MODES}")
        self._config = config
        self._frame_rate = frame_rate
        self._tracker = self._create_tracker()
        self._create_filters()
        self._last_ids: list[int] = []

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Track, vote classes, then smooth. The result always has a ``tracker_id`` array.

        With ``kalman`` smoothing, tracks seen fewer than ``min_hits`` frames are
        not returned, so one-frame false detections never reach counting or regions.
        """
        tracked = self._tracker.update_with_detections(detections)
        if tracked.tracker_id is None:
            tracked.tracker_id = np.empty(len(tracked), dtype=int)
        if self._config.smoothing == "kalman":
            tracked = self._kalman.update(tracked, hide_unconfirmed=False)  # stitches ids, adds coasting boxes
        elif self._config.smoothing == "moving_average" and len(tracked) > 0:
            tracked = self._average.update_with_detections(tracked)
        if tracked.tracker_id is None:
            tracked.tracker_id = np.empty(len(tracked), dtype=int)
        if len(tracked) > 0:                             # vote on the ids shown to the user
            measured = ~tracked.data.get(COASTING_KEY, np.zeros(len(tracked), dtype=bool))
            voted = tracked.class_id.copy()
            voted[measured] = self._voter.vote(tracked.tracker_id[measured], tracked.class_id[measured],
                                               None if tracked.confidence is None else tracked.confidence[measured])
            tracked.class_id = voted
            if CONFIRMED_KEY in tracked.data:            # vote saw every frame; now hide unconfirmed tracks
                tracked = tracked[tracked.data[CONFIRMED_KEY]]
        self._last_ids = [int(tracker_id) for tracker_id in tracked.tracker_id]
        return tracked

    def reset(self) -> None:
        """Forget every track, e.g. before re-running a video."""
        self._tracker.reset()
        self._create_filters()
        self._last_ids = []

    def get_tracker_ids(self) -> list[int]:
        """Tracker ids present in the latest frame."""
        return list(self._last_ids)

    def _create_filters(self) -> None:
        missing = max(self._config.lost_track_buffer, 1)
        self._kalman = KalmanBoxSmoother(
            self._config.min_hits, self._config.kalman_measurement_scale, missing, self._config.kalman_size_scale,
            self._config.coast_frames, self._config.stitch_frames,
        )
        self._average = sv.DetectionsSmoother(length=self._config.smoother_length)
        self._voter = ClassVoter(self._config.class_vote_window, max_missing=missing * 2)

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
