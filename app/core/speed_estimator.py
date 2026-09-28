"""Speed estimation using a perspective transform and per-track history."""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from app.config.settings import PerspectiveConfig, SpeedConfig
from app.core.exceptions import InvalidPerspectiveError
from app.utils.geometry import validate_quadrilateral

_MS_TO_KMH = 3.6


class SpeedEstimator:
    """Convert pixel anchors to metres and estimate km/h over a time window."""

    def __init__(self, perspective: PerspectiveConfig, speed: SpeedConfig, fps: float) -> None:
        """Build the pixel→metre homography. Raises InvalidPerspectiveError."""
        try:
            source = validate_quadrilateral(perspective.source_points)
        except ValueError as exc:
            raise InvalidPerspectiveError(str(exc)) from exc
        if perspective.road_width_m <= 0 or perspective.road_length_m <= 0:
            raise InvalidPerspectiveError("kích thước đường phải > 0")
        width, length = perspective.road_width_m, perspective.road_length_m
        target = np.array([[0, 0], [width, 0], [width, length], [0, length]], dtype=np.float32)
        self._matrix = cv2.getPerspectiveTransform(source, target)
        self._fps = fps if fps > 0 else 30.0
        self._window = max(2, round(self._fps * speed.window_seconds))
        self._min_samples = max(2, min(speed.min_samples, self._window))
        self._stale_frames = max(1, round(self._fps * speed.stale_seconds))
        self._tracks: dict[int, deque[tuple[int, float, float]]] = {}
        self._last_seen: dict[int, int] = {}
        self._speeds: dict[int, float] = {}
        self._session_speeds: dict[int, float] = {}
        self._max_speed = 0.0

    @property
    def window_size(self) -> int:
        """Number of samples kept per track (``fps × window_seconds``)."""
        return self._window

    @property
    def min_samples(self) -> int:
        """Samples required before a speed is reported."""
        return self._min_samples

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Map (N, 2) pixel points to (N, 2) road coordinates in metres."""
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2), self._matrix).reshape(-1, 2)

    def update_track(self, tracker_ids: np.ndarray, points: np.ndarray, frame_index: int) -> None:
        """Append metre positions per tracker id, refresh speeds and drop stale tracks."""
        for tracker_id, (x, y) in zip(tracker_ids, points):
            tid = int(tracker_id)
            history = self._tracks.setdefault(tid, deque(maxlen=self._window))
            history.append((frame_index, float(x), float(y)))
            self._last_seen[tid] = frame_index
            speed = self.calculate_speed(tid)
            if speed is not None:
                self._speeds[tid] = speed
                self._session_speeds[tid] = speed
                self._max_speed = max(self._max_speed, speed)
        self._drop_stale(frame_index)

    def calculate_speed(self, tracker_id: int) -> float | None:
        """Speed in km/h between the oldest and newest sample, or None if too few samples."""
        history = self._tracks.get(int(tracker_id))
        if history is None or len(history) < self._min_samples:
            return None
        first_frame, x0, y0 = history[0]
        last_frame, x1, y1 = history[-1]
        elapsed = (last_frame - first_frame) / self._fps
        if elapsed <= 0:
            return None
        return math.hypot(x1 - x0, y1 - y0) / elapsed * _MS_TO_KMH

    def get_speed(self, tracker_id: int) -> float | None:
        """Latest computed speed for a track."""
        return self._speeds.get(int(tracker_id))

    def get_speed_summary(self) -> tuple[float, float]:
        """Return (average of each vehicle's latest speed, maximum speed) for the session."""
        if not self._session_speeds:
            return 0.0, 0.0
        values = list(self._session_speeds.values())
        return sum(values) / len(values), self._max_speed

    def reset(self) -> None:
        """Forget all histories and session metrics."""
        self._tracks.clear()
        self._last_seen.clear()
        self._speeds.clear()
        self._session_speeds.clear()
        self._max_speed = 0.0

    def _drop_stale(self, frame_index: int) -> None:
        stale = [tid for tid, seen in self._last_seen.items() if frame_index - seen > self._stale_frames]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._speeds.pop(tid, None)
