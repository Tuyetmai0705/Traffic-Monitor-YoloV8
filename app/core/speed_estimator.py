"""Speed estimation using a perspective transform and per-track history."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from app.config.settings import PerspectiveConfig, SpeedConfig
from app.core.exceptions import InvalidPerspectiveError
from app.utils.geometry import validate_quadrilateral

_MS_TO_KMH = 3.6
_MIN_OUTLIER_M = 0.5


class SpeedEstimator:
    """Convert ground points to metres and estimate km/h over a time window.

    Accuracy rules:

    * Only points inside the calibrated A-B-C-D area are used. Outside it the
      homography extrapolates and one pixel can be several metres.
    * The ground point is the bottom-centre of the box: the homography maps the
      road plane, and the box centre lies ~1 m above the road, which would bias
      far vehicles.
    * Velocity is the least-squares slope of position over the whole window
      (every sample counts), not the distance between the first and last sample.
    * A window is only trusted when the vehicle moved at least
      ``min_pixel_travel`` pixels on screen (far vehicles move a few pixels
      while their box jitters by 2-3 px) and the first and second halves of
      the window agree within ``max_half_window_difference``. Otherwise no new
      speed is reported: no number is better than a wrong one.
    * Only samples where one pixel covers at most ``max_metres_per_pixel`` of
      road are used: near the horizon one pixel can be 2 m, so 2 px of box
      jitter would be a 4 m position error.
    * Estimates above ``max_kmh`` are rejected as tracking glitches; the
      displayed value is an exponential moving average.
    * Time comes from frame indices, so missed frames do not distort the speed.
    """

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
        self._area = source.reshape(-1, 1, 2)
        self._fps = fps if fps > 0 else 30.0
        self._window = max(2, round(self._fps * speed.window_seconds))
        self._min_samples = max(2, min(speed.min_samples, self._window))
        self._stale_frames = max(1, round(self._fps * speed.stale_seconds))
        self._max_kmh = speed.max_kmh
        self._alpha = speed.smoothing_alpha
        self._min_travel = speed.min_pixel_travel
        self._max_mpp = speed.max_metres_per_pixel
        self._max_half_difference = speed.max_half_window_difference
        self._tracks: dict[int, deque[tuple[int, float, float]]] = {}
        self._pixels: dict[int, deque[tuple[float, float]]] = {}
        self._last_seen: dict[int, int] = {}
        self._speeds: dict[int, float] = {}
        self._fresh: set[int] = set()
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
        """Map (N, 2) pixel points to (N, 2) road coordinates in metres; NaN points stay NaN.

        ``cv2.perspectiveTransform`` silently maps NaN to (0, 0), which would look
        like a jump to the start of the road, so missing points are masked here.
        """
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        world = cv2.perspectiveTransform(points.reshape(-1, 1, 2), self._matrix).reshape(-1, 2)
        world[np.isnan(points).any(axis=1)] = np.nan
        return world

    def in_calibrated_area(self, pixel_points: np.ndarray) -> np.ndarray:
        """Boolean mask of pixel points inside (or on) the A-B-C-D quadrilateral."""
        points = np.asarray(pixel_points, dtype=np.float32).reshape(-1, 2)
        return np.array([cv2.pointPolygonTest(self._area, (float(x), float(y)), False) >= 0 for x, y in points], dtype=bool)

    def metres_per_pixel(self, pixel_points: np.ndarray) -> np.ndarray:
        """Road distance covered by one image pixel (vertical step) at each pixel point."""
        points = np.asarray(pixel_points, dtype=np.float32).reshape(-1, 2)
        here = self.transform_points(points)
        below = self.transform_points(points + np.array([0, 1], dtype=np.float32))
        return np.hypot(*(below - here).T)

    def update_track(
        self,
        tracker_ids: np.ndarray,
        points: np.ndarray,
        frame_index: int,
        valid: np.ndarray | None = None,
        pixels: np.ndarray | None = None,
    ) -> None:
        """Append metre positions per tracker id, refresh speeds and drop stale tracks.

        ``valid`` marks samples that may be used (inside the calibrated area, box
        not cut by the frame edge). Invalid samples are skipped but keep the
        track alive, so its last speed stays visible. ``pixels`` are the same
        anchors in frame pixels, used for the minimum-travel check.
        """
        self._fresh.clear()
        valid = np.ones(len(tracker_ids), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        pixels = np.full((len(tracker_ids), 2), np.nan) if pixels is None else np.asarray(pixels, dtype=float)
        known = ~np.isnan(pixels).any(axis=1)
        if known.any():                                   # resolution gate (pixels unknown: no gate)
            fine = np.ones(len(tracker_ids), dtype=bool)
            fine[known] = self.metres_per_pixel(pixels[known]) <= self._max_mpp
            valid = valid & fine
        for tracker_id, (x, y), usable, (px, py) in zip(tracker_ids, points, valid, pixels):
            tid = int(tracker_id)
            self._last_seen[tid] = frame_index
            if not usable:
                continue
            history = self._tracks.setdefault(tid, deque(maxlen=self._window))
            history.append((frame_index, float(x), float(y)))
            self._pixels.setdefault(tid, deque(maxlen=self._window)).append((float(px), float(py)))
            speed = self._reliable_speed(tid)
            if speed is None or speed > self._max_kmh:
                continue
            previous = self._speeds.get(tid)
            smoothed = speed if previous is None else (1 - self._alpha) * previous + self._alpha * speed
            self._speeds[tid] = smoothed
            self._fresh.add(tid)
            self._session_speeds[tid] = smoothed
            self._max_speed = max(self._max_speed, smoothed)
        self._drop_stale(frame_index)

    def calculate_speed(self, tracker_id: int) -> float | None:
        """Least-squares speed in km/h over the window, or None with too few samples."""
        history = self._tracks.get(int(tracker_id))
        if history is None or len(history) < self._min_samples:
            return None
        return self._fit(np.asarray(history, dtype=np.float64))

    def _reliable_speed(self, tracker_id: int) -> float | None:
        """Whole-window speed, or None when the window is not trustworthy.

        Requires enough samples, enough on-screen travel, and agreement between
        the first half, the second half and the whole window (a box jump exactly
        at the middle leaves each half smooth but bends the whole-window line).
        """
        history = self._tracks[tracker_id]
        if len(history) < self._min_samples:
            return None
        pixels = self._pixels.get(tracker_id)
        if pixels and len(pixels) >= 2:
            (x0, y0), (x1, y1) = pixels[0], pixels[-1]
            if not np.isnan(x0) and float(np.hypot(x1 - x0, y1 - y0)) < self._min_travel:
                return None
        samples = np.asarray(history, dtype=np.float64)
        whole = self._fit(samples)
        half = len(samples) // 2
        if whole is None or half < 3:
            return whole
        first, second = self._fit(samples[:half]), self._fit(samples[half:])
        if first is None or second is None:
            return whole
        speeds = (first, second, whole)
        if max(speeds) - min(speeds) > self._max_half_difference * max(max(speeds), 1e-6):
            return None
        return whole

    def _fit(self, samples: np.ndarray) -> float | None:
        """Least-squares speed (km/h) of (frame, x, y) samples, refitted without outliers.

        Samples farther than ``max(0.5 m, 3 × robust sigma)`` from the first fit
        (a few broken boxes) are dropped and the line is fitted again.
        """
        fit = self._line(samples)
        if fit is None:
            return None
        velocity, residuals = fit
        sigma = 1.4826 * float(np.median(residuals))
        inliers = residuals <= max(_MIN_OUTLIER_M, 3.0 * sigma)
        if not inliers.all() and inliers.sum() >= max(3, self._min_samples // 2):
            refit = self._line(samples[inliers])
            if refit is not None:
                velocity = refit[0]
        return velocity * _MS_TO_KMH

    def _line(self, samples: np.ndarray) -> tuple[float, np.ndarray] | None:
        """Return (speed in m/s, distance of each sample from the fitted line)."""
        seconds = (samples[:, 0] - samples[0, 0]) / self._fps
        centred = seconds - seconds.mean()
        denominator = float(centred @ centred)
        if denominator <= 0:
            return None
        mean_x, mean_y = samples[:, 1].mean(), samples[:, 2].mean()
        vx = float(centred @ (samples[:, 1] - mean_x)) / denominator
        vy = float(centred @ (samples[:, 2] - mean_y)) / denominator
        residuals = np.hypot(samples[:, 1] - (mean_x + vx * centred), samples[:, 2] - (mean_y + vy * centred))
        return float(np.hypot(vx, vy)), residuals

    def reset_track(self, tracker_id: int) -> None:
        """Forget the position history of one track (its displayed speed is kept).

        Called when the box suddenly changes shape (occlusion starts or ends): the
        bottom of the box is then no longer the same point of the vehicle.
        """
        self._tracks.pop(int(tracker_id), None)
        self._pixels.pop(int(tracker_id), None)

    def get_speed(self, tracker_id: int) -> float | None:
        """Latest smoothed speed for a track (kept when it leaves the calibrated area)."""
        return self._speeds.get(int(tracker_id))

    def is_fresh(self, tracker_id: int) -> bool:
        """True when the track's speed was re-estimated in the latest frame."""
        return int(tracker_id) in self._fresh

    def get_speed_summary(self) -> tuple[float, float]:
        """Return (average of each vehicle's latest speed, maximum speed) for the session."""
        if not self._session_speeds:
            return 0.0, 0.0
        values = list(self._session_speeds.values())
        return sum(values) / len(values), self._max_speed

    def reset(self) -> None:
        """Forget all histories and session metrics."""
        self._tracks.clear()
        self._pixels.clear()
        self._last_seen.clear()
        self._speeds.clear()
        self._fresh.clear()
        self._session_speeds.clear()
        self._max_speed = 0.0

    def _drop_stale(self, frame_index: int) -> None:
        stale = [tid for tid, seen in self._last_seen.items() if frame_index - seen > self._stale_frames]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._pixels.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._speeds.pop(tid, None)
