"""Phase 6: SpeedEstimator transform, windowed speed and validation."""

from __future__ import annotations

import numpy as np
import pytest

from app.config.settings import PerspectiveConfig, SpeedConfig
from app.core.exceptions import InvalidPerspectiveError
from app.core.speed_estimator import SpeedEstimator

FPS = 30.0
# 10 px = 1 m: a 200 x 1000 px rectangle mapped to 20 x 100 m.
PERSPECTIVE = PerspectiveConfig(source_points=[[0, 0], [200, 0], [200, 1000], [0, 1000]], road_width_m=20, road_length_m=100)
SPEED = SpeedConfig(limit_kmh=80, window_seconds=1.0, min_samples=15, stale_seconds=2.0)


def _estimator() -> SpeedEstimator:
    return SpeedEstimator(PERSPECTIVE, SPEED, fps=FPS)


def test_transform_maps_corners_to_metres() -> None:
    estimator = _estimator()
    world = estimator.transform_points(np.array([[0, 0], [200, 0], [200, 1000], [0, 1000], [100, 500]]))
    assert world == pytest.approx(np.array([[0, 0], [20, 0], [20, 100], [0, 100], [10, 50]]), abs=1e-4)


def test_transform_empty() -> None:
    assert _estimator().transform_points(np.empty((0, 2))).shape == (0, 2)


def test_constant_20_mps_gives_72_kmh() -> None:
    estimator = _estimator()
    for frame in range(40):
        y_pixels = 100 + frame * (20 / FPS) * 10  # 20 m/s → pixels per frame
        world = estimator.transform_points(np.array([[100, y_pixels]]))
        estimator.update_track(np.array([7]), world, frame)
    assert estimator.get_speed(7) == pytest.approx(72.0, abs=0.5)
    average, maximum = estimator.get_speed_summary()
    assert average == pytest.approx(72.0, abs=0.5)
    assert maximum == pytest.approx(72.0, abs=0.5)


def test_window_uses_deque_not_two_frames() -> None:
    estimator = _estimator()
    assert estimator.window_size == 30
    for frame in range(40):
        estimator.update_track(np.array([1]), np.array([[0.0, frame * 1.0]]), frame)
    assert len(estimator._tracks[1]) == 30


def test_not_enough_samples_returns_none() -> None:
    estimator = _estimator()
    for frame in range(estimator.min_samples - 1):
        estimator.update_track(np.array([3]), np.array([[0.0, float(frame)]]), frame)
    assert estimator.calculate_speed(3) is None
    assert estimator.get_speed(3) is None


def test_speed_survives_missed_frames() -> None:
    estimator = _estimator()
    for frame in range(0, 60, 2):  # tracked every other frame, 10 m/s
        estimator.update_track(np.array([5]), np.array([[0.0, frame * 10 / FPS]]), frame)
    assert estimator.get_speed(5) == pytest.approx(36.0, abs=0.5)


def test_stale_tracks_are_dropped() -> None:
    estimator = _estimator()
    estimator.update_track(np.array([9]), np.array([[0.0, 0.0]]), 0)
    estimator.update_track(np.array([1]), np.array([[0.0, 0.0]]), 200)
    assert 9 not in estimator._tracks


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [100, 0], [200, 0], [0, 100]],        # collinear
        [[0, 0], [100, 100], [100, 0], [0, 100]],      # self-intersecting
        [[0, 0], [100, 0], [100, 100]],                # 3 points
    ],
)
def test_invalid_perspective_raises(points: list[list[float]]) -> None:
    with pytest.raises(InvalidPerspectiveError):
        SpeedEstimator(PerspectiveConfig(points, 10, 10), SPEED, fps=FPS)


def test_reset() -> None:
    estimator = _estimator()
    for frame in range(20):
        estimator.update_track(np.array([1]), np.array([[0.0, frame * 1.0]]), frame)
    estimator.reset()
    assert estimator.get_speed(1) is None
    assert estimator.get_speed_summary() == (0.0, 0.0)
