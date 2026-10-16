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
SPEED = SpeedConfig(limit_kmh=80, window_seconds=1.0, min_samples=15, stale_seconds=2.0,
                    max_kmh=250, smoothing_alpha=1.0, violation_confirm_samples=1,  # alpha 1 = raw estimate
                    min_pixel_travel=0, max_half_window_difference=0.25,
                    max_metres_per_pixel=10)


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


# ------------------------------------------------------------ accuracy rules
def test_regression_is_robust_to_pixel_noise() -> None:
    """20 m/s with ±3 px (±0.3 m) jitter: regression error stays small, endpoints would not."""
    rng = np.random.default_rng(1)
    estimator = _estimator()
    for frame in range(60):
        y = 100 + frame * (20 / FPS) * 10 + rng.normal(0, 3)
        estimator.update_track(np.array([1]), estimator.transform_points(np.array([[100, y]])), frame)
    assert estimator.calculate_speed(1) == pytest.approx(72.0, rel=0.04)


def test_samples_outside_calibrated_area_are_ignored() -> None:
    estimator = _estimator()
    pixels = np.array([[100, 500], [100, 1500], [-50, 500]])
    assert estimator.in_calibrated_area(pixels).tolist() == [True, False, False]
    for frame in range(40):
        estimator.update_track(np.array([2]), np.array([[0.0, frame * 1.0]]), frame, valid=np.array([False]))
    assert estimator.get_speed(2) is None


def test_last_speed_kept_after_leaving_area() -> None:
    estimator = _estimator()
    for frame in range(30):
        estimator.update_track(np.array([3]), np.array([[0.0, frame * (20 / FPS)]]), frame)
    assert estimator.is_fresh(3)
    estimator.update_track(np.array([3]), np.array([[0.0, 99.0]]), 30, valid=np.array([False]))
    assert not estimator.is_fresh(3)
    assert estimator.get_speed(3) == pytest.approx(72.0, abs=0.5)


def test_implausible_speed_rejected() -> None:
    estimator = _estimator()
    for frame in range(30):          # 100 m per frame = 10 800 km/h: a tracking glitch
        estimator.update_track(np.array([4]), np.array([[0.0, frame * 100.0]]), frame)
    assert estimator.get_speed(4) is None


def test_display_speed_is_smoothed() -> None:
    config = SpeedConfig(limit_kmh=80, window_seconds=1.0, min_samples=15, stale_seconds=2.0,
                         max_kmh=250, smoothing_alpha=0.3, violation_confirm_samples=1,
                         min_pixel_travel=0, max_half_window_difference=1.0,
                         max_metres_per_pixel=10)
    estimator = SpeedEstimator(PERSPECTIVE, config, fps=FPS)
    for frame in range(30):
        estimator.update_track(np.array([5]), np.array([[0.0, frame * (20 / FPS)]]), frame)
    position = 30 * (20 / FPS)
    for frame in range(30, 45):      # the car accelerates to 30 m/s (108 km/h)
        position += 30 / FPS
        estimator.update_track(np.array([5]), np.array([[0.0, position]]), frame)
    raw = estimator.calculate_speed(5)
    assert 72.0 < estimator.get_speed(5) < raw      # the displayed value follows with a short lag


def _gated() -> SpeedEstimator:
    config = SpeedConfig(limit_kmh=80, window_seconds=1.0, min_samples=15, stale_seconds=2.0, max_kmh=250,
                         smoothing_alpha=1.0, violation_confirm_samples=1, min_pixel_travel=20, max_half_window_difference=0.25,
                         max_metres_per_pixel=10)
    return SpeedEstimator(PERSPECTIVE, config, fps=FPS)


def test_too_little_screen_travel_gives_no_speed() -> None:
    """A far vehicle moving 0.5 px/frame (15 px per window) is not measured."""
    estimator = _gated()
    for frame in range(40):
        pixel = np.array([[100, 100 + frame * 0.5]])
        estimator.update_track(np.array([1]), estimator.transform_points(pixel), frame, pixels=pixel)
    assert estimator.get_speed(1) is None


def test_inconsistent_window_is_rejected() -> None:
    """Box jumps 30 px (occlusion): no window containing the jump is accepted; the last good speed stays."""
    estimator = _gated()
    for frame in range(30):
        y = 100 + frame * 2 + (30 if frame >= 15 else 0)   # 2 px/frame = 6 m/s = 21.6 km/h
        pixel = np.array([[100, y]])
        estimator.update_track(np.array([2]), estimator.transform_points(pixel), frame, pixels=pixel)
        if frame >= 15:
            assert not estimator.is_fresh(2), f"window with the jump accepted at frame {frame}"
    assert estimator.get_speed(2) == pytest.approx(21.6, abs=0.1)


def test_steady_vehicle_passes_reliability_checks() -> None:
    estimator = _gated()
    for frame in range(40):
        pixel = np.array([[100, 100 + frame * (20 / FPS) * 10]])  # 6.7 px/frame
        estimator.update_track(np.array([3]), estimator.transform_points(pixel), frame, pixels=pixel)
    assert estimator.get_speed(3) == pytest.approx(72.0, abs=0.5)


def test_transform_keeps_missing_points_missing() -> None:
    """cv2 maps NaN to (0, 0); predicted (coasting) boxes have NaN anchors and must stay unknown."""
    world = _estimator().transform_points(np.array([[100, 500], [np.nan, np.nan]]))
    assert world[0] == pytest.approx([10, 50], abs=1e-4)
    assert np.isnan(world[1]).all()


def test_far_samples_with_coarse_resolution_are_ignored() -> None:
    """Pixels that cover more road than max_metres_per_pixel are not used for speed."""
    far = PerspectiveConfig(source_points=[[0, 0], [200, 0], [200, 100], [0, 100]], road_width_m=20, road_length_m=200)
    config = SpeedConfig(limit_kmh=80, window_seconds=1.0, min_samples=15, stale_seconds=2.0, max_kmh=250,
                         smoothing_alpha=1.0, violation_confirm_samples=1, min_pixel_travel=0,
                         max_half_window_difference=0.25, max_metres_per_pixel=0.8)
    estimator = SpeedEstimator(far, config, fps=FPS)              # 2 m per pixel everywhere
    assert estimator.metres_per_pixel(np.array([[100, 50]]))[0] == pytest.approx(2.0, rel=1e-3)
    for frame in range(40):
        pixel = np.array([[100, 10 + frame * 0.5]])
        estimator.update_track(np.array([1]), estimator.transform_points(pixel), frame, pixels=pixel)
    assert estimator.get_speed(1) is None
