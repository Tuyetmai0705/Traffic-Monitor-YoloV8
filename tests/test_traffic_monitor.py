"""Phase 8: TrafficMonitor orchestration with a scripted detector."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from app.config.settings import AppConfig
from app.core.counter import VehicleCounter
from app.core.detector import VehicleDetector
from app.core.exceptions import InferenceError
from app.core.speed_estimator import SpeedEstimator
from app.core.tracker import VehicleTracker
from app.core.traffic_monitor import TrafficMonitor
from app.core.violation_detector import SpeedViolationDetector
from app.config.settings import PerspectiveConfig
from tests.conftest import make_detections

FPS = 25.0


class ScriptedDetector(VehicleDetector):
    """Detector returning one car moving down 1 m per frame (10 px = 1 m)."""

    def detect(self, frame: np.ndarray) -> sv.Detections:
        y = 100 + int(frame[0, 0, 0]) * 10
        return make_detections([[90, y - 30, 110, y]], [2], confidences=[0.9])


def _monitor(config: AppConfig, limit: float = 80) -> TrafficMonitor:
    perspective = PerspectiveConfig([[0, 0], [200, 0], [200, 2000], [0, 2000]], 20, 200)
    return TrafficMonitor(
        detector=ScriptedDetector(config.model, config.classes),
        tracker=VehicleTracker(config.tracker, FPS),
        counter=VehicleCounter([0, 500], [200, 500]),
        speed_estimator=SpeedEstimator(perspective, config.speed, FPS),
        violation_detector=SpeedViolationDetector(limit),
    )


def _run(monitor: TrafficMonitor, frames: int):
    result = None
    for index in range(frames):
        frame = np.full((4, 4, 3), index, dtype=np.uint8)
        result = monitor.process_frame(frame, index)
    return result


def test_pipeline_counts_speeds_and_flags_violation(config: AppConfig) -> None:
    monitor = _monitor(config, limit=80)
    result = _run(monitor, 60)  # 1 m/frame at 25 fps = 90 km/h
    stats = result.statistics
    assert stats.total_out == 1 and stats.by_class["car"]["out"] == 1
    tid = int(result.detections.tracker_id[0])
    assert result.speeds[tid] == pytest.approx(90.0, abs=3)
    assert tid in result.violating_ids
    assert stats.violations_count == 1
    assert stats.violations[0].class_name == "car"
    assert stats.max_speed_kmh == pytest.approx(90.0, abs=3)


def test_no_violation_under_limit(config: AppConfig) -> None:
    result = _run(_monitor(config, limit=120), 60)
    assert result.statistics.violations_count == 0
    assert result.violating_ids == set()


def test_reset(config: AppConfig) -> None:
    monitor = _monitor(config)
    _run(monitor, 60)
    monitor.reset()
    stats = monitor.get_statistics()
    assert (stats.total_in, stats.total_out, stats.violations_count, stats.max_speed_kmh) == (0, 0, 0, 0.0)


def test_inference_error_carries_frame_index(config: AppConfig) -> None:
    monitor = _monitor(config)

    def fail(frame: np.ndarray) -> sv.Detections:
        raise InferenceError("boom")

    monitor.detector.detect = fail  # type: ignore[method-assign]
    with pytest.raises(InferenceError) as info:
        monitor.process_frame(np.zeros((4, 4, 3), np.uint8), 42)
    assert info.value.frame_index == 42
    assert "42" in str(info.value)


def test_from_config_builds_components_without_loading(config: AppConfig) -> None:
    monitor = TrafficMonitor.from_config(config, sv.VideoInfo(width=3840, height=2160, fps=25, total_frames=10), load_model=False)
    assert not monitor.detector.is_loaded
    assert monitor.speed_limit == 80
