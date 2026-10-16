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


def _frame(index: int) -> np.ndarray:
    """A 300x2100 frame (the scripted car must not touch the border) whose first pixel encodes ``index``."""
    frame = np.zeros((2100, 300, 3), dtype=np.uint8)
    frame[0, 0, 0] = index
    return frame


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
        frame = _frame(index)
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


# ------------------------------------------------------------ region monitoring
def _region_monitor(config: AppConfig, max_dwell: float = 0.0) -> TrafficMonitor:
    from app.core.dwell_time_tracker import DwellTimeTracker
    from app.core.occupancy_counter import OccupancyCounter
    from app.core.roi_manager import ROIManager
    from app.models.roi import ROI

    # The scripted car moves down 10 px/frame from y=100; this zone covers y 200..400 (raw frames 10..30).
    zones = [ROI("zone_a", "Khu vực A", [[0, 200], [200, 200], [200, 400], [0, 400]], ["car"], max_dwell)]
    base = _monitor(config)
    return TrafficMonitor(
        base.detector, base._tracker, base.counter, base._speed, base._violations,
        roi_manager=ROIManager(zones), dwell_tracker=DwellTimeTracker(zones, grace_seconds=0.2),
        occupancy_counter=OccupancyCounter(["zone_a"]), fps=FPS,
    )


def test_region_entry_dwell_exit_through_monitor(config: AppConfig) -> None:
    from app.models.event import EventType

    monitor = _region_monitor(config)
    events = []
    occupancy = []
    for index in range(45):
        result = monitor.process_frame(_frame(index), index)
        events += result.region_events
        occupancy.append(result.statistics.regions[0].occupancy)
    assert [e.event_type for e in events] == [EventType.ENTER, EventType.EXIT]
    # The Kalman filter starts from the first box and lags the raw one by about 1 frame.
    assert events[0].timestamp == pytest.approx(11 / FPS, abs=1 / FPS)
    assert events[1].dwell_seconds == pytest.approx(20 / FPS, abs=1.5 / FPS)  # ±1 frame filter warm-up
    assert max(occupancy) == 1 and occupancy[-1] == 0
    region = result.statistics.regions[0]
    assert (region.entries, region.exits, region.peak_occupancy) == (1, 1, 1)
    assert region.average_dwell_seconds == pytest.approx(0.8, abs=1.5 / FPS)
    assert monitor.get_sessions()[0].status.value == "exited"


def test_region_overstay_through_monitor(config: AppConfig) -> None:
    monitor = _region_monitor(config, max_dwell=0.5)
    result = None
    for index in range(26):
        result = monitor.process_frame(_frame(index), index)
    region = result.statistics.regions[0]
    assert region.overstays == 1
    assert region.active_sessions[0].overstay
    assert result.statistics.overstay_alerts == 1
    monitor.reset()
    assert monitor.get_statistics().regions[0].entries == 0 and monitor.media_time == 0.0


def test_regions_disabled_in_config(config: AppConfig) -> None:
    config.regions.enabled = False
    monitor = TrafficMonitor.from_config(config, sv.VideoInfo(width=3840, height=2160, fps=25, total_frames=10), load_model=False)
    assert monitor.roi_manager is None and monitor.get_statistics().regions == ()
