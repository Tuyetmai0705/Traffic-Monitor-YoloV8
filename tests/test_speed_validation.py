"""Speed estimates on the real sample video agree with an independent section (point-to-point) speed.

The section speed only uses the time a vehicle needs between two virtual lines a
known distance apart, like average-speed enforcement cameras. It shares the
calibration with the app, so this checks the estimator (regression, gating,
smoothing), not the absolute scale of the calibration.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from app.config.settings import AppConfig
from app.core.box_smoother import RAW_BOXES_KEY
from app.core.traffic_monitor import TrafficMonitor
from app.pipeline.frame_processor import FrameProcessor
from app.pipeline.video_processor import VideoProcessor
from tests.conftest import SAMPLE_VIDEO, requires_gpu, requires_video

# After the overhead gantry, which occludes vehicles around 60 m.
SECTION_START_M, SECTION_END_M = 150.0, 245.0


def _crossing_frame(rows: list[tuple[int, float]], target: float) -> float | None:
    for (f0, y0), (f1, y1) in zip(rows, rows[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1 and f1 - f0 <= 3:
            return f0 + (target - y0) / (y1 - y0) * (f1 - f0)
    return None


@pytest.mark.gpu
@requires_gpu
@requires_video
def test_app_speed_matches_section_speed(config: AppConfig, tmp_path) -> None:
    processor = VideoProcessor(max_width=1920)
    info = processor.open(SAMPLE_VIDEO)
    run_config = config.scaled_geometry(processor.scale)
    monitor = TrafficMonitor.from_config(run_config, info)
    frame_processor = FrameProcessor(monitor, run_config, info)
    positions: dict[int, list[tuple[int, float]]] = {}
    estimates: dict[int, list[tuple[int, float]]] = {}
    process_frame = monitor.process_frame

    def recording(frame: np.ndarray, index: int):
        result = process_frame(frame, index)
        detections = result.detections
        anchors = sv.Detections(xyxy=detections.data.get(RAW_BOXES_KEY, detections.xyxy)).get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER)
        world = monitor._speed.transform_points(anchors)
        for tracker_id, (_, y) in zip(detections.tracker_id, world):
            tid = int(tracker_id)
            positions.setdefault(tid, []).append((index, float(y)))
            if monitor._speed.is_fresh(tid):
                estimates.setdefault(tid, []).append((index, result.speeds[tid]))
        return result

    monitor.process_frame = recording  # type: ignore[method-assign]
    processor.process(frame_processor, lambda *args: None, tmp_path / "out.mp4")

    differences = []
    for tid, rows in positions.items():
        start, end = _crossing_frame(rows, SECTION_START_M), _crossing_frame(rows, SECTION_END_M)
        if start is None or end is None:
            continue
        section_kmh = (SECTION_END_M - SECTION_START_M) / (abs(end - start) / info.fps) * 3.6
        low, high = sorted((start, end))
        app = [speed for frame, speed in estimates.get(tid, []) if low <= frame <= high]
        if len(app) >= 5:
            differences.append((np.mean(app) - section_kmh) / section_kmh)
    # No single estimate may jump far from the vehicle's own typical speed (catches occlusion spikes).
    for tid, values in estimates.items():
        speeds = np.array([speed for _, speed in values])
        if len(speeds) >= 10:
            worst = float(np.max(np.abs(speeds - np.median(speeds)) / np.median(speeds)))
            assert worst < 0.15, f"track #{tid}: estimate {worst:.0%} away from its median"
    assert len(differences) >= 3, "the sample video has at least 3 vehicles crossing the section"
    assert max(abs(d) for d in differences) < 0.04, f"differences: {[round(d * 100, 1) for d in differences]} %"
    assert abs(float(np.mean(differences))) < 0.02
