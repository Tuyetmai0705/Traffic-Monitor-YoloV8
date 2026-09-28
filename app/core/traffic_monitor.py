"""Orchestrates detection, tracking, counting, speed and violations for each frame."""

from __future__ import annotations

import numpy as np
import supervision as sv

from app.config.settings import AppConfig
from app.core.counter import VehicleCounter
from app.core.detector import VehicleDetector
from app.core.exceptions import InferenceError
from app.core.speed_estimator import SpeedEstimator
from app.core.tracker import VehicleTracker
from app.core.violation_detector import SpeedViolationDetector
from app.models.detection_result import DetectionResult
from app.models.traffic_statistics import TrafficStatistics


class TrafficMonitor:
    """Thin coordinator. All algorithms live in the injected components."""

    def __init__(
        self,
        detector: VehicleDetector,
        tracker: VehicleTracker,
        counter: VehicleCounter,
        speed_estimator: SpeedEstimator,
        violation_detector: SpeedViolationDetector,
    ) -> None:
        """Wire the five core components."""
        self._detector = detector
        self._tracker = tracker
        self._counter = counter
        self._speed = speed_estimator
        self._violations = violation_detector

    @classmethod
    def from_config(cls, config: AppConfig, video_info: sv.VideoInfo, load_model: bool = True) -> TrafficMonitor:
        """Build every component from configuration and the video's properties."""
        detector = VehicleDetector(config.model, config.classes)
        if load_model:
            detector.load_model()
        return cls(
            detector=detector,
            tracker=VehicleTracker(config.tracker, frame_rate=video_info.fps),
            counter=VehicleCounter(config.counting_line.start, config.counting_line.end),
            speed_estimator=SpeedEstimator(config.perspective, config.speed, fps=video_info.fps),
            violation_detector=SpeedViolationDetector(config.speed.limit_kmh),
        )

    @property
    def detector(self) -> VehicleDetector:
        """Detector, exposed for device and class-name lookups."""
        return self._detector

    @property
    def counter(self) -> VehicleCounter:
        """Counter, exposed for the line-zone annotator."""
        return self._counter

    @property
    def speed_limit(self) -> float:
        """Configured speed limit in km/h."""
        return self._violations.speed_limit

    def process_frame(self, frame: np.ndarray, frame_index: int) -> DetectionResult:
        """Run detect → track → count → speed → violation on one frame."""
        try:
            detections = self._detector.detect(frame)
        except InferenceError as exc:
            exc.frame_index = frame_index
            raise
        detections = self._tracker.update(detections)
        self._counter.count(detections)

        anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        self._speed.update_track(detections.tracker_id, self._speed.transform_points(anchors), frame_index)

        speeds: dict[int, float | None] = {}
        violating: set[int] = set()
        for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
            tid = int(tracker_id)
            speeds[tid] = self._speed.get_speed(tid)
            if self._violations.check(tid, speeds[tid], self._detector.get_class_name(class_id), frame_index):
                violating.add(tid)
        return DetectionResult(frame_index, detections, speeds, violating, self.get_statistics())

    def get_statistics(self) -> TrafficStatistics:
        """Aggregate counters, violations and speed summary."""
        violations = sorted(self._violations.get_violations().values(), key=lambda v: v.first_frame)
        average, maximum = self._speed.get_speed_summary()
        return TrafficStatistics(
            total_in=self._counter.get_total_in(),
            total_out=self._counter.get_total_out(),
            by_class=self._counter.get_class_counts(),
            violations_count=len(violations),
            violations=tuple(violations),
            average_speed_kmh=average,
            max_speed_kmh=maximum,
            active_tracks=len(self._tracker.get_tracker_ids()),
        )

    def reset(self) -> None:
        """Reset tracking, counting, speed history and violations."""
        self._tracker.reset()
        self._counter.reset()
        self._speed.reset()
        self._violations.reset()
