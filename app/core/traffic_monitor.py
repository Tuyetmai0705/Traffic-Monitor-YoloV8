"""Orchestrates detection, tracking, counting, speed and violations for each frame."""

from __future__ import annotations

import numpy as np
import supervision as sv

from app.config.settings import AppConfig
from app.core.counter import VehicleCounter
from app.core.detector import VehicleDetector
from app.core.dwell_time_tracker import DwellTimeTracker
from app.core.exceptions import InferenceError
from app.core.occupancy_counter import OccupancyCounter
from app.core.roi_manager import ROIManager
from app.core.speed_estimator import SpeedEstimator
from app.core.tracker import VehicleTracker
from app.core.violation_detector import SpeedViolationDetector
from app.models.detection_result import DetectionResult
from app.models.event import RegionEvent
from app.models.region_statistics import RegionStatistics
from app.models.track_session import TrackSession
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
        roi_manager: ROIManager | None = None,
        dwell_tracker: DwellTimeTracker | None = None,
        occupancy_counter: OccupancyCounter | None = None,
        fps: float = 25.0,
    ) -> None:
        """Wire the core components. The three region components are optional and go together."""
        self._detector = detector
        self._tracker = tracker
        self._counter = counter
        self._speed = speed_estimator
        self._violations = violation_detector
        self._roi = roi_manager
        self._dwell = dwell_tracker
        self._occupancy = occupancy_counter
        self._fps = fps if fps > 0 else 25.0
        self._media_time = 0.0

    @classmethod
    def from_config(cls, config: AppConfig, video_info: sv.VideoInfo, load_model: bool = True) -> TrafficMonitor:
        """Build every component from configuration and the video's properties."""
        detector = VehicleDetector(config.model, config.classes)
        if load_model:
            detector.load_model()
        regions = config.regions
        zones = regions.zones if regions.enabled else []
        return cls(
            detector=detector,
            tracker=VehicleTracker(config.tracker, frame_rate=video_info.fps),
            counter=VehicleCounter(config.counting_line.start, config.counting_line.end),
            speed_estimator=SpeedEstimator(config.perspective, config.speed, fps=video_info.fps),
            violation_detector=SpeedViolationDetector(config.speed.limit_kmh),
            roi_manager=ROIManager(zones, regions.anchor) if zones else None,
            dwell_tracker=DwellTimeTracker(zones, regions.grace_seconds) if zones else None,
            occupancy_counter=OccupancyCounter(zone.roi_id for zone in zones) if zones else None,
            fps=video_info.fps,
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
    def roi_manager(self) -> ROIManager | None:
        """Region manager, or None when region monitoring is off."""
        return self._roi

    @property
    def dwell_tracker(self) -> DwellTimeTracker | None:
        """Dwell-time tracker, or None when region monitoring is off."""
        return self._dwell

    @property
    def media_time(self) -> float:
        """Media time of the last processed frame in seconds."""
        return self._media_time

    @property
    def speed_limit(self) -> float:
        """Configured speed limit in km/h."""
        return self._violations.speed_limit

    def process_frame(self, frame: np.ndarray, frame_index: int) -> DetectionResult:
        """Run detect → track → count → speed → violation → regions on one frame."""
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
        self._media_time = frame_index / self._fps
        events = self._update_regions(detections)
        return DetectionResult(frame_index, detections, speeds, violating, self.get_statistics(), events)

    def _update_regions(self, detections: sv.Detections) -> list[RegionEvent]:
        """ROI membership → dwell sessions → occupancy. Time is media time, so it stops when paused."""
        if self._roi is None or self._dwell is None or self._occupancy is None:
            return []
        names = [self._detector.get_class_name(class_id) for class_id in detections.class_id]
        memberships = self._roi.evaluate(detections, names)
        events = self._dwell.update(memberships, detections.tracker_id, names, self._media_time)
        self._occupancy.update(self._dwell.get_open_track_ids())
        return events

    def get_sessions(self) -> list[TrackSession]:
        """Every region session of the run (empty when region monitoring is off)."""
        return self._dwell.get_sessions() if self._dwell is not None else []

    def _region_statistics(self) -> tuple[RegionStatistics, ...]:
        if self._roi is None or self._dwell is None or self._occupancy is None:
            return ()
        result = []
        for zone in self._roi.zones:
            counts = self._dwell.get_counts(zone.roi_id)
            closed = [s.dwell_seconds for s in self._dwell.get_closed_sessions(zone.roi_id)]
            active = tuple(self._dwell.get_active_sessions(zone.roi_id))
            result.append(RegionStatistics(
                roi_id=zone.roi_id,
                name=zone.name,
                occupancy=self._occupancy.get_occupancy(zone.roi_id),
                peak_occupancy=self._occupancy.get_peak(zone.roi_id),
                average_occupancy=self._occupancy.get_average(zone.roi_id),
                entries=counts["entries"],
                exits=counts["exits"],
                lost=counts["lost"],
                overstays=counts["overstays"],
                average_dwell_seconds=sum(closed) / len(closed) if closed else 0.0,
                max_dwell_seconds=max(closed + [s.dwell_seconds for s in active], default=0.0),
                dwell_limit_seconds=zone.max_dwell_seconds,
                active_sessions=active,
            ))
        return tuple(result)

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
            media_time=self._media_time,
            regions=self._region_statistics(),
        )

    def reset(self) -> None:
        """Reset tracking, counting, speed history, violations and region sessions."""
        self._tracker.reset()
        self._counter.reset()
        self._speed.reset()
        self._violations.reset()
        if self._dwell is not None and self._occupancy is not None:
            self._dwell.reset()
            self._occupancy.reset()
        self._media_time = 0.0
