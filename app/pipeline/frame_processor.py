"""Run the core monitor on a frame and draw all annotations."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import supervision as sv

from app.config.settings import AppConfig
from app.constants.classes import CLASS_COLORS, LINE_COLOR, OVERSTAY_COLOR, VIOLATION_COLOR
from app.core.traffic_monitor import TrafficMonitor
from app.models.detection_result import DetectionResult
from app.models.traffic_statistics import TrafficStatistics
from app.services.event_service import EventService
from app.utils.display import (
    TextStyle,
    build_overlay_lines,
    draw_box_labels,
    draw_line_counts,
    draw_regions,
    draw_text_panel,
    hex_to_bgr,
)
from app.utils.time_utils import frame_to_timestamp

_EMA = 0.1
_RECENT_EVENTS = 300
_WHITE = (255, 255, 255)


def _class_palette() -> sv.ColorPalette:
    """Palette indexed by COCO class id so ColorLookup.CLASS picks the class colour."""
    size = max(CLASS_COLORS) + 1
    colors = [sv.Color.from_hex(CLASS_COLORS.get(index, "#FFFFFF")) for index in range(size)]
    return sv.ColorPalette(colors)


class FrameProcessor:
    """Frame → TrafficMonitor → annotated frame + statistics."""

    def __init__(
        self,
        monitor: TrafficMonitor,
        config: AppConfig,
        video_info: sv.VideoInfo,
        event_service: EventService | None = None,
    ) -> None:
        """Create annotators and the text style scaled to the video resolution."""
        self._monitor = monitor
        self._events = event_service or EventService()
        self._fps = video_info.fps
        thickness = max(1, sv.calculate_optimal_line_thickness(video_info.resolution_wh) - 1)
        palette = _class_palette()
        self._style = TextStyle.for_frame(config.annotation, video_info.height)
        self._box = sv.BoxAnnotator(color=palette, thickness=thickness, color_lookup=sv.ColorLookup.CLASS)
        self._violation_box = sv.BoxAnnotator(color=sv.Color.from_hex(VIOLATION_COLOR), thickness=thickness * 2)
        self._overstay_box = sv.BoxAnnotator(color=sv.Color.from_hex(OVERSTAY_COLOR), thickness=thickness * 2)
        self._trace = sv.TraceAnnotator(
            color=palette, thickness=thickness, trace_length=max(10, round(video_info.fps * 2)),
            position=sv.Position.CENTER, color_lookup=sv.ColorLookup.CLASS,
        )
        # Counts are drawn by draw_line_counts with the configured font.
        self._line = sv.LineZoneAnnotator(
            thickness=thickness, color=sv.Color.from_hex(LINE_COLOR), display_in_count=False, display_out_count=False,
        )
        self._processing_ms = 0.0
        self._last_stats = TrafficStatistics(by_class=monitor.get_statistics().by_class)

    @property
    def monitor(self) -> TrafficMonitor:
        """Underlying traffic monitor."""
        return self._monitor

    @property
    def events(self) -> EventService:
        """Region event log of this run."""
        return self._events

    @property
    def last_statistics(self) -> TrafficStatistics:
        """Statistics of the most recent frame."""
        return self._last_stats

    def process(self, frame: np.ndarray, frame_index: int) -> tuple[np.ndarray, TrafficStatistics]:
        """Process one frame and return (annotated frame, statistics)."""
        started = time.perf_counter()
        result = self._monitor.process_frame(frame, frame_index)
        self._events.record(result.region_events)
        result.statistics = replace(
            result.statistics, recent_events=self._events.recent(_RECENT_EVENTS), events_total=self._events.count()
        )
        annotated = self.annotate(frame.copy(), result)
        elapsed = (time.perf_counter() - started) * 1000
        self._processing_ms = elapsed if self._processing_ms == 0 else (1 - _EMA) * self._processing_ms + _EMA * elapsed
        self._last_stats = replace(result.statistics, processing_ms=self._processing_ms)
        return annotated, self._last_stats

    def annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw regions, traces, boxes, labels, alerts, counting line and overlays."""
        detections = result.detections
        regions = result.statistics.regions
        style = self._style
        if self._monitor.roi_manager is not None:
            draw_regions(
                frame, self._monitor.roi_manager.zones, style,
                occupancy={r.roi_id: r.occupancy for r in regions},
                alerts={r.roi_id: sum(s.overstay for s in r.active_sessions) for r in regions},
            )
        frame = self._trace.annotate(frame, detections)
        frame = self._box.annotate(frame, detections)

        overstaying = self._overstaying(result)
        speeding = {int(t) for t in result.violating_ids}
        amber_only = {tid: ids for tid, ids in overstaying.items() if tid not in speeding}
        if amber_only:
            flagged = detections[np.isin(detections.tracker_id, list(amber_only))]
            frame = self._overstay_box.annotate(frame, flagged)
        if speeding:
            flagged = detections[np.isin(detections.tracker_id, list(speeding))]
            frame = self._violation_box.annotate(frame, flagged)

        colors = [hex_to_bgr(CLASS_COLORS.get(int(c), "#FFFFFF")) for c in detections.class_id]
        draw_box_labels(frame, detections.xyxy, self._labels(result), colors, style)
        alerts, alert_boxes, alert_colors, alert_text = [], [], [], []
        for box, tracker_id in zip(detections.xyxy, detections.tracker_id):
            tid = int(tracker_id)
            if tid in speeding:
                alerts.append(f"OVERSPEED + OVERSTAY {overstaying[tid]}" if tid in overstaying else "OVERSPEED")
                alert_colors.append(hex_to_bgr(VIOLATION_COLOR))
            elif tid in amber_only:
                alerts.append(f"OVERSTAY {amber_only[tid]}")
                alert_colors.append(hex_to_bgr(OVERSTAY_COLOR))
            else:
                continue
            alert_boxes.append(box)
        if alerts:
            draw_box_labels(frame, np.array(alert_boxes), alerts, alert_colors, style, text_color=_WHITE, below=True)

        line_zone = self._monitor.counter.line_zone
        frame = self._line.annotate(frame, line_counter=line_zone)
        start, end = line_zone.vector.start, line_zone.vector.end
        draw_line_counts(frame, (start.x, start.y), (end.x, end.y), line_zone.in_count, line_zone.out_count, style)
        stats = replace(result.statistics, fps=self._last_stats.fps)
        draw_text_panel(frame, build_overlay_lines(stats, self._monitor.speed_limit), style, anchor="top-left")
        timestamp = f"{frame_to_timestamp(result.frame_index, self._fps)}   f{result.frame_index}"
        draw_text_panel(frame, [(timestamp, (240, 235, 231))], style, anchor="top-right")
        return frame

    def set_fps(self, fps: float) -> None:
        """Record the measured pipeline throughput for the overlay."""
        self._last_stats = replace(self._last_stats, fps=fps)

    def reset(self) -> None:
        """Reset the monitor, the event log and timing."""
        self._monitor.reset()
        self._events.reset()
        self._processing_ms = 0.0
        self._last_stats = TrafficStatistics(by_class=self._monitor.get_statistics().by_class)

    def _labels(self, result: DetectionResult) -> list[str]:
        """``#id class conf | km/h`` plus a second line ``zone_a 3.2s`` for objects inside regions."""
        detections = result.detections
        dwell = self._dwell_lines(result)
        labels = []
        for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence):
            speed = result.speeds.get(int(tracker_id))
            text = f"#{tracker_id} {self._monitor.detector.get_class_name(class_id)} {confidence:.2f}"
            text = f"{text} | {speed:.0f} km/h" if speed is not None else text
            region_text = dwell.get(int(tracker_id))
            labels.append(f"{text}\n{region_text}" if region_text else text)
        return labels

    @staticmethod
    def _dwell_lines(result: DetectionResult) -> dict[int, str]:
        """Track id → ``zone_a 3.2s zone_b 1.0s`` for every open session."""
        lines: dict[int, list[str]] = {}
        for region in result.statistics.regions:
            for session in region.active_sessions:
                lines.setdefault(session.track_id, []).append(f"{region.roi_id} {session.dwell_seconds:.1f}s")
        return {track_id: "  ".join(parts) for track_id, parts in lines.items()}

    @staticmethod
    def _overstaying(result: DetectionResult) -> dict[int, str]:
        """Track id → region ids where the open session exceeded its limit."""
        flagged: dict[int, list[str]] = {}
        for region in result.statistics.regions:
            for session in region.active_sessions:
                if session.overstay:
                    flagged.setdefault(session.track_id, []).append(region.roi_id)
        return {track_id: ",".join(ids) for track_id, ids in flagged.items()}
