"""Run the core monitor on a frame and draw all annotations."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import supervision as sv

from app.config.settings import AppConfig
from app.constants.classes import CLASS_COLORS, LINE_COLOR, VIOLATION_COLOR
from app.core.traffic_monitor import TrafficMonitor
from app.models.detection_result import DetectionResult
from app.models.traffic_statistics import TrafficStatistics
from app.utils.display import build_overlay_lines, draw_text_panel, hex_to_bgr
from app.utils.time_utils import frame_to_timestamp

_EMA = 0.1


def _class_palette() -> sv.ColorPalette:
    """Palette indexed by COCO class id so ColorLookup.CLASS picks the class colour."""
    size = max(CLASS_COLORS) + 1
    colors = [sv.Color.from_hex(CLASS_COLORS.get(index, "#FFFFFF")) for index in range(size)]
    return sv.ColorPalette(colors)


class FrameProcessor:
    """Frame → TrafficMonitor → annotated frame + statistics."""

    def __init__(self, monitor: TrafficMonitor, config: AppConfig, video_info: sv.VideoInfo) -> None:
        """Create annotators scaled to the video resolution."""
        self._monitor = monitor
        self._fps = video_info.fps
        resolution = video_info.resolution_wh
        thickness = sv.calculate_optimal_line_thickness(resolution)
        text_scale = sv.calculate_optimal_text_scale(resolution)
        palette = _class_palette()
        violation = sv.Color.from_hex(VIOLATION_COLOR)
        self._box = sv.BoxAnnotator(color=palette, thickness=thickness, color_lookup=sv.ColorLookup.CLASS)
        self._violation_box = sv.BoxAnnotator(color=violation, thickness=thickness * 2)
        self._label = sv.LabelAnnotator(
            color=palette, text_scale=text_scale, text_thickness=thickness, text_color=sv.Color.BLACK,
            text_position=sv.Position.TOP_LEFT, color_lookup=sv.ColorLookup.CLASS, smart_position=True,
        )
        self._violation_label = sv.LabelAnnotator(
            color=violation, text_scale=text_scale, text_thickness=thickness, text_color=sv.Color.WHITE,
            text_position=sv.Position.BOTTOM_LEFT, smart_position=True,
        )
        self._trace = sv.TraceAnnotator(
            color=palette, thickness=thickness, trace_length=max(10, round(video_info.fps * 2)),
            position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.CLASS,
        )
        self._line = sv.LineZoneAnnotator(
            thickness=thickness, color=sv.Color.from_hex(LINE_COLOR), text_thickness=thickness,
            text_scale=text_scale, text_color=sv.Color.BLACK,
        )
        self._processing_ms = 0.0
        self._last_stats = TrafficStatistics(by_class=monitor.get_statistics().by_class)

    @property
    def monitor(self) -> TrafficMonitor:
        """Underlying traffic monitor."""
        return self._monitor

    @property
    def last_statistics(self) -> TrafficStatistics:
        """Statistics of the most recent frame."""
        return self._last_stats

    def process(self, frame: np.ndarray, frame_index: int) -> tuple[np.ndarray, TrafficStatistics]:
        """Process one frame and return (annotated frame, statistics)."""
        started = time.perf_counter()
        result = self._monitor.process_frame(frame, frame_index)
        annotated = self.annotate(frame.copy(), result)
        elapsed = (time.perf_counter() - started) * 1000
        self._processing_ms = elapsed if self._processing_ms == 0 else (1 - _EMA) * self._processing_ms + _EMA * elapsed
        self._last_stats = replace(result.statistics, processing_ms=self._processing_ms)
        return annotated, self._last_stats

    def annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw traces, boxes, labels, violations, counting line and overlays."""
        detections = result.detections
        frame = self._trace.annotate(frame, detections)
        frame = self._box.annotate(frame, detections)
        frame = self._label.annotate(frame, detections, labels=self._labels(result))
        if result.violating_ids:
            mask = np.isin(detections.tracker_id, list(result.violating_ids))
            flagged = detections[mask]
            frame = self._violation_box.annotate(frame, flagged)
            frame = self._violation_label.annotate(frame, flagged, labels=["OVERSPEED"] * len(flagged))
        frame = self._line.annotate(frame, line_counter=self._monitor.counter.line_zone)
        stats = replace(result.statistics, fps=self._last_stats.fps)
        draw_text_panel(frame, build_overlay_lines(stats, self._monitor.speed_limit), anchor="top-left")
        timestamp = f"{frame_to_timestamp(result.frame_index, self._fps)}  f{result.frame_index}"
        draw_text_panel(frame, [(timestamp, (240, 235, 231))], anchor="top-right")
        return frame

    def set_fps(self, fps: float) -> None:
        """Record the measured pipeline throughput for the overlay."""
        self._last_stats = replace(self._last_stats, fps=fps)

    def reset(self) -> None:
        """Reset the monitor and timing."""
        self._monitor.reset()
        self._processing_ms = 0.0
        self._last_stats = TrafficStatistics(by_class=self._monitor.get_statistics().by_class)

    def _labels(self, result: DetectionResult) -> list[str]:
        detections = result.detections
        labels = []
        for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence):
            speed = result.speeds.get(int(tracker_id))
            text = f"#{tracker_id} {self._monitor.detector.get_class_name(class_id)} {confidence:.2f}"
            labels.append(f"{text} | {speed:.0f} km/h" if speed is not None else text)
        return labels
