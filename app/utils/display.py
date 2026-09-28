"""Frame drawing and conversion helpers (OpenCV only; Qt is imported lazily)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.models.traffic_statistics import TrafficStatistics

if TYPE_CHECKING:
    from PySide6.QtGui import QImage

Color = tuple[int, int, int]
WHITE: Color = (240, 235, 231)
GREEN: Color = (156, 214, 79)
ORANGE: Color = (99, 168, 255)
RED: Color = (62, 75, 255)
GREY: Color = (178, 165, 154)


def hex_to_bgr(value: str) -> Color:
    """Convert ``#RRGGBB`` to an OpenCV BGR tuple."""
    value = value.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return blue, green, red


def overlay_scale(frame: np.ndarray) -> float:
    """Text scale proportional to frame height (1.0 at 1080p)."""
    return max(0.5, frame.shape[0] / 1080)


def draw_text_panel(frame: np.ndarray, lines: list[tuple[str, Color]], anchor: str = "top-left") -> np.ndarray:
    """Draw ASCII lines on a translucent panel in a frame corner (in place)."""
    if not lines:
        return frame
    scale = overlay_scale(frame)
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, max(1, round(2 * scale))
    line_height = int(34 * scale)
    padding = int(14 * scale)
    width = max(cv2.getTextSize(text, font, font_scale, thickness)[0][0] for text, _ in lines) + 2 * padding
    height = line_height * len(lines) + padding
    margin = int(16 * scale)
    x0 = margin if anchor == "top-left" else frame.shape[1] - width - margin
    y0 = margin
    x1, y1 = min(frame.shape[1], x0 + width), min(frame.shape[0], y0 + height)
    region = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(region, 0.3, np.full_like(region, (16, 12, 10)), 0.7, 0)
    for index, (text, color) in enumerate(lines):
        baseline = y0 + padding + line_height * index + int(20 * scale)
        cv2.putText(frame, text, (x0 + padding, baseline), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def build_overlay_lines(stats: TrafficStatistics, speed_limit: float) -> list[tuple[str, Color]]:
    """Lines for the realtime dashboard drawn inside the frame (ASCII, cv2.putText has no Unicode)."""
    lines: list[tuple[str, Color]] = [
        (f"TOTAL IN : {stats.total_in}", GREEN),
        (f"TOTAL OUT: {stats.total_out}", ORANGE),
    ]
    for name, counts in stats.by_class.items():
        lines.append((f"{name.upper():<10} IN {counts['in']:>3}  OUT {counts['out']:>3}", WHITE))
    lines.append((f"SPEED VIOLATIONS: {stats.violations_count}", RED))
    lines.append((f"LIMIT {speed_limit:.0f} km/h  FPS {stats.fps:.1f}", GREY))
    return lines


def draw_calibration(
    frame: np.ndarray,
    line_start: list[int],
    line_end: list[int],
    source_points: list[list[float]],
) -> np.ndarray:
    """Draw the counting line and the A-B-C-D perspective area on a copy of ``frame``."""
    canvas = frame.copy()
    scale = overlay_scale(canvas)
    thickness = max(2, round(3 * scale))
    polygon = np.array(source_points, dtype=np.int32).reshape(-1, 1, 2)
    fill = canvas.copy()
    cv2.fillPoly(fill, [polygon], (168, 92, 255))
    canvas = cv2.addWeighted(fill, 0.18, canvas, 0.82, 0)
    cv2.polylines(canvas, [polygon], True, (168, 92, 255), thickness, cv2.LINE_AA)
    for label, (x, y) in zip("ABCD", source_points):
        point = (int(x), int(y))
        cv2.circle(canvas, point, thickness * 3, (168, 92, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (point[0] + 12, point[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 255, 255), thickness, cv2.LINE_AA)
    start, end = tuple(map(int, line_start)), tuple(map(int, line_end))
    cv2.line(canvas, start, end, hex_to_bgr("#FFD23F"), thickness, cv2.LINE_AA)
    cv2.arrowedLine(canvas, start, (start[0] + (end[0] - start[0]) // 8, start[1] + (end[1] - start[1]) // 8),
                    hex_to_bgr("#FFD23F"), thickness * 2, cv2.LINE_AA, tipLength=0.3)
    label = "COUNTING LINE (start -> end)"
    font_scale = 1.0 * scale
    (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    mid_x = (start[0] + end[0]) // 2
    text_x = int(np.clip(mid_x - text_w // 2, 10, max(10, canvas.shape[1] - text_w - 10)))
    text_y = int(np.clip((start[1] + end[1]) // 2 - 20, 40, canvas.shape[0] - 10))
    cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness * 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, hex_to_bgr("#FFD23F"), thickness, cv2.LINE_AA)
    return canvas


def resize_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale a frame to ``max_width`` keeping aspect ratio; never upscale."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    new_height = round(height * max_width / width)
    return cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)


def to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert a BGR frame to an independent RGB ``QImage`` (safe to send across threads)."""
    from PySide6.QtGui import QImage

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
