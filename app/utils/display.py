"""Frame drawing and conversion helpers (OpenCV + TrueType text; Qt is imported lazily)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.config.settings import AnnotationConfig
from app.constants.classes import VEHICLE_CLASSES, roi_color
from app.models.roi import ROI
from app.models.traffic_statistics import TrafficStatistics
from app.utils.text import TextRenderer, get_renderer, scaled_px

if TYPE_CHECKING:
    from PySide6.QtGui import QImage

Color = tuple[int, int, int]
WHITE: Color = (240, 235, 231)
GREEN: Color = (156, 214, 79)
ORANGE: Color = (99, 168, 255)
RED: Color = (62, 75, 255)
AMBER: Color = (28, 159, 255)
GREY: Color = (178, 165, 154)


def hex_to_bgr(value: str) -> Color:
    """Convert ``#RRGGBB`` to an OpenCV BGR tuple."""
    value = value.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return blue, green, red


@dataclass(frozen=True)
class TextStyle:
    """Font and sizes for one frame size."""

    renderer: TextRenderer
    label_px: int
    panel_px: int

    @classmethod
    def for_frame(cls, annotation: AnnotationConfig | None, frame_height: int) -> TextStyle:
        """Scale the configured 1080p sizes to ``frame_height`` (defaults: Times New Roman 26/19 px)."""
        family, path, label, panel = ("Times New Roman", "", 26, 19) if annotation is None else (
            annotation.font_family, annotation.font_path, annotation.label_font_px, annotation.panel_font_px)
        return cls(get_renderer(family, path), scaled_px(label, frame_height), scaled_px(panel, frame_height))


def overlay_scale(frame: np.ndarray) -> float:
    """Line-thickness scale proportional to frame height (1.0 at 1080p)."""
    return max(0.5, frame.shape[0] / 1080)


def draw_text_panel(
    frame: np.ndarray, lines: list[tuple[str, Color]], style: TextStyle, anchor: str = "top-left"
) -> np.ndarray:
    """Draw lines on a translucent panel in a frame corner (in place)."""
    if not lines:
        return frame
    px = style.panel_px
    padding, margin = round(px * 0.55), round(px * 0.6)
    line_height = round(px * 1.22)
    widths = [style.renderer.measure(text, px)[0] for text, _ in lines]
    width = max(widths) + 2 * padding
    height = line_height * len(lines) + 2 * padding
    x0 = margin if anchor == "top-left" else frame.shape[1] - width - margin
    y0 = margin
    x1, y1 = min(frame.shape[1], x0 + width), min(frame.shape[0], y0 + height)
    region = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(region, 0.3, np.full_like(region, (16, 12, 10)), 0.7, 0)
    for index, (text, color) in enumerate(lines):
        style.renderer.draw(frame, text, (x0 + padding, y0 + padding + index * line_height), px, color)
    return frame


def build_overlay_lines(stats: TrafficStatistics, speed_limit: float) -> list[tuple[str, Color]]:
    """Lines for the realtime dashboard drawn inside the frame."""
    lines: list[tuple[str, Color]] = [
        (f"TOTAL IN: {stats.total_in}", GREEN),
        (f"TOTAL OUT: {stats.total_out}", ORANGE),
    ]
    for name, counts in stats.by_class.items():
        if name not in VEHICLE_CLASSES.values() and not (counts["in"] or counts["out"]):
            continue  # e.g. person: only shown once someone was counted
        lines.append((f"{name.upper()}   IN {counts['in']}   OUT {counts['out']}", WHITE))
    lines.append((f"SPEED VIOLATIONS: {stats.violations_count}", RED))
    for region in stats.regions:
        lines.append((f"{region.roi_id.upper()}   OCC {region.occupancy}   IN {region.entries}   ALERT {region.overstays}", AMBER))
    lines.append((f"LIMIT {speed_limit:.0f} km/h   FPS {stats.fps:.1f}", GREY))
    return lines


def draw_box_labels(
    frame: np.ndarray,
    xyxy: np.ndarray,
    labels: Sequence[str],
    backgrounds: Sequence[Color],
    style: TextStyle,
    text_color: Color = (16, 12, 10),
    below: bool = False,
) -> np.ndarray:
    """Draw one label per box at a fixed place: above the box (or under it with ``below``).

    The position depends only on the box, never on the other labels, so labels
    do not jump from frame to frame. A label that would leave the frame is
    moved inside the box edge instead.
    """
    px = style.label_px
    padding = max(2, round(px * 0.18))
    for box, text, background in zip(xyxy, labels, backgrounds):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        _, height = style.renderer.measure(text, px)
        box_h = height + 2 * padding
        if below:
            top = y2 if y2 + box_h <= frame.shape[0] else y2 - box_h
        else:
            top = y1 - box_h if y1 - box_h >= 0 else y1
        style.renderer.draw(frame, text, (x1, top), px, text_color, background=background, padding=padding)
    return frame


def draw_line_counts(
    frame: np.ndarray, start: Sequence[int], end: Sequence[int], in_count: int, out_count: int, style: TextStyle
) -> np.ndarray:
    """Draw ``in: N`` above and ``out: N`` below the middle of the counting line."""
    px = style.label_px
    padding = max(2, round(px * 0.2))
    middle_x, middle_y = (start[0] + end[0]) // 2, (start[1] + end[1]) // 2
    gap = round(px * 0.35)
    for text, offset in ((f"in: {in_count}", -1), (f"out: {out_count}", 1)):
        width, height = style.renderer.measure(text, px)
        top = middle_y - gap - height - 2 * padding if offset < 0 else middle_y + gap
        style.renderer.draw(frame, text, (middle_x - width // 2 - padding, top), px, (16, 12, 10),
                            background=hex_to_bgr("#FFD23F"), padding=padding)
    return frame


def draw_regions(
    frame: np.ndarray,
    zones: Sequence[ROI],
    style: TextStyle,
    occupancy: dict[str, int] | None = None,
    alerts: dict[str, int] | None = None,
) -> np.ndarray:
    """Draw ROI polygons with a translucent fill and a label ``roi_id: N`` (in place)."""
    if not zones:
        return frame
    thickness = max(2, round(3 * overlay_scale(frame)))
    fill = frame.copy()
    for index, zone in enumerate(zones):
        cv2.fillPoly(fill, [np.array(zone.polygon, dtype=np.int32)], hex_to_bgr(roi_color(index)))
    frame[:] = cv2.addWeighted(fill, 0.16, frame, 0.84, 0)
    px = style.label_px
    padding = max(2, round(px * 0.2))
    for index, zone in enumerate(zones):
        color = hex_to_bgr(roi_color(index))
        polygon = np.array(zone.polygon, dtype=np.int32)
        cv2.polylines(frame, [polygon], True, color, thickness, cv2.LINE_AA)
        text = zone.roi_id
        if occupancy is not None:
            text += f": {occupancy.get(zone.roi_id, 0)}"
        if alerts and alerts.get(zone.roi_id):
            text += f"  !{alerts[zone.roi_id]}"
        _, height = style.renderer.measure(text, px)
        top = polygon[np.argmin(polygon[:, 1])]
        # Inside the polygon, just under its top vertex: vehicle labels sit above boxes, so they no longer collide.
        style.renderer.draw(frame, text, (int(top[0]) + thickness, int(top[1]) + thickness), px,
                            (16, 12, 10), background=color, padding=padding)
    return frame


def draw_calibration(
    frame: np.ndarray,
    line_start: list[int],
    line_end: list[int],
    source_points: list[list[float]],
    zones: Sequence[ROI] = (),
    annotation: AnnotationConfig | None = None,
) -> np.ndarray:
    """Draw ROIs, the counting line and the A-B-C-D perspective area on a copy of ``frame``."""
    style = TextStyle.for_frame(annotation, frame.shape[0])
    canvas = draw_regions(frame.copy(), zones, style)
    scale = overlay_scale(canvas)
    thickness = max(2, round(3 * scale))
    polygon = np.array(source_points, dtype=np.int32).reshape(-1, 1, 2)
    fill = canvas.copy()
    cv2.fillPoly(fill, [polygon], (168, 92, 255))
    canvas = cv2.addWeighted(fill, 0.18, canvas, 0.82, 0)
    cv2.polylines(canvas, [polygon], True, (168, 92, 255), thickness, cv2.LINE_AA)
    big = round(style.label_px * 1.3)
    for label, (x, y) in zip("ABCD", source_points):
        point = (int(x), int(y))
        cv2.circle(canvas, point, thickness * 3, (168, 92, 255), -1, cv2.LINE_AA)
        style.renderer.draw(canvas, label, (point[0] + 12, point[1] - big - 12), big, (255, 255, 255))
    start, end = tuple(map(int, line_start)), tuple(map(int, line_end))
    cv2.line(canvas, start, end, hex_to_bgr("#FFD23F"), thickness, cv2.LINE_AA)
    cv2.arrowedLine(canvas, start, (start[0] + (end[0] - start[0]) // 8, start[1] + (end[1] - start[1]) // 8),
                    hex_to_bgr("#FFD23F"), thickness * 2, cv2.LINE_AA, tipLength=0.3)
    label = "COUNTING LINE (start → end)"
    width, height = style.renderer.measure(label, style.label_px)
    mid_x, mid_y = (start[0] + end[0]) // 2, (start[1] + end[1]) // 2
    style.renderer.draw(canvas, label, (mid_x - width // 2, mid_y - height - 2 * thickness - 8), style.label_px,
                        hex_to_bgr("#FFD23F"), background=(20, 20, 20), padding=4, alpha=0.6)
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
