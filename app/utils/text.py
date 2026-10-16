"""TrueType text on OpenCV frames (Times New Roman by default, Unicode-capable).

``cv2.putText`` only has the built-in Hershey fonts and no Vietnamese glyphs.
Text is rendered with Pillow on a small patch of the frame, so the cost stays
proportional to the text area, not the frame size.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.utils.file_utils import resolve_path

Color = tuple[int, int, int]

# Bold faces first: thin serif strokes are hard to read on video.
FONT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "times new roman": (
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",   # metric-compatible with Times
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    ),
}
REFERENCE_HEIGHT = 1080


def find_font(family: str, explicit_path: str = "") -> str | None:
    """Return a TrueType file for ``family`` (or ``explicit_path``), or None when none is installed."""
    if explicit_path:
        path = resolve_path(explicit_path)
        return str(path) if path.is_file() else None
    for candidate in FONT_CANDIDATES.get(family.strip().lower(), ()):
        if Path(candidate).is_file():
            return candidate
    return None


def scaled_px(base_px: int, frame_height: int) -> int:
    """Font size for a frame, given a size defined at 1080p."""
    return max(8, round(base_px * frame_height / REFERENCE_HEIGHT))


class TextRenderer:
    """Measure and draw text boxes with one font file (falls back to Hershey when missing)."""

    def __init__(self, font_path: str | None) -> None:
        """Use ``font_path`` (TTF/OTF) or OpenCV's Hershey font when None."""
        self._path = font_path

    @property
    def font_path(self) -> str | None:
        """Font file in use, or None for the Hershey fallback."""
        return self._path

    @staticmethod
    @lru_cache(maxsize=32)
    def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(path, size)

    def measure(self, text: str, px: int) -> tuple[int, int]:
        """Width and height of ``text`` (one or more lines) at ``px``."""
        lines = text.split("\n")
        if self._path is None:
            scale = px / 30
            sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, px // 14))[0] for line in lines]
            return max(w for w, _ in sizes), int(len(lines) * px * 1.15)
        font = self._font(self._path, px)
        ascent, descent = font.getmetrics()
        width = max(int(np.ceil(font.getlength(line))) for line in lines)
        return width, (ascent + descent) * len(lines)

    def draw(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        px: int,
        color: Color,
        background: Color | None = None,
        padding: int = 0,
        alpha: float = 1.0,
    ) -> tuple[int, int, int, int]:
        """Draw text with its box's top-left at ``origin`` (clamped inside the frame).

        Returns the box ``(x1, y1, x2, y2)`` actually used.
        """
        width, height = self.measure(text, px)
        box_w, box_h = width + 2 * padding, height + 2 * padding
        frame_h, frame_w = frame.shape[:2]
        x1 = int(np.clip(origin[0], 0, max(0, frame_w - box_w)))
        y1 = int(np.clip(origin[1], 0, max(0, frame_h - box_h)))
        x2, y2 = min(frame_w, x1 + box_w), min(frame_h, y1 + box_h)
        if x2 <= x1 or y2 <= y1:
            return x1, y1, x1, y1
        if background is not None:
            patch = frame[y1:y2, x1:x2]
            fill = np.empty_like(patch)
            fill[:] = background
            frame[y1:y2, x1:x2] = fill if alpha >= 1.0 else cv2.addWeighted(fill, alpha, patch, 1 - alpha, 0)
        if self._path is None:
            self._draw_hershey(frame, text, (x1 + padding, y1 + padding), px, color)
        else:
            self._draw_truetype(frame, text, (x1, y1, x2, y2), padding, px, color)
        return x1, y1, x2, y2

    def _draw_truetype(self, frame: np.ndarray, text: str, box: tuple[int, int, int, int], padding: int, px: int, color: Color) -> None:
        x1, y1, x2, y2 = box
        font = self._font(self._path, px)
        ascent, descent = font.getmetrics()
        patch = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        drawer = ImageDraw.Draw(patch)
        for index, line in enumerate(text.split("\n")):
            drawer.text((padding, padding + index * (ascent + descent)), line, font=font, fill=(color[2], color[1], color[0]))
        frame[y1:y2, x1:x2] = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _draw_hershey(frame: np.ndarray, text: str, origin: tuple[int, int], px: int, color: Color) -> None:
        scale, thickness = px / 30, max(1, px // 14)
        for index, line in enumerate(text.split("\n")):
            baseline = origin[1] + int((index + 0.85) * px * 1.15)
            cv2.putText(frame, line, (origin[0], baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


@lru_cache(maxsize=8)
def get_renderer(family: str, explicit_path: str = "") -> TextRenderer:
    """Shared renderer for a font family (cached; fonts are read once)."""
    return TextRenderer(find_font(family, explicit_path))
