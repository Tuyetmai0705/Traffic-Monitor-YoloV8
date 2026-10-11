"""Path helpers anchored at the project root."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
INPUT_DIR = PROJECT_ROOT / "data" / "input"
CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"
OUTPUT_DIRS: tuple[Path, ...] = (
    PROJECT_ROOT / "outputs" / "videos",
    PROJECT_ROOT / "outputs" / "reports",
    PROJECT_ROOT / "outputs" / "logs",
)


def resolve_path(path: str | Path) -> Path:
    """Return an absolute path; relative paths are resolved from the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed. Raises OSError on failure."""
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(file_path: str | Path) -> Path:
    """Create the parent directory of a file path and return the absolute file path."""
    target = resolve_path(file_path)
    ensure_dir(target.parent)
    return target


def to_display_path(path: str | Path) -> str:
    """Return a project-relative path with forward slashes when possible."""
    target = resolve_path(path)
    try:
        return target.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return target.as_posix()


def is_video_file(path: str | Path) -> bool:
    """Return True when the path has a supported video extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


_NUMBER_ARRAY = re.compile(r"\[\s*(-?[\d.]+(?:\s*,\s*-?[\d.]+)*)\s*\]")


def format_json(data: Any) -> str:
    """Indented JSON where arrays of plain numbers (points, lines) stay on one line."""
    text = json.dumps(data, indent=4, ensure_ascii=False)
    return _NUMBER_ARRAY.sub(lambda m: "[" + ", ".join(part.strip() for part in m.group(1).split(",")) + "]", text) + "\n"
