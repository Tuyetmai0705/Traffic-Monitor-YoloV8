"""Typed application configuration.

The dataclasses only describe the schema. Every value comes from
``default_config.json`` (optionally overridden by ``user_config.json`` and
per-video calibration files), so nothing is hard-coded in core logic.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from app.models.roi import ROI

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"
USER_CONFIG_PATH = CONFIG_DIR / "user_config.json"


@dataclass
class ModelConfig:
    """YOLO model and inference parameters."""

    path: str
    confidence: float
    iou: float
    device: str
    half: bool
    imgsz: int
    agnostic_nms: bool


@dataclass
class TrackerConfig:
    """ByteTrack and smoothing parameters."""

    track_activation_threshold: float
    lost_track_buffer: int
    minimum_matching_threshold: float
    minimum_consecutive_frames: int
    smoother_length: int
    smoothing: str
    min_hits: int
    kalman_measurement_scale: float
    kalman_size_scale: float
    coast_frames: int
    stitch_frames: int
    class_vote_window: int


@dataclass
class CountingLineConfig:
    """Counting line endpoints in frame pixels."""

    start: list[int]
    end: list[int]


@dataclass
class SpeedConfig:
    """Speed estimation and violation parameters."""

    limit_kmh: float
    window_seconds: float
    min_samples: int
    stale_seconds: float
    max_kmh: float
    smoothing_alpha: float
    violation_confirm_samples: int
    min_pixel_travel: float
    max_half_window_difference: float
    max_metres_per_pixel: float


@dataclass
class PerspectiveConfig:
    """Pixel quadrilateral A, B, C, D and its real-world size in metres."""

    source_points: list[list[float]]
    road_width_m: float
    road_length_m: float


@dataclass
class OutputConfig:
    """Output file locations, relative to the project root or absolute."""

    video_path: str
    report_path: str
    log_path: str


@dataclass
class UIConfig:
    """Desktop UI refresh behaviour."""

    dashboard_refresh_ms: int
    display_max_width: int


@dataclass
class AnnotationConfig:
    """Text drawn on frames. Sizes are pixels at 1080p and scale with the frame height."""

    font_family: str
    font_path: str
    label_font_px: int
    panel_font_px: int


@dataclass
class ProcessingConfig:
    """Throughput settings: frames wider than ``max_width`` are downscaled before processing."""

    max_width: int
    queue_size: int


@dataclass
class RegionsConfig:
    """Region-of-interest monitoring: zones, membership anchor and exit grace period."""

    enabled: bool
    anchor: str
    grace_seconds: float
    min_confirm_frames: int
    zones: list[ROI]

    @classmethod
    def from_dict(cls, data: Any) -> RegionsConfig:
        """Build from JSON, converting each zone to an ``ROI``."""
        if not isinstance(data, dict):
            raise ValueError("'regions' phải là object JSON")
        missing = {"enabled", "anchor", "grace_seconds", "min_confirm_frames", "zones"} - data.keys()
        if missing:
            raise ValueError(f"'regions' thiếu khóa: {', '.join(sorted(missing))}")
        if not isinstance(data["zones"], list):
            raise ValueError("'regions.zones' phải là danh sách")
        return cls(
            enabled=bool(data["enabled"]),
            anchor=str(data["anchor"]),
            grace_seconds=float(data["grace_seconds"]),
            min_confirm_frames=int(data["min_confirm_frames"]),
            zones=[ROI.from_dict(zone) for zone in data["zones"]],
        )


_Section = TypeVar("_Section")


def _build_section(section_cls: type[_Section], data: Any, name: str) -> _Section:
    """Build one dataclass section, rejecting missing or unknown keys."""
    if not isinstance(data, dict):
        raise ValueError(f"'{name}' phải là object JSON")
    expected = {f.name for f in fields(section_cls)}  # type: ignore[arg-type]
    missing = expected - data.keys()
    if missing:
        raise ValueError(f"'{name}' thiếu khóa: {', '.join(sorted(missing))}")
    return section_cls(**{key: data[key] for key in expected})


@dataclass
class AppConfig:
    """Complete application configuration."""

    model: ModelConfig
    classes: list[int]
    tracker: TrackerConfig
    counting_line: CountingLineConfig
    speed: SpeedConfig
    perspective: PerspectiveConfig
    output: OutputConfig
    ui: UIConfig
    processing: ProcessingConfig
    regions: RegionsConfig
    annotation: AnnotationConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    def copy(self) -> AppConfig:
        """Return a deep copy that can be edited safely."""
        return AppConfig.from_dict(copy.deepcopy(self.to_dict()))

    def scaled_geometry(self, scale: float) -> AppConfig:
        """Return a copy whose pixel coordinates are multiplied by ``scale``.

        Calibration is stored in original video pixels; processing may run on a
        downscaled frame, so the counting line, source points and ROI polygons
        must follow.
        """
        scaled = self.copy()
        if scale == 1.0:
            return scaled
        scaled.counting_line.start = [round(v * scale) for v in self.counting_line.start]
        scaled.counting_line.end = [round(v * scale) for v in self.counting_line.end]
        scaled.perspective.source_points = [[x * scale, y * scale] for x, y in self.perspective.source_points]
        scaled.regions.zones = [zone.scaled(scale) for zone in self.regions.zones]
        return scaled

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """Build a config from a dictionary. Raises ValueError on bad structure."""
        if not isinstance(data, dict):
            raise ValueError("Cấu hình phải là object JSON")
        if "classes" not in data or not isinstance(data["classes"], list):
            raise ValueError("'classes' phải là danh sách class id")
        return cls(
            model=_build_section(ModelConfig, data.get("model"), "model"),
            classes=[int(class_id) for class_id in data["classes"]],
            tracker=_build_section(TrackerConfig, data.get("tracker"), "tracker"),
            counting_line=_build_section(CountingLineConfig, data.get("counting_line"), "counting_line"),
            speed=_build_section(SpeedConfig, data.get("speed"), "speed"),
            perspective=_build_section(PerspectiveConfig, data.get("perspective"), "perspective"),
            output=_build_section(OutputConfig, data.get("output"), "output"),
            ui=_build_section(UIConfig, data.get("ui"), "ui"),
            processing=_build_section(ProcessingConfig, data.get("processing"), "processing"),
            regions=RegionsConfig.from_dict(data.get("regions")),
            annotation=_build_section(AnnotationConfig, data.get("annotation"), "annotation"),
        )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated recursively with ``override``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
