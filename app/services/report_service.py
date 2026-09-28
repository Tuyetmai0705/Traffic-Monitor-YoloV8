"""Build and export the JSON traffic report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import OutputPathError
from app.models.traffic_statistics import TrafficStatistics
from app.utils.file_utils import ensure_parent_dir, to_display_path
from app.utils.logger import get_logger
from app.utils.time_utils import frame_to_timestamp, now_iso

logger = get_logger("report_service")

REQUIRED_KEYS = ("video", "total_in", "total_out", "by_class", "speed_violations")


@dataclass(frozen=True)
class ReportMeta:
    """Run information that is not part of the traffic statistics."""

    video: str
    frames_processed: int
    fps: float
    model: str
    device: str
    confidence: float
    speed_limit_kmh: float
    output_video: str
    completed: bool


class ReportService:
    """Create report dictionaries and write them as UTF-8 JSON."""

    def build_report(self, statistics: TrafficStatistics, meta: ReportMeta) -> dict[str, Any]:
        """Return the report; the five required keys always come first."""
        violations = {
            str(v.tracker_id): {
                "class": v.class_name,
                "max_speed_kmh": round(v.max_speed_kmh, 1),
                "speed_limit_kmh": v.speed_limit_kmh,
                "first_seen": frame_to_timestamp(v.first_frame, meta.fps),
                "first_frame": v.first_frame,
                "last_frame": v.last_frame,
            }
            for v in statistics.violations
        }
        return {
            "video": meta.video,
            "total_in": statistics.total_in,
            "total_out": statistics.total_out,
            "by_class": {name: dict(counts) for name, counts in statistics.by_class.items()},
            "speed_violations": violations,
            "generated_at": now_iso(),
            "status": "completed" if meta.completed else "stopped",
            "frames_processed": meta.frames_processed,
            "duration_seconds": round(meta.frames_processed / meta.fps, 2) if meta.fps > 0 else 0.0,
            "model": meta.model,
            "device": meta.device,
            "confidence": meta.confidence,
            "speed_limit_kmh": meta.speed_limit_kmh,
            "speed_summary": {
                "average_kmh": round(statistics.average_speed_kmh, 1),
                "max_kmh": round(statistics.max_speed_kmh, 1),
            },
            "output_video": meta.output_video,
        }

    def export(self, report: dict[str, Any], path: str | Path) -> Path:
        """Write the report, creating the directory if needed."""
        missing = [key for key in REQUIRED_KEYS if key not in report]
        if missing:
            raise ValueError(f"report is missing keys: {missing}")
        try:
            target = ensure_parent_dir(path)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=4, ensure_ascii=False)
        except OSError as exc:
            raise OutputPathError(path, str(exc)) from exc
        logger.info("Report exported: %s", to_display_path(target))
        return target
