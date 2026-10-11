"""Build and export the JSON traffic report and the region CSV files."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.settings import RegionsConfig
from app.core.exceptions import OutputPathError
from app.models.event import RegionEvent
from app.models.track_session import TrackSession
from app.models.traffic_statistics import TrafficStatistics
from app.utils.file_utils import ensure_parent_dir, to_display_path
from app.utils.logger import get_logger
from app.utils.time_utils import format_timestamp, frame_to_timestamp, now_iso

logger = get_logger("report_service")

REQUIRED_KEYS = ("video", "total_in", "total_out", "by_class", "speed_violations")
SESSION_COLUMNS = (
    "session_id", "roi_id", "roi_name", "track_id", "class_name", "entry_time", "entry_timestamp",
    "exit_time", "exit_timestamp", "dwell_seconds", "status", "overstay",
)
EVENT_COLUMNS = ("event_id", "timestamp", "time", "roi_id", "roi_name", "event_type", "track_id", "class_name", "dwell_seconds")


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


def csv_paths(report_path: str | Path) -> tuple[Path, Path]:
    """Return the session and event CSV paths written next to a JSON report."""
    path = Path(report_path)
    return path.with_name(f"{path.stem}_sessions.csv"), path.with_name(f"{path.stem}_events.csv")


class ReportService:
    """Create report dictionaries and write them as UTF-8 JSON plus region CSVs."""

    def build_report(
        self,
        statistics: TrafficStatistics,
        meta: ReportMeta,
        sessions: Sequence[TrackSession] = (),
        events: Sequence[RegionEvent] = (),
        regions_config: RegionsConfig | None = None,
    ) -> dict[str, Any]:
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
        report = {
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
        if regions_config is not None and regions_config.enabled and regions_config.zones:
            report["regions"] = self._regions_section(statistics, sessions, events, regions_config)
        return report

    def export(self, report: dict[str, Any], path: str | Path) -> Path:
        """Write the JSON report (and region CSVs when present), creating directories if needed."""
        missing = [key for key in REQUIRED_KEYS if key not in report]
        if missing:
            raise ValueError(f"report is missing keys: {missing}")
        try:
            target = ensure_parent_dir(path)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=4, ensure_ascii=False)
            if "regions" in report:
                self._write_csvs(report["regions"], target)
        except OSError as exc:
            raise OutputPathError(path, str(exc)) from exc
        logger.info("Report exported: %s", to_display_path(target))
        return target

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _regions_section(
        statistics: TrafficStatistics,
        sessions: Sequence[TrackSession],
        events: Sequence[RegionEvent],
        config: RegionsConfig,
    ) -> dict[str, Any]:
        by_id = {region.roi_id: region for region in statistics.regions}
        names = {zone.roi_id: zone.name for zone in config.zones}
        zones = []
        for zone in config.zones:
            summary = by_id[zone.roi_id].to_dict() if zone.roi_id in by_id else {"roi_id": zone.roi_id, "name": zone.name}
            summary.update({
                "polygon": zone.polygon,
                "allowed_classes": zone.allowed_classes,
                "max_dwell_seconds": zone.max_dwell_seconds,
            })
            zones.append(summary)
        return {
            "time_base": "media_seconds",
            "anchor": config.anchor,
            "grace_seconds": config.grace_seconds,
            "membership_rule": "anchor point inside or on the polygon boundary and class allowed",
            "zones": zones,
            "sessions": [dict(s.to_dict(), roi_name=names.get(s.roi_id, s.roi_id)) for s in sessions],
            "events": [dict(e.to_dict(), roi_name=names.get(e.roi_id, e.roi_id)) for e in events],
        }

    @staticmethod
    def _write_csvs(regions: dict[str, Any], report_path: Path) -> None:
        """Write ``<report>_sessions.csv`` and ``<report>_events.csv`` (UTF-8 with BOM for Excel)."""
        sessions_path, events_path = csv_paths(report_path)
        with open(sessions_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SESSION_COLUMNS)
            writer.writeheader()
            for session in regions["sessions"]:
                exit_time = session["exit_time"]
                writer.writerow({
                    **{key: session.get(key) for key in SESSION_COLUMNS if key in session},
                    "entry_timestamp": format_timestamp(session["entry_time"]),
                    "exit_time": "" if exit_time is None else exit_time,
                    "exit_timestamp": "" if exit_time is None else format_timestamp(exit_time),
                })
        with open(events_path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
            writer.writeheader()
            for event in regions["events"]:
                writer.writerow({
                    **{key: event.get(key) for key in EVENT_COLUMNS if key in event},
                    "time": format_timestamp(event["timestamp"]),
                    "dwell_seconds": "" if event["dwell_seconds"] is None else event["dwell_seconds"],
                })
        logger.info("Region CSV exported: %s, %s", to_display_path(sessions_path), to_display_path(events_path))
