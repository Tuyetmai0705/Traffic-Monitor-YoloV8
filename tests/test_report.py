"""Phase 11: ReportService generation and export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.traffic_statistics import TrafficStatistics
from app.models.violation import Violation
from app.services.report_service import REQUIRED_KEYS, ReportMeta, ReportService

META = ReportMeta(
    video="data/input/vehicles.mp4", frames_processed=250, fps=25.0, model="yolov8n.pt", device="cuda:0",
    confidence=0.3, speed_limit_kmh=80, output_video="outputs/videos/traffic_result.mp4", completed=True,
)


def _stats() -> TrafficStatistics:
    return TrafficStatistics(
        total_in=47, total_out=52,
        by_class={"car": {"in": 40, "out": 48}, "motorcycle": {"in": 0, "out": 0}, "bus": {"in": 2, "out": 1}, "truck": {"in": 5, "out": 3}},
        violations_count=2,
        violations=(Violation(17, "car", 94.63, 80, 310, 330), Violation(63, "truck", 83.1, 80, 1645, 1660)),
        average_speed_kmh=71.44, max_speed_kmh=94.63,
    )


def test_report_has_required_keys_first() -> None:
    report = ReportService().build_report(_stats(), META)
    assert tuple(report)[:5] == REQUIRED_KEYS
    assert report["video"] == "data/input/vehicles.mp4"
    assert report["total_in"] == 47 and report["total_out"] == 52
    assert report["by_class"]["car"] == {"in": 40, "out": 48}


def test_speed_violations_keyed_by_tracker_id() -> None:
    report = ReportService().build_report(_stats(), META)
    assert set(report["speed_violations"]) == {"17", "63"}
    record = report["speed_violations"]["17"]
    assert record["class"] == "car"
    assert record["max_speed_kmh"] == 94.6
    assert record["first_seen"] == "00:00:12.4"
    assert report["speed_summary"] == {"average_kmh": 71.4, "max_kmh": 94.6}
    assert report["duration_seconds"] == 10.0
    assert report["status"] == "completed"


def test_export_creates_missing_directories(tmp_path: Path) -> None:
    service = ReportService()
    target = tmp_path / "nested" / "dir" / "traffic_report.json"
    written = service.export(service.build_report(_stats(), META), target)
    assert written == target
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["total_in"] == 47
    assert loaded["speed_violations"]["63"]["class"] == "truck"


def test_export_rejects_incomplete_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ReportService().export({"video": "x"}, tmp_path / "r.json")


def test_empty_statistics_still_produce_valid_report() -> None:
    report = ReportService().build_report(TrafficStatistics(), META)
    assert report["speed_violations"] == {}
    assert report["total_in"] == 0
