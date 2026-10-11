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


# ------------------------------------------------------------ region report
def _region_report(tmp_path: Path):
    import csv

    from app.config.settings import RegionsConfig
    from app.models.event import EventType, RegionEvent
    from app.models.region_statistics import RegionStatistics
    from app.models.roi import ROI
    from app.models.track_session import SessionStatus, TrackSession
    from app.services.report_service import csv_paths

    zone = ROI("zone_a", "Khu vực A", [[100, 100], [400, 100], [450, 350], [80, 350]], ["person"], 300)
    regions = RegionsConfig(enabled=True, anchor="bottom_center", grace_seconds=1.0, zones=[zone])
    sessions = [
        TrackSession(1, 18, "person", "zone_a", 12.4, 18.7, 18.7, 6.3, SessionStatus.EXITED),
        TrackSession(2, 21, "person", "zone_a", 20.0, 25.0, None, 5.0, SessionStatus.INSIDE),
    ]
    events = [
        RegionEvent("event-0001", 18, "zone_a", EventType.ENTER, 12.4, "person"),
        RegionEvent("event-0002", 18, "zone_a", EventType.EXIT, 18.7, "person", 6.3),
    ]
    stats = TrafficStatistics(regions=(RegionStatistics("zone_a", "Khu vực A", occupancy=1, peak_occupancy=2, entries=2, exits=1,
                                                        average_dwell_seconds=6.3, dwell_limit_seconds=300),))
    report = ReportService().build_report(stats, META, sessions, events, regions)
    path = ReportService().export(report, tmp_path / "out" / "traffic_report.json")
    sessions_csv, events_csv = csv_paths(path)
    with open(sessions_csv, encoding="utf-8-sig", newline="") as handle:
        session_rows = list(csv.DictReader(handle))
    with open(events_csv, encoding="utf-8-sig", newline="") as handle:
        event_rows = list(csv.DictReader(handle))
    return report, path, session_rows, event_rows


def test_region_section_in_json(tmp_path: Path) -> None:
    report, path, _, _ = _region_report(tmp_path)
    assert tuple(report)[:5] == REQUIRED_KEYS
    loaded = json.loads(path.read_text(encoding="utf-8"))
    regions = loaded["regions"]
    assert regions["time_base"] == "media_seconds" and regions["anchor"] == "bottom_center"
    zone = regions["zones"][0]
    assert zone["name"] == "Khu vực A" and zone["occupancy"] == 1 and zone["entries"] == 2
    assert zone["polygon"][0] == [100, 100] and zone["max_dwell_seconds"] == 300
    assert regions["sessions"][0]["dwell_seconds"] == 6.3 and regions["sessions"][1]["exit_time"] is None
    assert regions["events"][1]["event_type"] == "EXIT" and regions["events"][1]["roi_name"] == "Khu vực A"


def test_region_csv_files(tmp_path: Path) -> None:
    _, _, sessions, events = _region_report(tmp_path)
    assert sessions[0]["roi_name"] == "Khu vực A"
    assert sessions[0]["entry_timestamp"] == "00:00:12.4" and sessions[0]["exit_timestamp"] == "00:00:18.7"
    assert sessions[1]["status"] == "inside" and sessions[1]["exit_time"] == ""
    assert [row["event_type"] for row in events] == ["ENTER", "EXIT"]
    assert events[1]["dwell_seconds"] == "6.3" and events[0]["dwell_seconds"] == ""


def test_no_region_section_when_disabled(tmp_path: Path) -> None:
    from app.config.settings import RegionsConfig
    from app.services.report_service import csv_paths

    report = ReportService().build_report(_stats(), META, regions_config=RegionsConfig(False, "center", 1.0, []))
    assert "regions" not in report
    path = ReportService().export(report, tmp_path / "r.json")
    assert not any(p.exists() for p in csv_paths(path))
