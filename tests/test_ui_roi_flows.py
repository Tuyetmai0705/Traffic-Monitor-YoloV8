"""Region monitoring business flows driven through the real UI (Qt offscreen, YOLO on GPU)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from app.constants import messages as msg
from app.models.app_state import Action, AppState
from app.models.event import EventType
from app.services.report_service import csv_paths
from app.ui.main_window import MainWindow
from app.ui.roi_editor import ROIEditorDialog
from tests.conftest import SAMPLE_VIDEO, requires_gpu, requires_video
from tests.test_ui_flows import current_frame, qapp, short_clip, wait_ms, wait_until, window  # noqa: F401 (fixtures)

pytestmark = [pytest.mark.ui]


def open_editor(win: MainWindow) -> ROIEditorDialog:
    dialog = ROIEditorDialog(win.run_config, win._config_service, win._preview_frame,
                             win.video_path.name if win.video_path else None, win)
    dialog.resize(1320, 780)
    dialog.show()
    wait_ms(50)
    return dialog


def click(dialog: ROIEditorDialog, frame_point: tuple[int, int], button=Qt.MouseButton.LeftButton) -> None:
    QTest.mouseClick(dialog.canvas, button, Qt.KeyboardModifier.NoModifier, dialog.canvas.to_widget(frame_point).toPoint())


# ------------------------------------------------- NV11: draw and save regions
@requires_video
def test_draw_new_zone_with_mouse_and_save_for_video(window: MainWindow, tmp_path: Path) -> None:
    window.open_video(SAMPLE_VIDEO)
    assert window.region_panel.table.rowCount() == 2                     # sample zones from config
    dialog = open_editor(window)
    assert dialog.canvas.has_frame and len(dialog.zones) == 2

    zone = dialog.add_zone()
    assert zone.roi_id == "zone_c" and dialog.zone_list.currentRow() == 2
    corners = [(400, 300), (1400, 300), (1400, 800), (400, 800)]
    for point in corners:
        click(dialog, point)
    tolerance = 3840 / dialog.canvas.width() + 1                         # one screen pixel in frame pixels
    for drawn, expected in zip(zone.polygon, corners):
        assert abs(drawn[0] - expected[0]) <= tolerance and abs(drawn[1] - expected[1]) <= tolerance

    click(dialog, (900, 900))                                            # a fifth point …
    click(dialog, (900, 900), Qt.MouseButton.RightButton)                # … removed by right click
    assert len(zone.polygon) == 4

    start = dialog.canvas.to_widget(zone.polygon[2]).toPoint()           # drag vertex 3 down by 40 px
    QTest.mousePress(dialog.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(dialog.canvas, start + QPoint(0, 40))
    QTest.mouseRelease(dialog.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(0, 40))
    assert zone.polygon[2][1] > 800 + 100
    assert len(zone.polygon) == 4

    dialog.name.setText("Khu vực đỗ xe")
    dialog.name.textEdited.emit("Khu vực đỗ xe")
    dialog.class_boxes["person"].setChecked(True)
    dialog.max_dwell.setValue(30)
    assert "person" in dialog.class_warning.text()                       # person is not detected yet
    assert dialog.save_for_video_requested()
    dialog.accept()
    config = dialog.result_config()
    assert config is not None
    assert window.apply_regions(config, dialog.save_for_video_requested())

    saved = json.loads((tmp_path / "calibration" / "vehicles.json").read_text(encoding="utf-8"))
    new_zone = saved["regions"]["zones"][2]
    assert (new_zone["roi_id"], new_zone["name"], new_zone["allowed_classes"], new_zone["max_dwell_seconds"]) == (
        "zone_c", "Khu vực đỗ xe", ["person"], 30.0)
    assert window.region_panel.table.rowCount() == 3
    assert "vehicles.json" in window.message_label.text()
    assert window.state is AppState.VIDEO_SELECTED


@requires_video
def test_editor_rejects_incomplete_zone_and_deletes(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args: warnings.append(args[2])))
    window.open_video(SAMPLE_VIDEO)
    dialog = open_editor(window)
    dialog.add_zone()
    click(dialog, (500, 500))
    click(dialog, (900, 500))
    dialog.accept()
    assert dialog.result_config() is None
    assert warnings and "zone_c" in warnings[0] and "3 điểm" in warnings[0]
    assert dialog.zone_list.currentRow() == 2                            # offending zone selected

    dialog.delete_zone()
    dialog.zone_list.setCurrentRow(0)
    dialog.delete_zone()
    assert [z.roi_id for z in dialog.zones] == ["zone_b"]
    dialog.accept()
    assert dialog.result_config() is not None
    window.apply_regions(dialog.result_config(), save_for_video=False)
    assert [z.roi_id for z in window.run_config.regions.zones] == ["zone_b"]
    saved = json.loads(window._config_service.user_path.read_text(encoding="utf-8"))
    assert [z["roi_id"] for z in saved["regions"]["zones"]] == ["zone_b"]


def test_editor_without_video_cannot_draw(window: MainWindow) -> None:
    dialog = open_editor(window)
    assert not dialog.canvas.has_frame
    assert not dialog.save_for_video.isEnabled()
    zone = dialog.add_zone()
    QTest.mouseClick(dialog.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 100))
    assert zone.polygon == []


@requires_video
def test_disable_region_monitoring(window: MainWindow) -> None:
    window.open_video(SAMPLE_VIDEO)
    dialog = open_editor(window)
    dialog.enabled.setChecked(False)
    dialog.accept()
    window.apply_regions(dialog.result_config(), save_for_video=False)
    assert not window.run_config.regions.enabled
    assert window.region_panel.table.rowCount() == 0
    assert "Chưa có vùng ROI" in window.region_panel.summary.text()


@requires_video
def test_zones_outside_a_smaller_video_are_disabled(window: MainWindow, short_clip: Path) -> None:  # noqa: F811
    window.open_video(short_clip)                                        # 1920x1080, sample zones are 4K
    assert window.run_config.regions.zones == []
    wait_ms(20)
    assert "zone_a" in window.message_label.text() and "chưa có hiệu chỉnh" in window.message_label.text()


# ---------------------------------------- NV12: monitor regions while running
@requires_gpu
@requires_video
def test_run_with_regions_updates_panels_and_exports_csv(window: MainWindow) -> None:
    window.open_video(SAMPLE_VIDEO)
    window.handle_action(Action.START.value)
    assert not window.handle_action(Action.ROI.value)                     # blocked while running
    assert window.message_label.text() == msg.BLOCK_ROI

    wait_until(lambda: current_frame(window) >= 150)
    log = window.event_log
    assert any(event.event_type is EventType.ENTER for event in log.events)
    assert log.table.rowCount() == len(log.events)
    assert window.side_tabs.tabText(2).startswith("Sự kiện (")
    peak = int(window.region_panel.table.item(0, 2).text())
    entries = int(window.region_panel.table.item(0, 3).text())
    assert peak >= 1 and entries >= peak                                   # occupancy ≠ entry count
    assert "vùng · đang có" in window.region_panel.summary.text()

    log.filter.setCurrentIndex(log.filter.findData(EventType.ENTER))
    assert all(log.table.item(row, 1).text() == "ENTER" for row in range(log.table.rowCount()))

    window.handle_action(Action.STOP.value)
    wait_until(lambda: window.state is AppState.STOPPED)
    report = window.last_result.report
    regions = report["regions"]
    assert [zone["roi_id"] for zone in regions["zones"]] == ["zone_a", "zone_b"]
    assert len(regions["events"]) >= len(regions["sessions"]) > 0
    sessions_csv, events_csv = csv_paths(window.last_result.report_path)
    with open(events_csv, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(regions["events"])
    assert sessions_csv.is_file()
    assert "CSV" in window.message_label.text()

    exported = window.export_report(window.last_result.report_path.parent / "copy" / "again.json")
    assert all(p.is_file() for p in csv_paths(exported))
