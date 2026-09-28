"""End-to-end business flows driven through the real MainWindow (Qt offscreen, YOLO on GPU)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from app.config.settings import AppConfig
from app.constants import messages as msg
from app.models.app_state import Action, AppState
from app.services.config_service import ConfigService
from app.services.report_service import REQUIRED_KEYS
from app.ui.control_panel import ControlPanel
from app.ui.main_window import MainWindow
from app.ui.settings_dialog import SettingsDialog
from app.ui.styles import STYLESHEET
from tests.conftest import SAMPLE_VIDEO, requires_gpu, requires_video

pytestmark = [pytest.mark.ui]


@pytest.fixture(scope="module")
def qapp() -> Iterator[QApplication]:
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLESHEET)
    yield app


@pytest.fixture(scope="module")
def short_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """First 50 frames of the sample video at 1920x1080 (no calibration file)."""
    target = tmp_path_factory.mktemp("clips") / "short_clip.mp4"
    capture = cv2.VideoCapture(str(SAMPLE_VIDEO))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))
    for _ in range(50):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, (1920, 1080)))
    writer.release()
    capture.release()
    return target


@pytest.fixture()
def window(qapp: QApplication, tmp_path: Path) -> Iterator[MainWindow]:
    """MainWindow writing every output into a temp directory."""
    service = ConfigService(user_path=tmp_path / "user_config.json", calibration_dir=tmp_path / "calibration")
    config = service.load_defaults()
    config.output.video_path = str(tmp_path / "videos" / "traffic_result.mp4")
    config.output.report_path = str(tmp_path / "reports" / "traffic_report.json")
    win = MainWindow(config, service)
    win.dialogs_enabled = False
    win.show()
    yield win
    win.close()
    wait_ms(50)


def wait_ms(milliseconds: int) -> None:
    """Run the real Qt event loop (releases the GIL, unlike QTest.qWait)."""
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def wait_until(condition: Callable[[], bool], timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for condition")
        wait_ms(50)


def current_frame(win: MainWindow) -> int:
    return int(win.frame_label.text().split()[-1].split("/")[0])


# ----------------------------------------------------------- flow 1: guards
def test_initial_state_blocks_actions_with_messages(window: MainWindow) -> None:
    assert window.state is AppState.IDLE
    assert not window.video_widget.has_frame
    for action, message in (
        (Action.START, msg.BLOCK_START_NO_VIDEO),
        (Action.PAUSE, msg.BLOCK_PAUSE),
        (Action.RESUME, msg.BLOCK_RESUME),
        (Action.STOP, msg.BLOCK_STOP),
        (Action.EXPORT, msg.BLOCK_EXPORT),
    ):
        assert not window.handle_action(action.value)
        assert window.message_label.text() == message
        assert not window.control_panel.is_allowed(action)
    assert window.control_panel.is_allowed(Action.OPEN)
    assert window.control_panel.is_allowed(Action.SETTINGS)


def test_clicking_blocked_button_shows_reason(window: MainWindow) -> None:
    window.control_panel.buttons[Action.PAUSE].click()
    assert window.message_label.text() == msg.BLOCK_PAUSE
    assert window.state is AppState.IDLE


# ----------------------------------------------------- flow 2: open video
def test_open_missing_video_goes_to_error(window: MainWindow, tmp_path: Path) -> None:
    assert not window.open_video(tmp_path / "missing.mp4")
    assert window.state is AppState.ERROR
    assert "Không tìm thấy video" in window.message_label.text()
    assert not window.handle_action(Action.START.value)


def test_open_corrupted_video_goes_to_error(window: MainWindow, tmp_path: Path) -> None:
    broken = tmp_path / "corrupted.mp4"
    broken.write_bytes(b"\x00garbage" * 500)
    assert not window.open_video(broken)
    assert window.state is AppState.ERROR


@requires_video
def test_open_valid_video_after_error_recovers(window: MainWindow, tmp_path: Path) -> None:
    window.open_video(tmp_path / "missing.mp4")
    assert window.open_video(SAMPLE_VIDEO)
    assert window.state is AppState.VIDEO_SELECTED
    assert window.video_widget.has_frame                         # calibration preview
    assert window.file_label.text() == "vehicles.mp4"
    assert window.frame_label.text() == "frame 0/538"
    assert window.control_panel.is_allowed(Action.START)
    assert not window.control_panel.is_allowed(Action.EXPORT)


@requires_video
def test_uncalibrated_video_gets_default_line(window: MainWindow, short_clip: Path) -> None:
    window.open_video(short_clip)
    assert window.run_config.counting_line.start == [0, 756]    # 70% of 1080
    assert window.run_config.counting_line.end == [1920, 756]
    wait_ms(20)
    assert "chưa có hiệu chỉnh" in window.message_label.text()


# ------------------------------------ flow 3: start → pause → resume → stop
@requires_gpu
@requires_video
def test_run_pause_resume_stop_and_auto_export(window: MainWindow, tmp_path: Path) -> None:
    window.open_video(SAMPLE_VIDEO)
    assert window.handle_action(Action.START.value)
    assert window.state is AppState.RUNNING
    assert not window.handle_action(Action.SETTINGS.value)      # blocked while running
    assert not window.handle_action(Action.OPEN.value)
    assert not window.handle_action(Action.EXPORT.value)

    wait_until(lambda: current_frame(window) >= 40)
    assert "cuda" in window.device_label.text()
    assert window.video_widget.has_frame
    assert window.fps_label.text() != "FPS —"

    assert window.handle_action(Action.PAUSE.value)
    assert window.state is AppState.PAUSED
    wait_ms(600)
    paused_at = current_frame(window)
    wait_ms(1200)
    assert current_frame(window) <= paused_at + 1               # frozen while paused
    assert not window.handle_action(Action.PAUSE.value)

    assert window.handle_action(Action.RESUME.value)
    assert window.state is AppState.RUNNING
    wait_until(lambda: current_frame(window) >= paused_at + 20)

    assert window.handle_action(Action.STOP.value)
    wait_until(lambda: window.state is AppState.STOPPED)
    result = window.last_result
    assert result is not None and not result.completed
    assert result.frames_processed >= paused_at + 20
    assert result.report_path.is_file() and result.video_path.is_file()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert tuple(report)[:5] == REQUIRED_KEYS
    assert report["status"] == "stopped"
    assert "cuda" in report["device"]
    assert report["total_in"] + report["total_out"] == result.statistics.total_in + result.statistics.total_out
    capture = cv2.VideoCapture(str(result.video_path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == result.frames_processed
    capture.release()
    assert "Đã xuất video" in window.message_label.text()
    assert int(window.dashboard.total_in.text()) == result.statistics.total_in

    # flow 4: export report (Save As)
    assert window.control_panel.is_allowed(Action.EXPORT)
    exported = window.export_report(tmp_path / "copies" / "my_report.json")
    assert exported is not None and exported.is_file()
    assert json.loads(exported.read_text(encoding="utf-8"))["video"] == report["video"]


# --------------------------------------------- flow 5: run until COMPLETED
@requires_gpu
@requires_video
def test_run_to_completion_then_restart(window: MainWindow, short_clip: Path) -> None:
    window.open_video(short_clip)
    window.handle_action(Action.START.value)
    wait_until(lambda: window.state is AppState.COMPLETED)
    result = window.last_result
    assert result.completed and result.frames_processed == 50
    assert window.progress.value() == 1000
    assert json.loads(result.report_path.read_text(encoding="utf-8"))["status"] == "completed"

    assert window.handle_action(Action.START.value)              # re-run from COMPLETED
    assert window.last_result is None
    assert window.dashboard.total_in.text() == "0"
    wait_until(lambda: window.state is AppState.COMPLETED)
    assert window.last_result.frames_processed == 50


# --------------------------------------------------- flow 6: settings
@requires_video
def test_settings_save_and_calibration(window: MainWindow, tmp_path: Path) -> None:
    window.open_video(SAMPLE_VIDEO)
    dialog = SettingsDialog(window.run_config, window._config_service, window._preview_frame, "vehicles.mp4", window)
    assert dialog.preview.pixmap() is not None and not dialog.preview.pixmap().isNull()
    dialog.confidence.setValue(0.45)
    dialog.speed_limit.setValue(100)
    dialog.class_boxes[3].setChecked(False)
    dialog.line[1].setValue(1400)
    dialog.line[3].setValue(1400)
    dialog.accept()
    config = dialog.result_config()
    assert config is not None
    assert window.apply_settings(config, dialog.save_calibration_requested())
    assert window.run_config.model.confidence == pytest.approx(0.45)
    assert window.run_config.speed.limit_kmh == 100
    assert window.run_config.classes == [2, 5, 7]
    assert window.dashboard.speed_limit.text() == "100 km/h"
    saved = json.loads((tmp_path / "user_config.json").read_text(encoding="utf-8"))
    assert saved["speed"]["limit_kmh"] == 100
    calibration = json.loads((tmp_path / "calibration" / "vehicles.json").read_text(encoding="utf-8"))
    assert calibration["counting_line"]["start"] == [0, 1400]
    assert window.state is AppState.VIDEO_SELECTED


def test_settings_rejects_invalid_values(window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args: warnings.append(args[2])))
    dialog = SettingsDialog(window.run_config, window._config_service, None, None, window)
    for box in dialog.class_boxes.values():
        box.setChecked(False)
    dialog.accept()
    assert dialog.result_config() is None
    assert warnings and "classes" in warnings[0]
    dialog.class_boxes[2].setChecked(True)
    dialog.model_path.setText("models/missing.pt")
    dialog.accept()
    assert dialog.result_config() is None
    assert dialog.model_path.property("invalid") is True
    assert not dialog.save_calibration_requested()               # no video selected


# ------------------------------------------ flow 7: error during a run
@requires_video
def test_missing_model_during_start_goes_to_error(window: MainWindow) -> None:
    window.open_video(SAMPLE_VIDEO)
    window.run_config.model.path = "models/deleted.pt"
    window.handle_action(Action.START.value)
    wait_until(lambda: window.state is AppState.ERROR, timeout=30)
    assert "Không tìm thấy model" in window.message_label.text()
    assert window.control_panel.is_allowed(Action.OPEN)
    assert not window.control_panel.is_allowed(Action.EXPORT)


# ------------------------------------------ flow 8: close while running
@requires_gpu
@requires_video
def test_close_while_running_stops_worker(qapp: QApplication, tmp_path: Path) -> None:
    service = ConfigService(user_path=tmp_path / "u.json", calibration_dir=tmp_path / "c")
    config: AppConfig = service.load_defaults()
    config.output.video_path = str(tmp_path / "v.mp4")
    config.output.report_path = str(tmp_path / "r.json")
    win = MainWindow(config, service)
    win.dialogs_enabled = False
    win.open_video(SAMPLE_VIDEO)
    win.handle_action(Action.START.value)
    wait_until(lambda: current_frame(win) >= 5)
    started = time.monotonic()
    win.close()
    assert time.monotonic() - started < 10
    assert win._thread is None or not win._thread.isRunning()


def test_control_panel_emits_action_names(qapp: QApplication) -> None:
    panel = ControlPanel()
    received: list[str] = []
    panel.action_triggered.connect(received.append)
    for button in panel.buttons.values():
        button.click()
    assert received == [a.value for a in Action]
