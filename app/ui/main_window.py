"""Main window: layout, signal wiring and application state. No computer vision here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, QTimer
from PySide6.QtGui import QCloseEvent, QImage
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import AppConfig
from app.constants import messages as msg
from app.core.exceptions import TrafficMonitorError
from app.models.app_state import ALLOWED_STATES, Action, AppState, blocked_message, can_perform
from app.models.traffic_statistics import TrafficStatistics
from app.services.config_service import ConfigService
from app.services.report_service import ReportService
from app.services.video_service import VideoService
from app.ui.control_panel import ControlPanel
from app.ui.dashboard_widget import DashboardWidget
from app.ui.settings_dialog import SettingsDialog
from app.ui.styles import NOTIFY_COLORS, STATE_COLORS
from app.ui.video_widget import VideoWidget
from app.ui.video_worker import ModelWarmupWorker, RunResult, VideoWorker
from app.utils.device import describe_device, resolve_device
from app.utils.display import draw_calibration, resize_to_width, to_qimage
from app.utils.file_utils import INPUT_DIR, VIDEO_EXTENSIONS, resolve_path, to_display_path
from app.utils.geometry import line_inside_frame
from app.utils.logger import get_logger

logger = get_logger("main_window")

_DEFAULT_LINE_RATIO = 0.7
_THREAD_WAIT_MS = 10000


class MainWindow(QMainWindow):
    """Coordinates widgets, the worker thread and the application state machine."""

    def __init__(self, config: AppConfig, config_service: ConfigService, warm_up: bool = True) -> None:
        """Build the UI in the IDLE state and optionally preload the model on the GPU."""
        super().__init__()
        self.setWindowTitle(msg.APP_TITLE.title())
        self.resize(1480, 900)
        self._config = config
        self._run_config = config.copy()
        self._config_service = config_service
        self._video_service = VideoService()
        self._report_service = ReportService()
        self._state = AppState.IDLE
        self._video_path: Path | None = None
        self._preview_frame: np.ndarray | None = None
        self._video_fps = 25.0
        self._last_result: RunResult | None = None
        self._thread: QThread | None = None
        self._worker: VideoWorker | None = None
        self._warmup_thread: QThread | None = None
        self._warmup_worker: ModelWarmupWorker | None = None
        self._stopping = False
        self.dialogs_enabled = True  # automated tests turn modal error boxes off
        self._build_ui()
        self.control_panel.action_triggered.connect(self.handle_action)
        self._set_state(AppState.IDLE)
        self.dashboard.set_context(config.speed.limit_kmh, self._video_fps)
        self._show_device(describe_device(resolve_device(config.model.device)))
        if warm_up:
            QTimer.singleShot(0, self.warm_up_model)

    @property
    def is_warming_up(self) -> bool:
        """True while the model is being preloaded in the background."""
        return self._warmup_thread is not None

    def warm_up_model(self) -> None:
        """Load YOLO on the configured device in a background thread."""
        if self._warmup_thread is not None:
            return
        self.device_label.setText(f"{describe_device(resolve_device(self._config.model.device))} (đang khởi tạo)")
        self.notify(msg.WARMING_UP.format(model=resolve_path(self._config.model.path).name), "info")
        self._warmup_thread = QThread(self)
        self._warmup_worker = ModelWarmupWorker(self._config)
        self._warmup_worker.moveToThread(self._warmup_thread)
        self._warmup_thread.started.connect(self._warmup_worker.run)
        self._warmup_worker.ready.connect(self._on_warmup_ready)
        self._warmup_worker.failed.connect(lambda text: self.notify(text, "warning"))
        self._warmup_worker.done.connect(self._warmup_thread.quit)
        self._warmup_thread.finished.connect(self._on_warmup_finished)
        self._warmup_thread.start()

    def _on_warmup_ready(self, description: str, is_gpu: bool) -> None:
        self._show_device(description)
        if not is_gpu:
            self.notify(msg.GPU_FALLBACK, "warning")
        elif self._state is AppState.IDLE:
            self.notify(msg.MODEL_READY.format(device=description), "success")

    def _on_warmup_finished(self) -> None:
        if self._warmup_worker is not None:
            self._warmup_worker.deleteLater()
        if self._warmup_thread is not None:
            self._warmup_thread.deleteLater()
        self._warmup_worker = None
        self._warmup_thread = None

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> AppState:
        """Current application state."""
        return self._state

    @property
    def last_result(self) -> RunResult | None:
        """Result of the most recent finished run."""
        return self._last_result

    @property
    def video_path(self) -> Path | None:
        """Selected video."""
        return self._video_path

    @property
    def run_config(self) -> AppConfig:
        """Configuration used for the next run (includes per-video calibration)."""
        return self._run_config

    def handle_action(self, action_name: str) -> bool:
        """Entry point for every button. Returns True when the action was performed."""
        action = Action(action_name)
        if self._stopping or not can_perform(self._state, action):
            self.notify(msg.STOPPING if self._stopping else blocked_message(self._state, action), "warning")
            return False
        handlers = {
            Action.OPEN: self.open_video,
            Action.START: self.start_processing,
            Action.PAUSE: self.pause_processing,
            Action.RESUME: self.resume_processing,
            Action.STOP: self.stop_processing,
            Action.EXPORT: self.export_report,
            Action.SETTINGS: self.open_settings,
        }
        handlers[action]()
        return True

    # ---------------------------------------------------------------- actions
    def open_video(self, path: str | Path | None = None) -> bool:
        """Ask for (or accept) a video, validate it and show a calibrated preview."""
        if path is None:
            patterns = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
            chosen, _ = QFileDialog.getOpenFileName(self, "Open Video", str(INPUT_DIR), f"Video ({patterns})")
            if not chosen:
                return False
            path = chosen
        return self.load_video(Path(path))

    def load_video(self, path: Path) -> bool:
        """Validate a video file and switch to VIDEO_SELECTED, or ERROR on failure."""
        try:
            info, frame = self._video_service.probe(path)
        except TrafficMonitorError as exc:
            logger.error("Error: %s", exc)
            self._video_path = None
            self._preview_frame = None
            self.video_widget.clear_frame(f"ERROR\n{exc}")
            self.file_label.setText(Path(path).name)
            self._set_state(AppState.ERROR)
            self.show_error(str(exc))
            return False
        self._video_path = Path(path)
        self._preview_frame = frame
        self._video_fps = info.fps
        self._apply_calibration(info.width, info.height)
        self._last_result = None
        self.dashboard.reset()
        self.dashboard.set_context(self._run_config.speed.limit_kmh, info.fps)
        self.file_label.setText(self._video_path.name)
        self.frame_label.setText(f"frame 0/{info.total_frames}")
        self.progress.setValue(0)
        self._show_preview()
        self._set_state(AppState.VIDEO_SELECTED)
        self.notify(msg.VIDEO_SELECTED.format(name=self._video_path.name, width=info.width, height=info.height,
                                              fps=info.fps, frames=info.total_frames), "success")
        return True

    def start_processing(self) -> None:
        """Start (or restart) processing the selected video on a worker thread."""
        if self._video_path is None:
            self.notify(msg.BLOCK_START_NO_VIDEO, "warning")
            return
        self._wait_for_thread()
        self._last_result = None
        self.dashboard.reset()
        self.dashboard.set_context(self._run_config.speed.limit_kmh, self._video_fps)
        self.progress.setValue(0)
        self._thread = QThread(self)
        self._worker = VideoWorker(self._run_config, self._video_path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.stats_ready.connect(self._on_stats)
        self._worker.status_message.connect(lambda text: self.notify(text, "info"))
        self._worker.model_ready.connect(self._on_model_ready)
        self._worker.run_finished.connect(self._on_run_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._set_state(AppState.RUNNING)
        self._thread.start()

    def pause_processing(self) -> None:
        """Pause the running worker."""
        if self._worker is not None:
            self._worker.pause()
        self._set_state(AppState.PAUSED)
        self.notify(msg.PAUSED.format(frame=self._processed_frames()), "warning")

    def resume_processing(self) -> None:
        """Resume the paused worker."""
        if self._worker is not None:
            self._worker.resume()
        self._set_state(AppState.RUNNING)
        self.notify(msg.RESUMED, "info")

    def stop_processing(self) -> None:
        """Ask the worker to stop; state becomes STOPPED once results are exported."""
        if self._worker is None:
            return
        self._stopping = True
        self._worker.stop()
        self._refresh_controls()
        self.notify(msg.STOPPING, "info")

    def export_report(self, path: str | Path | None = None) -> Path | None:
        """Save the last report to a user-chosen path (Save As)."""
        if self._last_result is None:
            self.notify(msg.BLOCK_EXPORT, "warning")
            return None
        if path is None:
            default = resolve_path(self._run_config.output.report_path)
            chosen, _ = QFileDialog.getSaveFileName(self, "Export Report", str(default), "JSON (*.json)")
            if not chosen:
                return None
            path = chosen
        try:
            written = self._report_service.export(self._last_result.report, path)
        except TrafficMonitorError as exc:
            self.show_error(str(exc))
            return None
        self.notify(msg.REPORT_SAVED.format(path=to_display_path(written)), "success")
        return written

    def open_settings(self, dialog: SettingsDialog | None = None) -> bool:
        """Edit settings; persist them and optional per-video calibration."""
        dialog = dialog or SettingsDialog(
            self._run_config, self._config_service, self._preview_frame,
            self._video_path.name if self._video_path else None, self,
        )
        if not dialog.exec() or dialog.result_config() is None:
            return False
        return self.apply_settings(dialog.result_config(), dialog.save_calibration_requested())

    def apply_settings(self, config: AppConfig, save_calibration: bool) -> bool:
        """Persist a validated config and update the UI."""
        try:
            self._config_service.save(config)
            if save_calibration and self._video_path is not None:
                path = self._config_service.save_calibration(self._video_path, config)
                self.notify(msg.CALIBRATION_SAVED.format(path=to_display_path(path)), "success")
            else:
                self.notify(msg.SETTINGS_SAVED, "success")
        except TrafficMonitorError as exc:
            self.show_error(str(exc))
            return False
        model_changed = (config.model.path, config.model.device) != (self._config.model.path, self._config.model.device)
        self._config = config.copy()
        self._run_config = config.copy()
        self.dashboard.set_context(config.speed.limit_kmh, self._video_fps)
        self._show_device(describe_device(resolve_device(config.model.device)))
        if self._state in (AppState.VIDEO_SELECTED, AppState.STOPPED, AppState.COMPLETED):
            self._show_preview()
        if model_changed:
            self.warm_up_model()
        return True

    # ---------------------------------------------------------- worker slots
    def _on_frame(self, image: QImage) -> None:
        self.video_widget.show_frame(image)
        if self._worker is not None:
            self._worker.frame_displayed()

    def _on_stats(self, stats: TrafficStatistics) -> None:
        self.dashboard.update_statistics(stats)
        self.fps_label.setText(f"FPS {stats.fps:5.1f}")
        self.ms_label.setText(f"Processing {stats.processing_ms:6.1f} ms")
        self.frame_label.setText(f"frame {stats.frame_index}/{stats.total_frames}")
        if stats.total_frames:
            self.progress.setValue(round(stats.frame_index * 1000 / stats.total_frames))

    def _on_model_ready(self, description: str, is_gpu: bool) -> None:
        self._show_device(description)
        if is_gpu:
            self.notify(msg.MODEL_READY.format(device=description), "success")
        else:
            self.notify(msg.GPU_FALLBACK, "warning")

    def _on_run_finished(self, result: RunResult) -> None:
        self._last_result = result
        self._stopping = False
        self.dashboard.update_statistics(result.statistics)
        if result.completed:
            self.progress.setValue(1000)
        self._set_state(AppState.COMPLETED if result.completed else AppState.STOPPED)
        video = to_display_path(result.video_path) if result.video_path else "—"
        self.notify(msg.EXPORTED.format(video=video, report=to_display_path(result.report_path)), "success")

    def _on_error(self, message: str) -> None:
        self._stopping = False
        self._set_state(AppState.ERROR)
        self.show_error(message)

    def _on_thread_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    def _wait_for_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(_THREAD_WAIT_MS)
            self._on_thread_finished()

    # -------------------------------------------------------------- helpers
    def notify(self, text: str, level: str = "info") -> None:
        """Show a message in the status bar, coloured by level."""
        self.message_label.setStyleSheet(f"color: {NOTIFY_COLORS.get(level, NOTIFY_COLORS['info'])};")
        self.message_label.setText(text)
        self.message_label.setToolTip(text)
        if level == "warning":
            QTimer.singleShot(6000, lambda t=text: self.message_label.text() == t and self.message_label.setText(""))

    def show_error(self, text: str) -> None:
        """Notify and open a modal error box."""
        self.notify(text, "error")
        if self.dialogs_enabled:
            QMessageBox.critical(self, "Lỗi", text)

    def _apply_calibration(self, width: int, height: int) -> None:
        try:
            config, found = self._config_service.load_calibration(self._video_path, self._config)
        except TrafficMonitorError as exc:
            self.notify(str(exc), "warning")
            config, found = self._config.copy(), False
        if found:
            self.notify(msg.CALIBRATION_LOADED.format(name=self._video_path.name), "success")
        elif not line_inside_frame(config.counting_line.start, config.counting_line.end, width, height):
            y = round(height * _DEFAULT_LINE_RATIO)
            config.counting_line.start, config.counting_line.end = [0, y], [width, y]
            QTimer.singleShot(0, lambda: self.notify(msg.CALIBRATION_MISSING.format(name=self._video_path.name), "warning"))
        self._run_config = config

    def _show_preview(self) -> None:
        if self._preview_frame is None:
            return
        line = self._run_config.counting_line
        canvas = draw_calibration(self._preview_frame, line.start, line.end, self._run_config.perspective.source_points)
        self.video_widget.show_frame(to_qimage(resize_to_width(canvas, self._run_config.ui.display_max_width)))

    def _set_state(self, state: AppState) -> None:
        if state is not self._state:
            logger.info("State: %s -> %s", self._state.value, state.value)
        self._state = state
        self.state_label.setText(state.value)
        self.state_label.setStyleSheet(f"background: {STATE_COLORS[state.value]};")
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        allowed = {action: (not self._stopping) and self._state in states for action, states in ALLOWED_STATES.items()}
        reasons = {action: blocked_message(self._state, action) for action in Action}
        self.control_panel.set_allowed(allowed, reasons)

    def _processed_frames(self) -> int:
        return self._worker.frames_processed if self._worker is not None else 0

    def _show_device(self, description: str) -> None:
        self.device_label.setText(description)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        title = QLabel(msg.APP_TITLE)
        title.setObjectName("title")
        root.addWidget(title)
        body = QHBoxLayout()
        body.setSpacing(0)
        self.video_widget = VideoWidget()
        self.dashboard = DashboardWidget()
        body.addWidget(self.video_widget, 1)
        body.addWidget(self.dashboard)
        root.addLayout(body, 1)
        self.control_panel = ControlPanel()
        root.addWidget(self.control_panel)
        self.setCentralWidget(central)

        status = self.statusBar()
        self.state_label = QLabel()
        self.state_label.setObjectName("statePill")
        self.message_label = QLabel()
        self.file_label = QLabel("—")
        self.device_label = QLabel()
        self.fps_label = QLabel("FPS —")
        self.ms_label = QLabel("Processing — ms")
        self.frame_label = QLabel("frame 0/0")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(160)
        status.addWidget(self.state_label)
        status.addWidget(self.message_label, 1)
        for widget in (self.file_label, self.device_label, self.fps_label, self.ms_label, self.frame_label, self.progress):
            status.addPermanentWidget(widget)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        """Stop the worker cleanly before closing."""
        if self._worker is not None:
            self._worker.stop()
        for thread in (self._thread, self._warmup_thread):
            if thread is not None:
                thread.quit()
                thread.wait(_THREAD_WAIT_MS * 3)
        logger.info("Application closed")
        super().closeEvent(event)
