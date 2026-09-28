"""Background worker running the CV pipeline inside a QThread."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from app.config.settings import AppConfig
from app.constants import messages as msg
from app.core.detector import VehicleDetector
from app.core.exceptions import TrafficMonitorError
from app.core.traffic_monitor import TrafficMonitor
from app.models.traffic_statistics import TrafficStatistics
from app.pipeline.frame_processor import FrameProcessor
from app.pipeline.video_processor import VideoProcessor
from app.services.report_service import ReportMeta, ReportService
from app.utils.device import describe_device
from app.utils.display import resize_to_width, to_qimage
from app.utils.file_utils import resolve_path, to_display_path
from app.utils.logger import get_logger

logger = get_logger("video_worker")


class ModelWarmupWorker(QObject):
    """Loads YOLO onto the GPU in the background when the app starts."""

    ready = Signal(str, bool)   # device description, is_gpu
    failed = Signal(str)
    done = Signal()

    def __init__(self, config: AppConfig) -> None:
        """Warm up the model described by ``config``."""
        super().__init__()
        self._config = config.copy()

    @Slot()
    def run(self) -> None:
        """Load the model once so the first Start is fast. Never raises."""
        try:
            detector = VehicleDetector(self._config.model, self._config.classes)
            detector.load_model()
            self.ready.emit(describe_device(detector.device), detector.device.startswith("cuda"))
        except TrafficMonitorError as exc:
            logger.warning("Model warm-up failed: %s", exc)
            self.failed.emit(str(exc))
        except Exception:
            logger.exception("Error: model warm-up failed")
            self.failed.emit(msg.UNEXPECTED_ERROR)
        finally:
            self.done.emit()


@dataclass(frozen=True)
class RunResult:
    """Everything the UI needs after a run ends."""

    completed: bool
    statistics: TrafficStatistics
    report: dict[str, Any]
    report_path: Path
    video_path: Path | None
    frames_processed: int


class VideoWorker(QObject):
    """Owns one processing run. Lives in a QThread; control methods are thread-safe."""

    frame_ready = Signal(QImage)
    stats_ready = Signal(object)          # TrafficStatistics
    status_message = Signal(str)
    model_ready = Signal(str, bool)       # device description, is_gpu
    run_finished = Signal(object)         # RunResult
    error_occurred = Signal(str)
    done = Signal()

    def __init__(self, config: AppConfig, video_path: Path) -> None:
        """Prepare a run for ``video_path`` with a snapshot of ``config``."""
        super().__init__()
        self._config = config.copy()
        self._video_path = Path(video_path)
        self._processor = VideoProcessor(max_width=config.processing.max_width, queue_size=config.processing.queue_size)
        self._reports = ReportService()
        self._frame_processor: FrameProcessor | None = None
        self._device = "cpu"
        self._model_name = resolve_path(config.model.path).name
        self._frame_pending = False
        self._last_stats_emit = 0.0
        self._last_stats: TrafficStatistics | None = None

    # ------------------------------------------------------ control (UI thread)
    @property
    def frames_processed(self) -> int:
        """Frames processed so far in this run."""
        return self._processor.frame_index

    def pause(self) -> None:
        """Pause after the current frame."""
        self._processor.pause()

    def resume(self) -> None:
        """Resume processing."""
        self._processor.resume()

    def stop(self) -> None:
        """Stop processing; results are exported by the worker thread."""
        self._processor.stop()

    def frame_displayed(self) -> None:
        """Called by the UI when the last frame was painted, allowing the next one."""
        self._frame_pending = False

    # ------------------------------------------------------ run (worker thread)
    @Slot()
    def run(self) -> None:
        """Open, load the model, process and export. Never raises."""
        try:
            self._run()
        except TrafficMonitorError as exc:
            logger.error("Error: %s", exc)
            self._export_partial()
            self.error_occurred.emit(str(exc))
        except Exception:
            logger.exception("Error: unexpected failure")
            self._export_partial()
            self.error_occurred.emit(msg.UNEXPECTED_ERROR)
        finally:
            self._processor.release()
            self._frame_processor = None
            self._free_gpu_memory()
            self.done.emit()

    def _run(self) -> None:
        info = self._processor.open(self._video_path)
        run_config = self._config.scaled_geometry(self._processor.scale)
        self.status_message.emit(msg.LOADING_MODEL.format(model=self._model_name, device=self._config.model.device))
        monitor = TrafficMonitor.from_config(run_config, info)
        self._device = monitor.detector.device
        self.model_ready.emit(describe_device(self._device), self._device.startswith("cuda"))
        self._frame_processor = FrameProcessor(monitor, run_config, info)
        self.status_message.emit(msg.PROCESSING)
        summary = self._processor.process(self._frame_processor, self._on_frame, resolve_path(self._config.output.video_path))
        stats = self._last_stats or self._frame_processor.last_statistics
        self.stats_ready.emit(stats)
        report, report_path = self._export(stats, summary.frames_processed, summary.completed, summary.output_video)
        self.run_finished.emit(RunResult(summary.completed, stats, report, report_path, summary.output_video, summary.frames_processed))

    def _on_frame(self, annotated: np.ndarray, stats: TrafficStatistics, frame_number: int) -> None:
        self._last_stats = stats
        if not self._frame_pending:
            self._frame_pending = True
            self.frame_ready.emit(to_qimage(resize_to_width(annotated, self._config.ui.display_max_width)))
        now = time.monotonic()
        if (now - self._last_stats_emit) * 1000 >= self._config.ui.dashboard_refresh_ms:
            self._last_stats_emit = now
            self.stats_ready.emit(stats)

    def _export(
        self, stats: TrafficStatistics, frames: int, completed: bool, video: Path | None
    ) -> tuple[dict[str, Any], Path]:
        info = self._processor.video_info
        meta = ReportMeta(
            video=to_display_path(self._video_path),
            frames_processed=frames,
            fps=info.fps if info else 0.0,
            model=self._model_name,
            device=describe_device(self._device),
            confidence=self._config.model.confidence,
            speed_limit_kmh=self._config.speed.limit_kmh,
            output_video=to_display_path(video) if video else "",
            completed=completed,
        )
        report = self._reports.build_report(stats, meta)
        return report, self._reports.export(report, resolve_path(self._config.output.report_path))

    def _export_partial(self) -> None:
        """Save whatever was processed before an error (spec: results up to the failing frame)."""
        if self._frame_processor is None or self._processor.frame_index == 0:
            return
        try:
            stats = self._last_stats or self._frame_processor.last_statistics
            self._processor.release()
            self._export(stats, self._processor.frame_index, False, resolve_path(self._config.output.video_path))
        except Exception:
            logger.exception("Error: partial report export failed")

    @staticmethod
    def _free_gpu_memory() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
