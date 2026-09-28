"""Sequential frame-by-frame video processing with pause, resume and stop.

Decoding and encoding run in two helper threads connected by bounded queues,
so the GPU never waits on disk I/O and at most ``queue_size`` frames are held
in memory. Frames are still processed strictly in order.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from app.core.exceptions import VideoOpenError
from app.models.traffic_statistics import TrafficStatistics
from app.pipeline.frame_processor import FrameProcessor
from app.services.video_service import VideoService
from app.utils.logger import get_logger

logger = get_logger("video_processor")

FrameCallback = Callable[[np.ndarray, TrafficStatistics, int], None]
_WAIT_SECONDS = 0.1
_EMA = 0.1
_END = None  # queue sentinel


@dataclass(frozen=True)
class ProcessSummary:
    """Outcome of one ``process`` call."""

    completed: bool
    frames_processed: int
    output_video: Path | None
    average_fps: float


class VideoProcessor:
    """Read → process → write one frame at a time. Thread-safe pause/resume/stop."""

    def __init__(self, video_service: VideoService | None = None, max_width: int | None = None, queue_size: int = 8) -> None:
        """Create an idle processor; frames wider than ``max_width`` are downscaled."""
        self._service = video_service or VideoService()
        self._max_width = max_width
        self._queue_size = max(1, queue_size)
        self._capture: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._source_info: sv.VideoInfo | None = None
        self._info: sv.VideoInfo | None = None
        self._scale = 1.0
        self._path: Path | None = None
        self._frame_index = 0
        self._stop_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()

    @property
    def video_info(self) -> sv.VideoInfo | None:
        """Info of the frames actually processed (after downscaling)."""
        return self._info

    @property
    def source_info(self) -> sv.VideoInfo | None:
        """Info of the original video file."""
        return self._source_info

    @property
    def scale(self) -> float:
        """Processing scale relative to the source resolution (≤ 1)."""
        return self._scale

    @property
    def frame_index(self) -> int:
        """Number of frames processed so far."""
        return self._frame_index

    @property
    def is_paused(self) -> bool:
        """True while paused."""
        return not self._resume_event.is_set()

    def open(self, path: str | Path) -> sv.VideoInfo:
        """Open a video for sequential reading and return the processing info."""
        self.release()
        self._capture = self._service.open_capture(path)
        self._source_info = self._service.get_video_info(self._capture)
        width, height = self._source_info.resolution_wh
        self._scale = min(1.0, self._max_width / width) if self._max_width else 1.0
        self._info = replace(self._source_info, width=round(width * self._scale), height=round(height * self._scale))
        self._path = Path(path)
        self._frame_index = 0
        # Reset control flags here (not in process) so a pause/stop requested
        # while the model is still loading is honoured when processing begins.
        self._stop_event.clear()
        self._resume_event.set()
        logger.info(
            "Video loaded: %s %dx%d @%.1ffps, %d frames (processing %dx%d)",
            self._path.name, width, height, self._source_info.fps, self._source_info.total_frames,
            self._info.width, self._info.height,
        )
        return self._info

    def read_frame(self) -> np.ndarray | None:
        """Read and downscale the next frame, or None at the end of the video."""
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok:
            return None
        if self._scale < 1.0 and self._info is not None:
            frame = cv2.resize(frame, self._info.resolution_wh, interpolation=cv2.INTER_AREA)
        return frame

    def process(self, frame_processor: FrameProcessor, on_frame: FrameCallback, output_path: str | Path) -> ProcessSummary:
        """Process until the video ends or ``stop`` is called. Always releases resources."""
        if self._capture is None or self._info is None:
            raise VideoOpenError(self._path or "?")
        self._writer, written = self._service.create_writer(output_path, self._info.fps, self._info.resolution_wh)
        read_queue: queue.Queue = queue.Queue(maxsize=self._queue_size)
        write_queue: queue.Queue = queue.Queue(maxsize=self._queue_size)
        reader = threading.Thread(target=self._read_loop, args=(read_queue,), name="video-reader", daemon=True)
        writer = threading.Thread(target=self._write_loop, args=(write_queue,), name="video-writer", daemon=True)
        reader.start()
        writer.start()
        logger.info("Processing started: %s", self._path.name if self._path else "?")
        completed, fps, started = False, 0.0, time.perf_counter()
        try:
            while not self._stop_event.is_set():
                if not self._resume_event.wait(timeout=_WAIT_SECONDS):
                    continue
                tick = time.perf_counter()
                try:
                    frame = read_queue.get(timeout=_WAIT_SECONDS)
                except queue.Empty:
                    continue
                if frame is _END:
                    completed = True
                    break
                annotated, stats = frame_processor.process(frame, self._frame_index)
                write_queue.put(annotated)
                self._frame_index += 1
                instant = 1.0 / max(1e-6, time.perf_counter() - tick)
                fps = instant if fps == 0 else (1 - _EMA) * fps + _EMA * instant
                frame_processor.set_fps(fps)
                stats = replace(stats, fps=fps, frame_index=self._frame_index, total_frames=self._info.total_frames)
                on_frame(annotated, stats, self._frame_index)
        finally:
            self._stop_event.set()
            self._resume_event.set()
            self._drain(read_queue)
            reader.join()
            write_queue.put(_END)
            writer.join()
            self.release()
        elapsed = time.perf_counter() - started
        average = self._frame_index / elapsed if elapsed > 0 else 0.0
        if completed:
            logger.info("Processing completed: %d frames, avg %.1f fps", self._frame_index, average)
        else:
            logger.info("Processing stopped at frame %d", self._frame_index)
        return ProcessSummary(completed, self._frame_index, written, average)

    def pause(self) -> None:
        """Pause after the current frame."""
        if not self.is_paused:
            self._resume_event.clear()
            logger.info("Processing paused at frame %d", self._frame_index)

    def resume(self) -> None:
        """Resume a paused run."""
        if self.is_paused:
            self._resume_event.set()
            logger.info("Processing resumed at frame %d", self._frame_index)

    def stop(self) -> None:
        """Request the loop to finish; also releases a paused wait."""
        self._stop_event.set()
        self._resume_event.set()

    def release(self) -> None:
        """Close the capture and writer (the output video is finalised here)."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    # ------------------------------------------------------------ worker loops
    def _read_loop(self, frames: queue.Queue) -> None:
        """Decode frames ahead of processing until the end or a stop request."""
        while not self._stop_event.is_set():
            frame = self.read_frame()
            if not self._put(frames, frame):
                return
            if frame is None:
                return

    def _write_loop(self, frames: queue.Queue) -> None:
        """Encode annotated frames in order until the end sentinel."""
        while True:
            frame = frames.get()
            if frame is _END:
                return
            if self._writer is not None:
                self._writer.write(frame)

    def _put(self, frames: queue.Queue, item: np.ndarray | None) -> bool:
        """Blocking put that gives up when processing stops."""
        while not self._stop_event.is_set():
            try:
                frames.put(item, timeout=_WAIT_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def _drain(frames: queue.Queue) -> None:
        while True:
            try:
                frames.get_nowait()
            except queue.Empty:
                return
