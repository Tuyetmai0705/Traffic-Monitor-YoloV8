"""Phases 9–10: VideoService, VideoProcessor control flow and the real GPU pipeline."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config.settings import AppConfig
from app.core.exceptions import VideoCodecError, VideoNotFoundError, VideoOpenError
from app.core.traffic_monitor import TrafficMonitor
from app.models.traffic_statistics import TrafficStatistics
from app.pipeline.frame_processor import FrameProcessor
from app.pipeline.video_processor import VideoProcessor
from app.services.video_service import VideoService
from tests.conftest import SAMPLE_VIDEO, requires_gpu, requires_video


def make_video(path: Path, frames: int = 30, size: tuple[int, int] = (96, 64)) -> Path:
    """Write a small synthetic MP4."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for index in range(frames):
        writer.write(np.full((size[1], size[0], 3), index * 5 % 255, dtype=np.uint8))
    writer.release()
    return path


class EchoProcessor:
    """Stand-in for FrameProcessor that returns the frame unchanged."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.frames: list[int] = []

    def process(self, frame: np.ndarray, index: int) -> tuple[np.ndarray, TrafficStatistics]:
        time.sleep(self.delay)
        self.frames.append(index)
        return frame, TrafficStatistics()

    def set_fps(self, fps: float) -> None:
        pass


def test_open_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoNotFoundError):
        VideoProcessor().open(tmp_path / "missing.mp4")


def test_open_corrupted_video_raises(tmp_path: Path) -> None:
    broken = tmp_path / "corrupted.mp4"
    broken.write_bytes(b"\x00not a real video" * 100)
    with pytest.raises((VideoOpenError, VideoCodecError)):
        VideoProcessor().open(broken)


def test_probe_returns_info_and_first_frame(tmp_path: Path) -> None:
    info, frame = VideoService().probe(make_video(tmp_path / "clip.mp4"))
    assert (info.width, info.height) == (96, 64)
    assert info.total_frames == 30
    assert frame.shape == (64, 96, 3)


def test_frames_are_downscaled_to_max_width(tmp_path: Path) -> None:
    processor = VideoProcessor(max_width=48)
    info = processor.open(make_video(tmp_path / "clip.mp4"))
    assert processor.scale == pytest.approx(0.5)
    assert (info.width, info.height) == (48, 32)
    assert processor.source_info.width == 96
    assert processor.read_frame().shape == (32, 48, 3)
    processor.release()


def test_scaled_geometry_follows_processing_scale(config: AppConfig) -> None:
    scaled = config.scaled_geometry(0.5)
    assert scaled.counting_line.start == [0, 750]
    assert scaled.counting_line.end == [1920, 750]
    assert scaled.perspective.source_points[0] == [626, 393.5]
    assert config.counting_line.start == [0, 1500]  # original untouched


def test_process_until_end_writes_output(tmp_path: Path) -> None:
    processor = VideoProcessor()
    processor.open(make_video(tmp_path / "clip.mp4"))
    seen: list[int] = []
    summary = processor.process(EchoProcessor(), lambda f, s, i: seen.append(i), tmp_path / "out" / "result.mp4")
    assert summary.completed
    assert summary.frames_processed == 30
    assert seen == list(range(1, 31))
    assert summary.output_video.is_file()
    capture = cv2.VideoCapture(str(summary.output_video))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    capture.release()


def test_pause_resume_stop_from_another_thread(tmp_path: Path) -> None:
    processor = VideoProcessor()
    processor.open(make_video(tmp_path / "clip.mp4", frames=200))
    echo = EchoProcessor(delay=0.01)
    result: dict = {}
    worker = threading.Thread(target=lambda: result.setdefault("summary", processor.process(echo, lambda *a: None, tmp_path / "o.mp4")))
    worker.start()
    time.sleep(0.2)
    processor.pause()
    assert processor.is_paused
    time.sleep(0.15)
    paused_at = len(echo.frames)
    time.sleep(0.3)
    assert len(echo.frames) <= paused_at + 1  # no progress while paused
    processor.resume()
    time.sleep(0.2)
    assert len(echo.frames) > paused_at + 1
    processor.stop()
    worker.join(timeout=5)
    summary = result["summary"]
    assert not summary.completed
    assert 0 < summary.frames_processed < 200
    assert summary.output_video.is_file()


def test_stop_while_paused_exits(tmp_path: Path) -> None:
    processor = VideoProcessor()
    processor.open(make_video(tmp_path / "clip.mp4", frames=100))
    processor.pause()
    worker = threading.Thread(target=processor.process, args=(EchoProcessor(), lambda *a: None, tmp_path / "o.mp4"))
    worker.start()
    time.sleep(0.2)
    processor.stop()
    worker.join(timeout=3)
    assert not worker.is_alive()


@pytest.mark.gpu
@requires_gpu
@requires_video
def test_full_gpu_pipeline_on_sample_video(config: AppConfig, tmp_path: Path) -> None:
    """Detect, track, count, estimate speed and annotate 175 real frames on the GPU."""
    processor = VideoProcessor(max_width=config.processing.max_width, queue_size=config.processing.queue_size)
    info = processor.open(SAMPLE_VIDEO)
    assert info.width == 1920
    monitor = TrafficMonitor.from_config(config.scaled_geometry(processor.scale), info)
    assert monitor.detector.device.startswith("cuda")
    frame_processor = FrameProcessor(monitor, config.scaled_geometry(processor.scale), info)
    collected: list[TrafficStatistics] = []

    def on_frame(frame: np.ndarray, stats: TrafficStatistics, index: int) -> None:
        assert frame.shape == (info.height, info.width, 3)
        collected.append(stats)
        if index >= 175:
            processor.stop()

    summary = processor.process(frame_processor, on_frame, tmp_path / "result.mp4")
    final = collected[-1]
    assert summary.frames_processed == 175
    assert final.total_in + final.total_out > 0, "vehicles should cross the counting line in 7 seconds"
    assert final.max_speed_kmh > 30, "speeds should be estimated on a highway"
    assert final.average_speed_kmh < 200
    assert final.fps > 0 and final.processing_ms > 0
    assert summary.output_video.stat().st_size > 100_000
    assert summary.average_fps > 5, f"GPU pipeline too slow: {summary.average_fps:.1f} fps"
    capture = cv2.VideoCapture(str(summary.output_video))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 175
    capture.release()
