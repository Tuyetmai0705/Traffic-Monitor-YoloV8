"""Shared fixtures: configuration and synthetic ``sv.Detections``."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import supervision as sv
import app.ui  # noqa: F401  (preloads torch and patches PySide6 imports, see qt_compat)

from app.config.settings import AppConfig
from app.services.config_service import ConfigService
from app.utils.file_utils import PROJECT_ROOT

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SAMPLE_VIDEO = PROJECT_ROOT / "data" / "input" / "vehicles.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"


def make_detections(
    boxes: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    confidences: Sequence[float] | None = None,
    tracker_ids: Sequence[int] | None = None,
) -> sv.Detections:
    """Build ``sv.Detections`` from plain lists."""
    count = len(boxes)
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(count, 4),
        confidence=np.asarray(confidences if confidences is not None else [0.9] * count, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=int),
        tracker_id=None if tracker_ids is None else np.asarray(tracker_ids, dtype=int),
    )


@pytest.fixture()
def config_service(tmp_path: Path) -> ConfigService:
    """ConfigService writing user config and calibration into a temp dir."""
    return ConfigService(user_path=tmp_path / "user_config.json", calibration_dir=tmp_path / "calibration")


@pytest.fixture()
def config(config_service: ConfigService) -> AppConfig:
    """Validated default configuration."""
    return config_service.load_defaults()


def gpu_available() -> bool:
    """Return True when CUDA can be used."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(not (gpu_available() and MODEL_PATH.is_file()), reason="cần GPU CUDA và models/yolov8n.pt")
requires_video = pytest.mark.skipif(not SAMPLE_VIDEO.is_file(), reason="cần data/input/vehicles.mp4")
