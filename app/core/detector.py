"""YOLO vehicle detection returning ``sv.Detections``."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from app.config.settings import ModelConfig
from app.constants.classes import VEHICLE_CLASSES
from app.core.exceptions import InferenceError, ModelLoadError, ModelNotFoundError
from app.utils.device import describe_device, resolve_device
from app.utils.file_utils import resolve_path
from app.utils.logger import get_logger

logger = get_logger("detector")

_WARMUP_SHAPE = (320, 320, 3)

# Loading YOLO the first time pulls in ~1000 lazily imported modules and creates
# the CUDA context (10+ s on a laptop GPU). One loaded model is therefore kept
# per process and shared by successive runs; the lock makes a background
# warm-up and a user-started run safe to overlap.
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, int, str], Any] = {}


def _get_or_load_model(path: Path, device: str, warmup_kwargs: dict[str, Any]) -> tuple[Any, bool]:
    """Return (model, was_cached) for weights on a device, loading at most once."""
    key = (str(path), path.stat().st_mtime_ns, device)
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            return model, True
        from ultralytics import YOLO

        model = YOLO(str(path))
        model.predict(np.zeros(_WARMUP_SHAPE, dtype=np.uint8), **warmup_kwargs)
        _MODEL_CACHE.clear()  # keep a single model in GPU memory
        _MODEL_CACHE[key] = model
        return model, False


class VehicleDetector:
    """Run YOLO on frames and keep only confident vehicle detections."""

    def __init__(self, config: ModelConfig, class_ids: Sequence[int]) -> None:
        """Store parameters. The model is loaded later by ``load_model``."""
        self._config = config
        self._class_ids = np.array(sorted(set(class_ids)), dtype=int)
        self._model: Any | None = None
        self._device = "cpu"
        self._frames_processed = 0
        self._total_detections = 0
        self._inference_ms_total = 0.0

    @property
    def is_loaded(self) -> bool:
        """True once ``load_model`` succeeded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Device actually used for inference (``cuda:N`` or ``cpu``)."""
        return self._device

    @property
    def confidence(self) -> float:
        """Confidence threshold used by ``filter_detections``."""
        return self._config.confidence

    @property
    def model_name(self) -> str:
        """File name of the YOLO weights."""
        return resolve_path(self._config.path).name

    def load_model(self) -> None:
        """Load YOLO weights onto the configured device and warm it up."""
        path = resolve_path(self._config.path)
        if not path.is_file():
            raise ModelNotFoundError(path)
        self._device = resolve_device(self._config.device)
        started = time.perf_counter()
        try:
            model, cached = _get_or_load_model(path, self._device, self._predict_kwargs())
        except Exception as exc:  # ultralytics raises many unrelated types
            logger.exception("Model load failed: %s", path)
            raise ModelLoadError(path, str(exc)) from exc
        self._model = model
        logger.info(
            "Model loaded: %s on %s in %.0f ms%s",
            path.name,
            describe_device(self._device),
            (time.perf_counter() - started) * 1000,
            " (cached)" if cached else "",
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run inference on one BGR frame and return filtered vehicle detections."""
        if self._model is None:
            raise InferenceError("model chưa được tải")
        started = time.perf_counter()
        try:
            result = self._model.predict(frame, **self._predict_kwargs())[0]
            detections = sv.Detections.from_ultralytics(result)
        except Exception as exc:
            logger.exception("YOLO inference failed")
            raise InferenceError(str(exc)) from exc
        filtered = self.filter_detections(detections)
        self._frames_processed += 1
        self._total_detections += len(filtered)
        self._inference_ms_total += (time.perf_counter() - started) * 1000
        return filtered

    def filter_detections(self, detections: sv.Detections) -> sv.Detections:
        """Keep detections with confidence ≥ threshold and a selected vehicle class."""
        if len(detections) == 0:
            return detections
        confidence = detections.confidence if detections.confidence is not None else np.ones(len(detections))
        mask = (confidence >= self._config.confidence) & np.isin(detections.class_id, self._class_ids)
        return detections[mask]

    @staticmethod
    def get_class_name(class_id: int) -> str:
        """Return the vehicle name for a COCO class id, or ``unknown``."""
        return VEHICLE_CLASSES.get(int(class_id), "unknown")

    def get_statistics(self) -> dict[str, float]:
        """Return detector counters: frames, detections and mean inference time."""
        frames = self._frames_processed
        return {
            "frames_processed": frames,
            "total_detections": self._total_detections,
            "avg_inference_ms": self._inference_ms_total / frames if frames else 0.0,
        }

    def _predict_kwargs(self) -> dict[str, Any]:
        return {
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "imgsz": self._config.imgsz,
            "device": self._device,
            "half": bool(self._config.half and self._device.startswith("cuda")),
            "verbose": False,
        }
