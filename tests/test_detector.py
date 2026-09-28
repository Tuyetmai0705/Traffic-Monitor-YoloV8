"""Phase 3: VehicleDetector initialisation, filtering, errors and real GPU inference."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import supervision as sv

from app.config.settings import AppConfig
from app.core.detector import VehicleDetector
from app.utils.device import available_devices, resolve_device
from app.core.exceptions import InferenceError, ModelNotFoundError
from tests.conftest import SAMPLE_VIDEO, make_detections, requires_gpu, requires_video


def test_init_does_not_load_model(config: AppConfig) -> None:
    detector = VehicleDetector(config.model, config.classes)
    assert not detector.is_loaded
    assert detector.confidence == pytest.approx(0.30)
    assert detector.get_statistics()["frames_processed"] == 0


def test_filter_keeps_confident_vehicles_only(config: AppConfig) -> None:
    detector = VehicleDetector(config.model, config.classes)
    detections = make_detections(
        boxes=[[0, 0, 10, 10]] * 5,
        class_ids=[2, 2, 0, 7, 9],  # car, car, person, truck, traffic light
        confidences=[0.25, 0.30, 0.95, 0.80, 0.90],
    )
    filtered = detector.filter_detections(detections)
    assert filtered.class_id.tolist() == [2, 7]
    assert filtered.confidence.tolist() == pytest.approx([0.30, 0.80])


def test_filter_respects_selected_classes(config: AppConfig) -> None:
    detector = VehicleDetector(config.model, [3])
    detections = make_detections([[0, 0, 5, 5]] * 3, class_ids=[2, 3, 5])
    assert detector.filter_detections(detections).class_id.tolist() == [3]


def test_get_class_name() -> None:
    assert VehicleDetector.get_class_name(2) == "car"
    assert VehicleDetector.get_class_name(3) == "motorcycle"
    assert VehicleDetector.get_class_name(5) == "bus"
    assert VehicleDetector.get_class_name(7) == "truck"
    assert VehicleDetector.get_class_name(0) == "unknown"


def test_missing_model_raises(config: AppConfig) -> None:
    model_config = config.copy().model
    model_config.path = "models/does_not_exist.pt"
    with pytest.raises(ModelNotFoundError):
        VehicleDetector(model_config, config.classes).load_model()


def test_detect_before_load_raises(config: AppConfig) -> None:
    with pytest.raises(InferenceError):
        VehicleDetector(config.model, config.classes).detect(np.zeros((64, 64, 3), np.uint8))


def test_detect_uses_mocked_model_and_filters(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    detector = VehicleDetector(config.model, config.classes)
    raw = make_detections([[0, 0, 10, 10]] * 3, class_ids=[2, 0, 5], confidences=[0.9, 0.9, 0.1])
    detector._model = SimpleNamespace(predict=lambda frame, **kwargs: [object()])
    monkeypatch.setattr(sv.Detections, "from_ultralytics", staticmethod(lambda result: raw))
    result = detector.detect(np.zeros((64, 64, 3), np.uint8))
    assert result.class_id.tolist() == [2]
    assert detector.get_statistics()["total_detections"] == 1


def test_inference_error_is_wrapped(config: AppConfig) -> None:
    detector = VehicleDetector(config.model, config.classes)

    def broken(frame, **kwargs):
        raise RuntimeError("CUDA out of memory")

    detector._model = SimpleNamespace(predict=broken)
    with pytest.raises(InferenceError, match="CUDA out of memory"):
        detector.detect(np.zeros((64, 64, 3), np.uint8))


def test_cpu_device_is_kept() -> None:
    assert resolve_device("cpu") == "cpu"
    assert available_devices()[-1] == ("cpu", "CPU")


def test_unknown_gpu_index_falls_back_to_cpu() -> None:
    assert resolve_device("cuda:99") == "cpu"


@pytest.mark.gpu
@requires_gpu
@requires_video
def test_yolov8n_runs_on_gpu_and_finds_vehicles(config: AppConfig) -> None:
    detector = VehicleDetector(config.model, config.classes)
    detector.load_model()
    assert detector.device.startswith("cuda")
    assert detector.model_name == "yolov8n.pt"
    capture = cv2.VideoCapture(str(SAMPLE_VIDEO))
    ok, frame = capture.read()
    capture.release()
    assert ok
    detections = detector.detect(frame)
    assert len(detections) >= 3
    assert set(detections.class_id.tolist()) <= {2, 3, 5, 7}
    assert float(detections.confidence.min()) >= 0.30
