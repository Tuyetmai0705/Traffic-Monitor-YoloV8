"""Phase 2: configuration loading, validation, persistence and calibration."""

from __future__ import annotations

import json

import pytest

from app.config.settings import AppConfig
from app.core.exceptions import InvalidConfigError
from app.services.config_service import ConfigService


def test_default_config_is_valid_and_uses_gpu_yolov8n(config: AppConfig) -> None:
    assert config.model.path == "models/yolov8n.pt"
    assert config.model.device.startswith("cuda")
    assert config.model.confidence == pytest.approx(0.30)
    assert config.speed.limit_kmh == 80
    assert config.classes == [2, 3, 5, 7]


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda c: setattr(c.model, "confidence", 0.99), "model.confidence"),
        (lambda c: setattr(c.model, "path", "models/missing.pt"), "model.path"),
        (lambda c: setattr(c.model, "device", "tpu"), "model.device"),
        (lambda c: setattr(c, "classes", []), "classes"),
        (lambda c: setattr(c, "classes", [0]), "classes"),
        (lambda c: setattr(c.speed, "limit_kmh", 0), "speed.limit_kmh"),
        (lambda c: setattr(c.perspective, "road_width_m", 0), "perspective.road_width_m"),
        (lambda c: setattr(c.perspective, "source_points", [[0, 0], [10, 0], [20, 0], [0, 10]]), "perspective.source_points"),
        (lambda c: setattr(c.counting_line, "end", c.counting_line.start), "counting_line"),
        (lambda c: setattr(c.output, "video_path", "out.txt"), "output.video_path"),
        (lambda c: setattr(c.output, "report_path", "out.csv"), "output.report_path"),
    ],
)
def test_validate_rejects_invalid_values(config_service: ConfigService, config: AppConfig, mutate, field: str) -> None:
    broken = config.copy()
    mutate(broken)
    with pytest.raises(InvalidConfigError) as info:
        config_service.validate(broken)
    assert info.value.field == field


def test_save_then_load_round_trip(config_service: ConfigService, config: AppConfig) -> None:
    edited = config.copy()
    edited.model.confidence = 0.45
    edited.speed.limit_kmh = 60
    config_service.save(edited)
    loaded = config_service.load()
    assert loaded.model.confidence == pytest.approx(0.45)
    assert loaded.speed.limit_kmh == 60


def test_invalid_user_config_falls_back_to_defaults(config_service: ConfigService) -> None:
    config_service.user_path.write_text(json.dumps({"speed": {"limit_kmh": -5}}), encoding="utf-8")
    loaded = config_service.load()
    assert loaded.speed.limit_kmh == 80


def test_calibration_round_trip(config_service: ConfigService, config: AppConfig) -> None:
    video = "data/input/my_road.mp4"
    assert config_service.load_calibration(video, config) == (config, False)
    edited = config.copy()
    edited.counting_line.start = [0, 400]
    edited.counting_line.end = [1280, 400]
    path = config_service.save_calibration(video, edited)
    assert path.name == "my_road.json"
    calibrated, found = config_service.load_calibration(video, config)
    assert found
    assert calibrated.counting_line.start == [0, 400]
    assert calibrated.model.confidence == config.model.confidence


def test_missing_section_raises_invalid_config() -> None:
    with pytest.raises(ValueError):
        AppConfig.from_dict({"classes": [2]})
