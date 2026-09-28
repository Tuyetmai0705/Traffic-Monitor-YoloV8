"""Load, validate and save configuration and per-video calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config.settings import DEFAULT_CONFIG_PATH, USER_CONFIG_PATH, AppConfig, deep_merge
from app.constants.classes import VEHICLE_CLASSES
from app.core.exceptions import InvalidConfigError, OutputPathError
from app.utils.file_utils import CALIBRATION_DIR, ensure_parent_dir, resolve_path
from app.utils.geometry import validate_line, validate_quadrilateral
from app.utils.logger import get_logger

logger = get_logger("config_service")

CALIBRATION_KEYS = ("counting_line", "perspective")


class ConfigService:
    """Single entry point for reading and writing configuration files."""

    def __init__(
        self,
        default_path: Path = DEFAULT_CONFIG_PATH,
        user_path: Path = USER_CONFIG_PATH,
        calibration_dir: Path = CALIBRATION_DIR,
    ) -> None:
        self._default_path = Path(default_path)
        self._user_path = Path(user_path)
        self._calibration_dir = Path(calibration_dir)

    @property
    def user_path(self) -> Path:
        """Location where Settings are persisted."""
        return self._user_path

    def load_defaults(self) -> AppConfig:
        """Load and validate ``default_config.json`` only."""
        config = self._from_dict(self._read_json(self._default_path))
        self.validate(config)
        return config

    def load(self) -> AppConfig:
        """Load defaults merged with the user config. Falls back to defaults if the user file is invalid."""
        defaults = self._read_json(self._default_path)
        if not self._user_path.is_file():
            config = self._from_dict(defaults)
            self.validate(config)
            return config
        try:
            config = self._from_dict(deep_merge(defaults, self._read_json(self._user_path)))
            self.validate(config)
            return config
        except InvalidConfigError as exc:
            logger.warning("User config ignored (%s); using defaults", exc)
            config = self._from_dict(defaults)
            self.validate(config)
            return config

    def save(self, config: AppConfig) -> Path:
        """Validate then persist the config as the user config."""
        self.validate(config)
        self._write_json(self._user_path, config.to_dict())
        logger.info("Config saved: %s", self._user_path)
        return self._user_path

    def validate(self, config: AppConfig) -> None:
        """Raise InvalidConfigError naming the first invalid field."""
        self._validate_model(config)
        self._validate_classes(config)
        self._validate_tracker(config)
        self._validate_geometry(config)
        self._validate_speed(config)
        self._validate_output(config)
        self._validate_ui(config)

    def calibration_path(self, video_path: str | Path) -> Path:
        """Return the calibration file used for a video."""
        return self._calibration_dir / f"{Path(video_path).stem}.json"

    def load_calibration(self, video_path: str | Path, config: AppConfig) -> tuple[AppConfig, bool]:
        """Return ``config`` with the video's calibration applied and whether one was found."""
        path = self.calibration_path(video_path)
        if not path.is_file():
            return config, False
        data = self._read_json(path)
        override = {key: data[key] for key in CALIBRATION_KEYS if key in data}
        calibrated = self._from_dict(deep_merge(config.to_dict(), override))
        self.validate(calibrated)
        logger.info("Calibration loaded: %s", path)
        return calibrated, True

    def save_calibration(self, video_path: str | Path, config: AppConfig) -> Path:
        """Persist the counting line and perspective of ``config`` for one video."""
        self.validate(config)
        path = self.calibration_path(video_path)
        data = {key: config.to_dict()[key] for key in CALIBRATION_KEYS}
        self._write_json(path, data)
        logger.info("Calibration saved: %s", path)
        return path

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            raise InvalidConfigError(str(path), "không tìm thấy file cấu hình") from exc
        except json.JSONDecodeError as exc:
            raise InvalidConfigError(str(path), f"JSON sai cú pháp ở dòng {exc.lineno}") from exc
        if not isinstance(data, dict):
            raise InvalidConfigError(str(path), "nội dung phải là object JSON")
        return data

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        try:
            target = ensure_parent_dir(path)
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=4, ensure_ascii=False)
        except OSError as exc:
            raise OutputPathError(path, str(exc)) from exc

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> AppConfig:
        try:
            return AppConfig.from_dict(data)
        except (TypeError, ValueError, KeyError) as exc:
            raise InvalidConfigError("config", str(exc)) from exc

    @staticmethod
    def _require(condition: bool, field: str, reason: str) -> None:
        if not condition:
            raise InvalidConfigError(field, reason)

    def _validate_model(self, config: AppConfig) -> None:
        model = config.model
        self._require(str(model.path).lower().endswith(".pt"), "model.path", "phải là file .pt")
        self._require(resolve_path(model.path).is_file(), "model.path", f"không tìm thấy {model.path}")
        self._require(0.05 <= model.confidence <= 0.95, "model.confidence", "phải trong khoảng 0.05 – 0.95")
        self._require(0.0 < model.iou <= 1.0, "model.iou", "phải trong khoảng (0, 1]")
        device = str(model.device).lower()
        self._require(device == "cpu" or device.startswith("cuda"), "model.device", "chỉ nhận 'cpu' hoặc 'cuda:N'")
        self._require(isinstance(model.half, bool), "model.half", "phải là true/false")
        self._require(160 <= model.imgsz <= 1920 and model.imgsz % 32 == 0, "model.imgsz", "bội số của 32, từ 160 đến 1920")

    def _validate_classes(self, config: AppConfig) -> None:
        self._require(len(config.classes) > 0, "classes", "chọn ít nhất một loại xe")
        unknown = set(config.classes) - VEHICLE_CLASSES.keys()
        self._require(not unknown, "classes", f"class id không hỗ trợ: {sorted(unknown)}")

    def _validate_tracker(self, config: AppConfig) -> None:
        tracker = config.tracker
        self._require(0.0 < tracker.track_activation_threshold < 1.0, "tracker.track_activation_threshold", "trong khoảng (0, 1)")
        self._require(tracker.lost_track_buffer >= 1, "tracker.lost_track_buffer", "phải ≥ 1")
        self._require(0.0 < tracker.minimum_matching_threshold <= 1.0, "tracker.minimum_matching_threshold", "trong khoảng (0, 1]")
        self._require(tracker.minimum_consecutive_frames >= 1, "tracker.minimum_consecutive_frames", "phải ≥ 1")
        self._require(tracker.smoother_length >= 1, "tracker.smoother_length", "phải ≥ 1")

    def _validate_geometry(self, config: AppConfig) -> None:
        try:
            validate_line(config.counting_line.start, config.counting_line.end)
        except ValueError as exc:
            raise InvalidConfigError("counting_line", str(exc)) from exc
        try:
            validate_quadrilateral(config.perspective.source_points)
        except ValueError as exc:
            raise InvalidConfigError("perspective.source_points", str(exc)) from exc
        self._require(config.perspective.road_width_m > 0, "perspective.road_width_m", "phải > 0")
        self._require(config.perspective.road_length_m > 0, "perspective.road_length_m", "phải > 0")

    def _validate_speed(self, config: AppConfig) -> None:
        speed = config.speed
        self._require(10 <= speed.limit_kmh <= 200, "speed.limit_kmh", "phải từ 10 đến 200 km/h")
        self._require(0.2 <= speed.window_seconds <= 5.0, "speed.window_seconds", "phải từ 0.2 đến 5 giây")
        self._require(speed.min_samples >= 2, "speed.min_samples", "phải ≥ 2")
        self._require(speed.stale_seconds > 0, "speed.stale_seconds", "phải > 0")

    def _validate_output(self, config: AppConfig) -> None:
        output = config.output
        self._require(Path(output.video_path).suffix.lower() in (".mp4", ".avi"), "output.video_path", "đuôi phải là .mp4 hoặc .avi")
        self._require(Path(output.report_path).suffix.lower() == ".json", "output.report_path", "đuôi phải là .json")
        self._require(Path(output.log_path).suffix.lower() == ".log", "output.log_path", "đuôi phải là .log")

    def _validate_ui(self, config: AppConfig) -> None:
        self._require(50 <= config.ui.dashboard_refresh_ms <= 2000, "ui.dashboard_refresh_ms", "phải từ 50 đến 2000 ms")
        self._require(320 <= config.ui.display_max_width <= 3840, "ui.display_max_width", "phải từ 320 đến 3840 px")
        self._require(640 <= config.processing.max_width <= 7680, "processing.max_width", "phải từ 640 đến 7680 px")
        self._require(1 <= config.processing.queue_size <= 64, "processing.queue_size", "phải từ 1 đến 64")
