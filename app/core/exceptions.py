"""Domain exceptions. ``str(exc)`` is always a message safe for end users."""

from __future__ import annotations

from pathlib import Path

from app.constants import messages as msg


class TrafficMonitorError(Exception):
    """Base class for every expected application error."""


class VideoNotFoundError(TrafficMonitorError):
    """The selected video file does not exist."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(msg.VIDEO_NOT_FOUND.format(path=path))
        self.path = Path(path)


class VideoOpenError(TrafficMonitorError):
    """OpenCV cannot open the video container."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(msg.VIDEO_OPEN_FAILED.format(path=path))
        self.path = Path(path)


class VideoCodecError(TrafficMonitorError):
    """Frames cannot be decoded, or no writer codec is available."""

    def __init__(self, detail: str) -> None:
        super().__init__(msg.VIDEO_CODEC_FAILED.format(detail=detail))
        self.detail = detail


class ModelNotFoundError(TrafficMonitorError):
    """The YOLO weights file does not exist."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(msg.MODEL_NOT_FOUND.format(path=path))
        self.path = Path(path)


class ModelLoadError(TrafficMonitorError):
    """The YOLO weights exist but cannot be loaded."""

    def __init__(self, path: str | Path, detail: str = "") -> None:
        super().__init__(msg.MODEL_LOAD_FAILED.format(path=path))
        self.path = Path(path)
        self.detail = detail


class InferenceError(TrafficMonitorError):
    """YOLO inference failed on a frame."""

    def __init__(self, detail: str, frame_index: int | None = None) -> None:
        self.detail = detail
        self.frame_index = frame_index
        super().__init__(detail)

    def __str__(self) -> str:
        frame = "?" if self.frame_index is None else str(self.frame_index)
        return msg.INFERENCE_FAILED.format(frame=frame, detail=self.detail)


class InvalidConfigError(TrafficMonitorError):
    """A configuration value is missing or out of range."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(msg.INVALID_CONFIG.format(field=field, reason=reason))
        self.field = field
        self.reason = reason


class InvalidPerspectiveError(TrafficMonitorError):
    """The perspective source points do not form a valid quadrilateral."""

    def __init__(self, reason: str) -> None:
        super().__init__(msg.INVALID_PERSPECTIVE.format(reason=reason))
        self.reason = reason


class OutputPathError(TrafficMonitorError):
    """An output directory or file cannot be written."""

    def __init__(self, path: str | Path, detail: str) -> None:
        super().__init__(msg.OUTPUT_PATH_FAILED.format(path=path, detail=detail))
        self.path = Path(path)
