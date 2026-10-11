"""Region of interest (ROI) configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ANCHORS: tuple[str, ...] = ("bottom_center", "center")


@dataclass
class ROI:
    """A polygon zone in original-frame pixels with its own rules.

    ``roi_id`` is ASCII because it is drawn on frames with ``cv2.putText``;
    ``name`` may be Vietnamese and is shown in the UI only. An empty
    ``allowed_classes`` list means every detected class. ``max_dwell_seconds``
    of 0 disables the overstay alert.
    """

    roi_id: str
    name: str
    polygon: list[list[int]]
    allowed_classes: list[str] = field(default_factory=list)
    max_dwell_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ROI:
        """Build an ROI from JSON. Raises ValueError on missing keys or bad types."""
        if not isinstance(data, dict):
            raise ValueError("mỗi vùng ROI phải là object JSON")
        missing = {"roi_id", "name", "polygon"} - data.keys()
        if missing:
            raise ValueError(f"vùng ROI thiếu khóa: {', '.join(sorted(missing))}")
        try:
            polygon = [[int(round(float(x))), int(round(float(y)))] for x, y in data["polygon"]]
        except (TypeError, ValueError) as exc:
            raise ValueError("polygon phải là danh sách điểm [x, y]") from exc
        return cls(
            roi_id=str(data["roi_id"]),
            name=str(data["name"]),
            polygon=polygon,
            allowed_classes=[str(name) for name in data.get("allowed_classes", [])],
            max_dwell_seconds=float(data.get("max_dwell_seconds", 0.0)),
        )

    def allows(self, class_name: str) -> bool:
        """True when objects of ``class_name`` are monitored in this region."""
        return not self.allowed_classes or class_name in self.allowed_classes

    def scaled(self, scale: float) -> ROI:
        """Return a copy with the polygon multiplied by ``scale``."""
        return ROI(
            roi_id=self.roi_id,
            name=self.name,
            polygon=[[round(x * scale), round(y * scale)] for x, y in self.polygon],
            allowed_classes=list(self.allowed_classes),
            max_dwell_seconds=self.max_dwell_seconds,
        )
