"""Speed violation record."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Violation:
    """One vehicle that exceeded the speed limit. One record per tracker id."""

    tracker_id: int
    class_name: str
    max_speed_kmh: float
    speed_limit_kmh: float
    first_frame: int
    last_frame: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)
