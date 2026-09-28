"""Speed limit violation detection."""

from __future__ import annotations

from dataclasses import replace

from app.models.violation import Violation
from app.utils.logger import get_logger

logger = get_logger("violation_detector")


class SpeedViolationDetector:
    """Flag vehicles faster than the limit and keep their maximum speed."""

    def __init__(self, speed_limit_kmh: float) -> None:
        """Create a detector for a speed limit in km/h."""
        self._speed_limit = float(speed_limit_kmh)
        self._violations: dict[int, Violation] = {}

    @property
    def speed_limit(self) -> float:
        """Speed limit in km/h."""
        return self._speed_limit

    def check(self, tracker_id: int, speed: float | None, class_name: str, frame_index: int) -> bool:
        """Register and return True when ``speed`` exceeds the limit; None speeds are ignored."""
        if speed is None or not self.is_violation(speed):
            return False
        self.register_violation(tracker_id, speed, class_name, frame_index)
        return True

    def is_violation(self, speed: float) -> bool:
        """Strictly greater than the limit is a violation; equal is not."""
        return speed > self._speed_limit

    def register_violation(self, tracker_id: int, speed: float, class_name: str, frame_index: int) -> None:
        """Create a record for a new vehicle or update the existing one's max speed."""
        tid = int(tracker_id)
        existing = self._violations.get(tid)
        if existing is None:
            self._violations[tid] = Violation(tid, class_name, speed, self._speed_limit, frame_index, frame_index)
            logger.warning("Speed violation: #%d %s %.1f km/h > %.0f km/h", tid, class_name, speed, self._speed_limit)
            return
        self._violations[tid] = replace(
            existing, max_speed_kmh=max(existing.max_speed_kmh, speed), last_frame=frame_index
        )

    def get_violations(self) -> dict[int, Violation]:
        """Copy of all violations keyed by tracker id."""
        return dict(self._violations)

    def reset(self) -> None:
        """Clear all violations."""
        self._violations.clear()
