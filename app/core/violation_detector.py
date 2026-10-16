"""Speed limit violation detection."""

from __future__ import annotations

from dataclasses import replace

from app.models.violation import Violation
from app.utils.logger import get_logger

logger = get_logger("violation_detector")


class SpeedViolationDetector:
    """Flag vehicles faster than the limit and keep their maximum speed."""

    def __init__(self, speed_limit_kmh: float, confirm_samples: int = 1) -> None:
        """Create a detector; a vehicle must exceed the limit in ``confirm_samples`` consecutive estimates."""
        self._speed_limit = float(speed_limit_kmh)
        self._confirm = max(1, int(confirm_samples))
        self._violations: dict[int, Violation] = {}
        self._streaks: dict[int, int] = {}

    @property
    def speed_limit(self) -> float:
        """Speed limit in km/h."""
        return self._speed_limit

    def check(self, tracker_id: int, speed: float | None, class_name: str, frame_index: int, fresh: bool = True) -> bool:
        """Return True when a confirmed violator is currently above the limit.

        Only fresh estimates (re-computed this frame) advance the streak, so a
        single noisy spike is never recorded. Once confirmed, the record keeps
        its maximum speed; the flag follows the current speed.
        """
        tid = int(tracker_id)
        if speed is None:
            return False
        if fresh:
            self._streaks[tid] = self._streaks.get(tid, 0) + 1 if self.is_violation(speed) else 0
        if tid in self._violations:
            if fresh and self.is_violation(speed):
                self.register_violation(tid, speed, class_name, frame_index)
            return self.is_violation(speed)
        if self._streaks.get(tid, 0) >= self._confirm:
            self.register_violation(tid, speed, class_name, frame_index)
            return True
        return False

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
        self._streaks.clear()
