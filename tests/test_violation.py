"""Phase 7: SpeedViolationDetector rules."""

from __future__ import annotations

from app.core.violation_detector import SpeedViolationDetector


def test_equal_to_limit_is_not_a_violation() -> None:
    detector = SpeedViolationDetector(80)
    assert not detector.is_violation(80.0)
    assert detector.is_violation(80.1)
    assert not detector.check(1, 80.0, "car", 10)
    assert detector.get_violations() == {}


def test_none_speed_is_ignored() -> None:
    detector = SpeedViolationDetector(80)
    assert not detector.check(1, None, "car", 10)


def test_one_record_per_vehicle_with_max_speed() -> None:
    detector = SpeedViolationDetector(80)
    assert detector.check(4, 81.0, "car", 10)
    assert detector.check(4, 95.0, "car", 11)
    assert detector.check(4, 88.0, "car", 12)
    violations = detector.get_violations()
    assert list(violations) == [4]
    record = violations[4]
    assert record.max_speed_kmh == 95.0
    assert record.first_frame == 10
    assert record.last_frame == 12
    assert record.speed_limit_kmh == 80.0


def test_multiple_vehicles_and_reset() -> None:
    detector = SpeedViolationDetector(60)
    detector.check(1, 70, "truck", 1)
    detector.check(2, 50, "car", 1)
    detector.check(3, 61, "bus", 2)
    assert sorted(detector.get_violations()) == [1, 3]
    detector.reset()
    assert detector.get_violations() == {}
