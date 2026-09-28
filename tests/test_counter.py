"""Phase 5: VehicleCounter IN/OUT totals and per-class counts."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.counter import VehicleCounter
from tests.conftest import make_detections


def _move(counter: VehicleCounter, tracker_id: int, class_id: int, ys: list[int]) -> None:
    for y in ys:
        counter.count(make_detections([[90, y - 20, 110, y]], [class_id], tracker_ids=[tracker_id]))


def test_crossing_downwards_counts_out_and_upwards_counts_in() -> None:
    counter = VehicleCounter(start=[0, 100], end=[200, 100])
    _move(counter, 1, 2, [60, 90, 110, 140])     # car going down
    _move(counter, 2, 7, [140, 110, 90, 60])     # truck going up
    assert counter.get_total_out() == 1
    assert counter.get_total_in() == 1
    counts = counter.get_class_counts()
    assert counts["car"] == {"in": 0, "out": 1}
    assert counts["truck"] == {"in": 1, "out": 0}
    assert counts["bus"] == {"in": 0, "out": 0}
    assert counts["motorcycle"] == {"in": 0, "out": 0}


def test_per_class_counts_for_all_classes() -> None:
    counter = VehicleCounter(start=[0, 100], end=[1000, 100])
    for index, class_id in enumerate([2, 3, 5, 7, 3]):
        x = 100 + index * 150
        for y in (140, 110, 90, 60):
            counter.count(make_detections([[x, y - 20, x + 20, y]], [class_id], tracker_ids=[index + 1]))
    counts = counter.get_class_counts()
    assert counter.get_total_in() == 5
    assert counts["motorcycle"]["in"] == 2
    assert sum(c["in"] for c in counts.values()) == counter.get_total_in()


def test_trigger_called_exactly_once_per_frame() -> None:
    counter = VehicleCounter(start=[0, 100], end=[200, 100])
    with patch.object(counter.line_zone, "trigger", wraps=counter.line_zone.trigger) as spy:
        counter.count(make_detections([[0, 0, 10, 10]] * 3, [2, 3, 7], tracker_ids=[1, 2, 3]))
    assert spy.call_count == 1


def test_untracked_detections_rejected() -> None:
    counter = VehicleCounter(start=[0, 100], end=[200, 100])
    with pytest.raises(ValueError):
        counter.count(make_detections([[0, 0, 10, 10]], [2]))


def test_reset() -> None:
    counter = VehicleCounter(start=[0, 100], end=[200, 100])
    _move(counter, 1, 2, [60, 140])
    counter.reset()
    assert counter.get_total_in() == counter.get_total_out() == 0
    assert all(c == {"in": 0, "out": 0} for c in counter.get_class_counts().values())
