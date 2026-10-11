"""Occupancy is the number of objects inside now, not the cumulative number of entries."""

from __future__ import annotations

import pytest

from app.core.occupancy_counter import OccupancyCounter
from app.models.event import EventType, RegionEvent
from app.services.event_service import EventService


def test_current_peak_and_average() -> None:
    counter = OccupancyCounter(["zone_a", "zone_b"])
    counter.update({"zone_a": {1, 2, 3}, "zone_b": set()})
    counter.update({"zone_a": {2}, "zone_b": {7}})
    assert counter.get_occupancy("zone_a") == 1
    assert counter.get_peak("zone_a") == 3
    assert counter.get_average("zone_a") == pytest.approx(2.0)
    assert counter.update({}) == {"zone_a": 0, "zone_b": 0}


def test_occupancy_differs_from_entry_count() -> None:
    counter = OccupancyCounter(["zone_a"])
    entries = 0
    inside: set[int] = set()
    for track_id in range(1, 6):            # five objects pass through one after another
        inside = {track_id}
        entries += 1
        counter.update({"zone_a": inside})
    assert entries == 5
    assert counter.get_occupancy("zone_a") == 1
    assert counter.get_peak("zone_a") == 1


def test_reset() -> None:
    counter = OccupancyCounter(["zone_a"])
    counter.update({"zone_a": {1, 2}})
    counter.reset()
    assert (counter.get_occupancy("zone_a"), counter.get_peak("zone_a"), counter.get_average("zone_a")) == (0, 0, 0.0)


def test_event_service_records_in_order() -> None:
    service = EventService()
    events = [
        RegionEvent(f"event-{i:04d}", i, "zone_a", kind, float(i), "car", None if kind is EventType.ENTER else 1.0)
        for i, kind in enumerate([EventType.ENTER, EventType.OVERSTAY, EventType.EXIT, EventType.ENTER, EventType.LOST], 1)
    ]
    service.record(events)
    assert service.count() == 5
    assert service.count(EventType.ENTER) == 2
    assert [e.event_id for e in service.recent(2)] == ["event-0004", "event-0005"]
    assert service.recent(0) == ()
    service.reset()
    assert service.get_events() == []
