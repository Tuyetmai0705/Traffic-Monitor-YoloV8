"""Chronological store of region events, with logging of alerts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from app.models.event import EventType, RegionEvent
from app.utils.logger import get_logger
from app.utils.time_utils import format_timestamp

logger = get_logger("event_service")


class EventService:
    """Record ENTER / EXIT / LOST / OVERSTAY events for the UI log and the report."""

    def __init__(self) -> None:
        """Start with an empty log."""
        self._events: list[RegionEvent] = []
        self._counts: Counter[str] = Counter()

    def record(self, events: Iterable[RegionEvent]) -> None:
        """Append events in order; alerts are also written to the application log."""
        for event in events:
            self._events.append(event)
            self._counts[event.event_type.value] += 1
            if event.event_type in (EventType.OVERSTAY, EventType.LOST):
                logger.warning(
                    "Region %s: #%d %s %s at %s (dwell %.1f s)",
                    event.event_type.value, event.track_id, event.class_name, event.roi_id,
                    format_timestamp(event.timestamp), event.dwell_seconds or 0.0,
                )

    def get_events(self) -> list[RegionEvent]:
        """Every recorded event, oldest first."""
        return list(self._events)

    def recent(self, limit: int) -> tuple[RegionEvent, ...]:
        """The newest ``limit`` events, oldest first."""
        return tuple(self._events[-limit:]) if limit > 0 else ()

    def count(self, event_type: EventType | None = None) -> int:
        """Number of events, optionally of one type."""
        return len(self._events) if event_type is None else self._counts[event_type.value]

    def reset(self) -> None:
        """Clear the log."""
        self._events.clear()
        self._counts.clear()
