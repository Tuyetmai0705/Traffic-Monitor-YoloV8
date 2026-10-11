"""Visit sessions per (region, track): entry time, dwell time, exit time and alerts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.models.event import EventType, RegionEvent
from app.models.roi import ROI
from app.models.track_session import SessionStatus, TrackSession


@dataclass
class _Visit:
    """Mutable state of an open visit."""

    session_id: int
    track_id: int
    class_name: str
    roi_id: str
    entry_time: float
    last_seen_time: float
    overstay: bool = False
    seen_outside: bool = False  # observed outside the region after its last time inside


class DwellTimeTracker:
    """Turn per-frame membership into ENTER / EXIT / LOST / OVERSTAY events.

    * A track entering a region opens a session at the current media time.
    * While inside, ``dwell = now - entry_time``. During the grace period the
      dwell stays at the last time the object was confirmed inside, so a
      closed session always ends with ``T_exit - T_entry``.
    * A track must stay outside (or unseen) for longer than ``grace_seconds``
      before the session closes, so boundary flicker and short occlusions do
      not split a visit. The exit time is the last time it was seen inside.
    * EXIT when the object was observed outside the region after its last
      time inside; LOST when it vanished while still inside (occlusion,
      tracker loss). Two different track ids are never merged.
    * Leaving and coming back after the grace period starts a new session.
    """

    def __init__(self, zones: Sequence[ROI], grace_seconds: float) -> None:
        """Track sessions for ``zones`` with an exit grace period in seconds."""
        if grace_seconds < 0:
            raise ValueError("grace_seconds phải ≥ 0")
        self._zones = {zone.roi_id: zone for zone in zones}
        self._grace = grace_seconds
        self.reset()

    @property
    def grace_seconds(self) -> float:
        """Exit grace period."""
        return self._grace

    def update(
        self,
        memberships: dict[str, np.ndarray],
        tracker_ids: np.ndarray,
        class_names: Sequence[str],
        timestamp: float,
    ) -> list[RegionEvent]:
        """Apply one frame's membership at media time ``timestamp`` and return new events."""
        self._now = timestamp
        events: list[RegionEvent] = []
        visible = {int(tid) for tid in tracker_ids}
        for roi_id, zone in self._zones.items():
            mask = memberships.get(roi_id, np.zeros(len(tracker_ids), dtype=bool))
            inside_now: set[int] = set()
            for tid, name, inside in zip(tracker_ids, class_names, mask):
                if not inside:
                    continue
                tid = int(tid)
                inside_now.add(tid)
                events.extend(self._mark_inside(zone, tid, str(name), timestamp))
            events.extend(self._close_expired(roi_id, inside_now, visible, timestamp))
        return events

    def get_active_sessions(self, roi_id: str | None = None) -> list[TrackSession]:
        """Open sessions (status inside), optionally for one region, oldest first."""
        visits = [v for v in self._open.values() if roi_id is None or v.roi_id == roi_id]
        return [self._snapshot(v) for v in sorted(visits, key=lambda v: v.session_id)]

    def get_sessions(self) -> list[TrackSession]:
        """Every session of the run (closed and open), in entry order."""
        sessions = self._closed + self.get_active_sessions()
        return sorted(sessions, key=lambda session: session.session_id)

    def get_closed_sessions(self, roi_id: str | None = None) -> list[TrackSession]:
        """Finished sessions, optionally for one region."""
        return [s for s in self._closed if roi_id is None or s.roi_id == roi_id]

    def get_counts(self, roi_id: str) -> dict[str, int]:
        """Cumulative ENTER, EXIT, LOST and OVERSTAY counts of a region."""
        return dict(self._counts[roi_id])

    def get_open_track_ids(self) -> dict[str, set[int]]:
        """``{roi_id: {track ids inside}}`` for occupancy."""
        result: dict[str, set[int]] = {roi_id: set() for roi_id in self._zones}
        for roi_id, track_id in self._open:
            result[roi_id].add(track_id)
        return result

    def get_dwell(self, roi_id: str, track_id: int) -> float | None:
        """Current dwell time of an open session (up to the last time seen inside), or None."""
        visit = self._open.get((roi_id, track_id))
        return None if visit is None else visit.last_seen_time - visit.entry_time

    def is_overstaying(self, roi_id: str, track_id: int) -> bool:
        """True when an open session has exceeded the region's limit."""
        visit = self._open.get((roi_id, track_id))
        return bool(visit and visit.overstay)

    def reset(self) -> None:
        """Forget every session and counter."""
        self._open: dict[tuple[str, int], _Visit] = {}
        self._closed: list[TrackSession] = []
        self._counts = {roi_id: {"entries": 0, "exits": 0, "lost": 0, "overstays": 0} for roi_id in self._zones}
        self._next_session = 1
        self._next_event = 1
        self._now = 0.0

    # ------------------------------------------------------------------ helpers
    def _mark_inside(self, zone: ROI, track_id: int, class_name: str, timestamp: float) -> list[RegionEvent]:
        events: list[RegionEvent] = []
        key = (zone.roi_id, track_id)
        visit = self._open.get(key)
        if visit is None:
            visit = _Visit(self._next_session, track_id, class_name, zone.roi_id, timestamp, timestamp)
            self._next_session += 1
            self._open[key] = visit
            self._counts[zone.roi_id]["entries"] += 1
            events.append(self._event(visit, EventType.ENTER, timestamp))
        visit.last_seen_time = timestamp
        visit.class_name = class_name
        visit.seen_outside = False
        dwell = timestamp - visit.entry_time
        if zone.max_dwell_seconds > 0 and dwell > zone.max_dwell_seconds and not visit.overstay:
            visit.overstay = True
            self._counts[zone.roi_id]["overstays"] += 1
            events.append(self._event(visit, EventType.OVERSTAY, timestamp, dwell))
        return events

    def _close_expired(self, roi_id: str, inside_now: set[int], visible: set[int], timestamp: float) -> list[RegionEvent]:
        events: list[RegionEvent] = []
        for key in [k for k in self._open if k[0] == roi_id and k[1] not in inside_now]:
            visit = self._open[key]
            if visit.track_id in visible:
                visit.seen_outside = True
            if timestamp - visit.last_seen_time <= self._grace:
                continue
            lost = not visit.seen_outside
            status = SessionStatus.LOST if lost else SessionStatus.EXITED
            dwell = visit.last_seen_time - visit.entry_time
            self._closed.append(self._snapshot(visit, status=status, exit_time=visit.last_seen_time))
            self._counts[roi_id]["lost" if lost else "exits"] += 1
            events.append(self._event(visit, EventType.LOST if lost else EventType.EXIT, visit.last_seen_time, dwell))
            del self._open[key]
        return events

    def _snapshot(
        self, visit: _Visit, status: SessionStatus = SessionStatus.INSIDE, exit_time: float | None = None
    ) -> TrackSession:
        end = visit.last_seen_time if exit_time is None else exit_time
        return TrackSession(
            session_id=visit.session_id,
            track_id=visit.track_id,
            class_name=visit.class_name,
            roi_id=visit.roi_id,
            entry_time=visit.entry_time,
            last_seen_time=visit.last_seen_time,
            exit_time=exit_time,
            dwell_seconds=max(0.0, end - visit.entry_time),
            status=status,
            overstay=visit.overstay,
        )

    def _event(self, visit: _Visit, kind: EventType, timestamp: float, dwell: float | None = None) -> RegionEvent:
        event = RegionEvent(
            event_id=f"event-{self._next_event:04d}",
            track_id=visit.track_id,
            roi_id=visit.roi_id,
            event_type=kind,
            timestamp=timestamp,
            class_name=visit.class_name,
            dwell_seconds=dwell,
        )
        self._next_event += 1
        return event
