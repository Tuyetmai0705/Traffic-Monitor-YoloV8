"""Dwell-time sessions: entry, dwell, exit, grace timeout, lost tracks, overstay."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.dwell_time_tracker import DwellTimeTracker
from app.models.event import EventType
from app.models.roi import ROI
from app.models.track_session import SessionStatus

FPS = 10.0


def make_tracker(max_dwell: float = 0.0, grace: float = 0.5, roi_ids=("zone_a",)) -> DwellTimeTracker:
    zones = [ROI(r, r, [[0, 0], [10, 0], [10, 10]], [], max_dwell) for r in roi_ids]
    return DwellTimeTracker(zones, grace_seconds=grace)


def step(tracker: DwellTimeTracker, frame: int, inside: dict[int, bool], roi_id: str = "zone_a", extra: dict | None = None):
    """Feed one frame. ``inside`` maps visible track id → inside zone_a."""
    ids = np.array(list(inside), dtype=int)
    memberships = {roi_id: np.array(list(inside.values()), dtype=bool)}
    if extra:
        memberships.update({key: np.array(value, dtype=bool) for key, value in extra.items()})
    return tracker.update(memberships, ids, ["car"] * len(ids), frame / FPS)


def test_enter_dwell_and_exit_after_grace() -> None:
    tracker = make_tracker(grace=0.5)
    events = step(tracker, 10, {7: True})
    assert [e.event_type for e in events] == [EventType.ENTER]
    assert events[0].timestamp == pytest.approx(1.0)
    for frame in range(11, 31):
        step(tracker, frame, {7: True})
    assert tracker.get_dwell("zone_a", 7) == pytest.approx(2.0)
    assert tracker.get_active_sessions()[0].dwell_seconds == pytest.approx(2.0)

    # Outside from frame 31; it stays visible, so after the 0.5 s grace the session ends as EXIT.
    for frame in range(31, 36):
        assert step(tracker, frame, {7: False}) == []
    events = step(tracker, 36, {7: False})
    assert [e.event_type for e in events] == [EventType.EXIT]
    assert events[0].timestamp == pytest.approx(3.0)              # exit time = last time seen inside
    assert events[0].dwell_seconds == pytest.approx(2.0)          # T_exit - T_entry
    session = tracker.get_sessions()[0]
    assert (session.status, session.exit_time, session.dwell_seconds) == (SessionStatus.EXITED, pytest.approx(3.0), pytest.approx(2.0))
    assert tracker.get_counts("zone_a") == {"entries": 1, "exits": 1, "lost": 0, "overstays": 0}


def test_boundary_flicker_within_grace_keeps_one_session() -> None:
    tracker = make_tracker(grace=0.5)
    pattern = [True, True, False, False, True, False, True, True]  # gaps of ≤ 0.2 s
    for frame, inside in enumerate(pattern):
        step(tracker, frame, {3: inside})
    assert tracker.get_counts("zone_a")["entries"] == 1
    assert len(tracker.get_sessions()) == 1


def test_track_lost_while_inside_is_lost_not_exit() -> None:
    tracker = make_tracker(grace=0.5)
    for frame in range(5):
        step(tracker, frame, {4: True})
    assert step(tracker, 5, {}) == []                              # missing for 0.1 s: still inside
    assert tracker.get_open_track_ids()["zone_a"] == {4}
    events = []
    for frame in range(6, 11):
        events += step(tracker, frame, {})
    assert [e.event_type for e in events] == [EventType.LOST]
    assert tracker.get_sessions()[0].status is SessionStatus.LOST
    assert tracker.get_counts("zone_a")["lost"] == 1


def test_seen_outside_then_vanishing_is_exit() -> None:
    tracker = make_tracker(grace=0.5)
    for frame in range(5):
        step(tracker, frame, {4: True})
    step(tracker, 5, {4: False})                                   # observed outside once
    events = []
    for frame in range(6, 11):
        events += step(tracker, frame, {})                         # then leaves the frame
    assert [e.event_type for e in events] == [EventType.EXIT]


def test_reentry_after_grace_starts_new_session() -> None:
    tracker = make_tracker(grace=0.2)
    for frame in range(3):
        step(tracker, frame, {1: True})
    for frame in range(3, 8):
        step(tracker, frame, {1: False})
    events = step(tracker, 8, {1: True})
    assert [e.event_type for e in events] == [EventType.ENTER]
    sessions = tracker.get_sessions()
    assert [s.session_id for s in sessions] == [1, 2]
    assert sessions[1].entry_time == pytest.approx(0.8)


def test_overstay_alert_once_per_session() -> None:
    tracker = make_tracker(max_dwell=1.0)
    events = []
    for frame in range(30):
        events += step(tracker, frame, {9: True})
    overstays = [e for e in events if e.event_type is EventType.OVERSTAY]
    assert len(overstays) == 1
    assert overstays[0].timestamp == pytest.approx(1.1)            # first frame with dwell > 1.0 s
    assert tracker.is_overstaying("zone_a", 9)
    assert tracker.get_active_sessions()[0].overstay
    assert tracker.get_counts("zone_a")["overstays"] == 1


def test_zero_limit_disables_overstay() -> None:
    tracker = make_tracker(max_dwell=0.0)
    events = []
    for frame in range(100):
        events += step(tracker, frame, {9: True})
    assert all(e.event_type is not EventType.OVERSTAY for e in events)


def test_regions_are_tracked_independently() -> None:
    tracker = make_tracker(roi_ids=("zone_a", "zone_b"))
    step(tracker, 0, {1: True}, extra={"zone_b": [True]})
    step(tracker, 1, {1: True}, extra={"zone_b": [False]})
    open_ids = tracker.get_open_track_ids()
    assert open_ids == {"zone_a": {1}, "zone_b": set()}             # seen outside zone_b: not counted
    assert [s.roi_id for s in tracker.get_active_sessions(include_leaving=True)] == ["zone_a", "zone_b"]
    for frame in range(2, 10):
        step(tracker, frame, {1: True}, extra={"zone_b": [False]})
    assert tracker.get_open_track_ids() == {"zone_a": {1}, "zone_b": set()}


def test_media_time_drives_dwell_so_pause_freezes_it() -> None:
    tracker = make_tracker()
    step(tracker, 0, {1: True})
    step(tracker, 5, {1: True})
    # A pause produces no frames; wall-clock time never enters the computation.
    assert tracker.get_dwell("zone_a", 1) == pytest.approx(0.5)


def test_reset_and_invalid_grace() -> None:
    tracker = make_tracker()
    step(tracker, 0, {1: True})
    tracker.reset()
    assert tracker.get_sessions() == []
    assert tracker.get_counts("zone_a")["entries"] == 0
    with pytest.raises(ValueError):
        make_tracker(grace=-1)


def test_session_and_event_json_shape() -> None:
    tracker = make_tracker()
    event = step(tracker, 124, {18: True})[0]
    assert event.to_dict() == {
        "event_id": "event-0001", "track_id": 18, "roi_id": "zone_a", "event_type": "ENTER",
        "timestamp": 12.4, "class_name": "car", "dwell_seconds": None,
    }
    step(tracker, 187, {18: True})
    session = tracker.get_active_sessions()[0].to_dict()
    assert session["entry_time"] == 12.4 and session["last_seen_time"] == 18.7
    assert session["exit_time"] is None and session["dwell_seconds"] == 6.3 and session["status"] == "inside"


def test_dwell_does_not_grow_while_outside_during_grace() -> None:
    tracker = make_tracker(max_dwell=0.35, grace=1.0)
    for frame in range(4):                      # inside 0.0 .. 0.3 s
        step(tracker, frame, {5: True})
    events = []
    for frame in range(4, 9):                   # outside but within the grace period
        events += step(tracker, frame, {5: False})
    assert tracker.get_dwell("zone_a", 5) == pytest.approx(0.3)
    assert tracker.get_active_sessions() == []                       # leaving: hidden like occupancy
    assert tracker.get_active_sessions(include_leaving=True)[0].dwell_seconds == pytest.approx(0.3)
    assert not any(e.event_type is EventType.OVERSTAY for e in events)   # never inside for > 0.35 s


def test_entry_needs_consecutive_frames() -> None:
    zones = [ROI("zone_a", "zone_a", [[0, 0], [10, 0], [10, 10]], [], 0.0)]
    tracker = DwellTimeTracker(zones, grace_seconds=0.5, min_confirm_frames=3)
    events = []
    events += step(tracker, 0, {1: True})
    events += step(tracker, 1, {1: True})
    events += step(tracker, 2, {1: False})                   # broken streak: nothing yet
    assert events == [] and tracker.get_open_track_ids()["zone_a"] == set()
    for frame in range(3, 6):
        events += step(tracker, frame, {1: True})
    assert [e.event_type for e in events] == [EventType.ENTER]
    assert events[0].timestamp == pytest.approx(0.3)           # entry = first of the 3 frames
    assert tracker.get_active_sessions()[0].entry_time == pytest.approx(0.3)


def test_one_frame_track_never_enters() -> None:
    zones = [ROI("zone_a", "zone_a", [[0, 0], [10, 0], [10, 10]], [], 0.0)]
    tracker = DwellTimeTracker(zones, grace_seconds=0.2, min_confirm_frames=3)
    events = step(tracker, 0, {9: True})
    for frame in range(1, 10):
        events += step(tracker, frame, {})
    assert events == [] and tracker.get_sessions() == []
