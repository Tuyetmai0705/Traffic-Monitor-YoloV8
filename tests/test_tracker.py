"""Phase 4: VehicleTracker assigns stable ids and resets."""

from __future__ import annotations

import supervision as sv

from app.config.settings import AppConfig
from app.core.tracker import VehicleTracker
from tests.conftest import make_detections


def _frame(offset: float) -> sv.Detections:
    return make_detections(
        boxes=[[100 + offset, 100, 160 + offset, 150], [400, 300 + offset, 480, 380 + offset]],
        class_ids=[2, 7],
        confidences=[0.9, 0.85],
    )


def test_update_assigns_tracker_ids(config: AppConfig) -> None:
    tracker = VehicleTracker(config.tracker, frame_rate=25)
    tracked = tracker.update(_frame(0))
    assert tracked.tracker_id is not None
    assert len(tracked.tracker_id) == 2
    assert len(set(tracked.tracker_id.tolist())) == 2


def test_ids_are_stable_across_frames(config: AppConfig) -> None:
    tracker = VehicleTracker(config.tracker, frame_rate=25)
    first = tracker.update(_frame(0))
    ids_by_class = dict(zip(first.class_id.tolist(), first.tracker_id.tolist()))
    for step in range(1, 6):
        tracked = tracker.update(_frame(step * 3))
        assert dict(zip(tracked.class_id.tolist(), tracked.tracker_id.tolist())) == ids_by_class
    assert sorted(tracker.get_tracker_ids()) == sorted(ids_by_class.values())


def test_empty_detections(config: AppConfig) -> None:
    tracker = VehicleTracker(config.tracker, frame_rate=25)
    tracked = tracker.update(sv.Detections.empty())
    assert len(tracked) == 0
    assert tracked.tracker_id is not None
    assert tracker.get_tracker_ids() == []


def test_reset_clears_ids(config: AppConfig) -> None:
    tracker = VehicleTracker(config.tracker, frame_rate=25)
    tracker.update(_frame(0))
    tracker.reset()
    assert tracker.get_tracker_ids() == []
    assert len(tracker.update(_frame(0))) == 2
