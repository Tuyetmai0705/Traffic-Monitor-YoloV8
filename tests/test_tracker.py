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


def _warm(tracker: VehicleTracker, frames: int = 3) -> sv.Detections:
    """Feed ``frames`` frames; tracks are reported once seen ``min_hits`` (3) times."""
    tracked = None
    for step in range(frames):
        tracked = tracker.update(_frame(step))
    return tracked


def test_update_assigns_tracker_ids(config: AppConfig) -> None:
    tracker = VehicleTracker(config.tracker, frame_rate=25)
    tracked = _warm(tracker)
    assert tracked.tracker_id is not None
    assert len(tracked.tracker_id) == 2
    assert len(set(tracked.tracker_id.tolist())) == 2


def test_ids_are_stable_across_frames(config: AppConfig) -> None:
    tracker = VehicleTracker(config.tracker, frame_rate=25)
    first = _warm(tracker)
    ids_by_class = dict(zip(first.class_id.tolist(), first.tracker_id.tolist()))
    for step in range(3, 9):
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
    _warm(tracker)
    tracker.reset()
    assert tracker.get_tracker_ids() == []
    assert len(tracker.update(_frame(0))) == 0          # unconfirmed again after reset
    assert len(_warm(tracker)) == 2


def test_one_frame_detection_is_never_reported(config: AppConfig) -> None:
    tracker = VehicleTracker(config.tracker, frame_rate=25)
    ghost = make_detections([[700, 700, 760, 740]], [2], confidences=[0.9])
    assert len(tracker.update(ghost)) == 0
    for _ in range(5):
        assert len(tracker.update(sv.Detections.empty())) == 0


def test_class_vote_keeps_one_class_per_track(config: AppConfig) -> None:
    tracker = VehicleTracker(config.tracker, frame_rate=25)
    pattern = [2, 2, 7, 2, 7, 2, 2, 7, 2, 2]                 # car mostly, truck sometimes
    classes = []
    for step, class_id in enumerate(pattern):
        tracked = tracker.update(make_detections([[100 + step * 4, 100, 180 + step * 4, 160]], [class_id], confidences=[0.8]))
        classes += tracked.class_id.tolist()
    assert classes and set(classes) == {2}


def test_kalman_reduces_box_jitter(config: AppConfig) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    tracker = VehicleTracker(config.tracker, frame_rate=25)
    raw, smooth = [], []
    for step in range(60):
        x = 100 + step * 5.0                                    # true motion: 5 px/frame
        noisy = [x + rng.normal(0, 3), 200 + rng.normal(0, 3), x + 80 + rng.normal(0, 3), 260 + rng.normal(0, 3)]
        tracked = tracker.update(make_detections([noisy], [2], confidences=[0.9]))
        if len(tracked):
            raw.append((noisy[0] + noisy[2]) / 2)
            smooth.append(float(tracked.xyxy[0, [0, 2]].mean()))
    raw_jitter = np.abs(np.diff(raw, 2)).mean()
    smooth_jitter = np.abs(np.diff(smooth, 2)).mean()
    assert smooth_jitter < raw_jitter * 0.5
    assert abs(smooth[-1] - (100 + 59 * 5.0 + 40)) < 6          # follows motion without big lag


def test_box_coasts_through_short_detection_gap() -> None:
    """YOLO misses a confirmed car for 4 frames: the box keeps moving with the predicted centre."""
    import numpy as np

    from app.core.box_smoother import COASTING_KEY, KalmanBoxSmoother

    smoother = KalmanBoxSmoother(min_hits=3, coast_frames=8)
    shown = []
    for step in range(20):
        if 8 <= step < 12:
            frame = make_detections(np.empty((0, 4)), [], tracker_ids=[])
        else:
            frame = make_detections([[100 + step * 5, 100, 180 + step * 5, 160]], [2], tracker_ids=[1])
        out = smoother.update(frame)
        shown.append((step, len(out), bool(out.data[COASTING_KEY][0]) if len(out) else None,
                      float(out.xyxy[0, [0, 2]].mean()) if len(out) else None))
    for step, count, coasting, centre in shown[8:12]:
        assert count == 1 and coasting, f"box disappeared at step {step}"
        assert abs(centre - (140 + step * 5)) < 12          # close to the true position (filter lag ~1-2 frames)
    steps = [shown[i + 1][3] - shown[i][3] for i in range(8, 11)]
    assert all(abs(d - 5) < 1.5 for d in steps), steps       # keeps moving at the car's speed
    assert shown[12][2] is False                              # real detection again


def test_new_bytetrack_id_is_stitched_to_lost_track() -> None:
    """Occlusion makes ByteTrack switch id 4 -> 9 at the same place: the user keeps seeing #4."""
    from app.core.box_smoother import KalmanBoxSmoother

    smoother = KalmanBoxSmoother(min_hits=3, stitch_frames=15)
    ids = []
    for step in range(16):
        if 6 <= step < 9:
            frame = make_detections([[500, 500, 560, 540]], [2], tracker_ids=[77])   # unrelated far car
        else:
            raw_id = 4 if step < 6 else 9
            frame = make_detections([[100 + step * 5, 100, 180 + step * 5, 160], [500, 500, 560, 540]], [2, 2],
                                    tracker_ids=[raw_id, 77])
        out = smoother.update(frame)
        ids.append(sorted(int(t) for t in out.tracker_id))
    assert all(4 in row for row in ids[2:]), ids             # never renamed to 9
    assert smoother.aliases == {9: 4}


def test_far_away_new_object_is_not_stitched() -> None:
    from app.core.box_smoother import KalmanBoxSmoother

    smoother = KalmanBoxSmoother(min_hits=1, stitch_frames=15)
    for step in range(5):
        smoother.update(make_detections([[100 + step * 5, 100, 180 + step * 5, 160]], [2], tracker_ids=[4]))
    out = smoother.update(make_detections([[900, 600, 980, 660]], [2], tracker_ids=[9]))
    assert 9 in out.tracker_id.tolist() and smoother.aliases == {}


def test_duplicate_box_of_same_car_is_merged() -> None:
    """A partly occluded car gets a second, overlapping box with a new id: no second track appears."""
    from app.core.box_smoother import KalmanBoxSmoother

    smoother = KalmanBoxSmoother(min_hits=1)
    for step in range(5):
        smoother.update(make_detections([[100, 100 + step * 4, 180, 160 + step * 4]], [2], tracker_ids=[4]))
    out = smoother.update(make_detections([[100, 120, 180, 180], [98, 118, 184, 186]], [2, 2], tracker_ids=[4, 8]))
    assert out.tracker_id.tolist() == [4]
    out = smoother.update(make_detections([[99, 122, 183, 190]], [2], tracker_ids=[8]))   # only the new id remains
    assert out.tracker_id.tolist() == [4]


def test_two_cars_side_by_side_are_not_merged() -> None:
    from app.core.box_smoother import KalmanBoxSmoother

    smoother = KalmanBoxSmoother(min_hits=1)
    smoother.update(make_detections([[100, 100, 180, 160]], [2], tracker_ids=[1]))
    out = smoother.update(make_detections([[100, 100, 180, 160], [150, 100, 230, 160]], [2, 2], tracker_ids=[1, 2]))
    assert sorted(out.tracker_id.tolist()) == [1, 2]                                    # IoU 0.45 < 0.6
