"""Per-track Kalman filter: steady boxes centred on the object, short-gap coasting and ID stitching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

RAW_BOXES_KEY = "raw_xyxy"   # unsmoothed YOLO boxes (NaN while coasting), kept for measurements (speed)
COASTING_KEY = "coasting"    # True for boxes predicted while YOLO missed the object
CONFIRMED_KEY = "confirmed"  # False until the track was seen min_hits times

# Constant-velocity model on (cx, cy, w, h). Noise is proportional to box height so a
# far car (20 px) and a near truck (400 px) are filtered alike (same weights as DeepSORT).
_STD_POSITION = 1.0 / 20
_STD_VELOCITY = 1.0 / 160
_F = np.eye(8)
_F[:4, 4:] = np.eye(4)
_H = np.eye(4, 8)


@dataclass
class _Track:
    mean: np.ndarray
    covariance: np.ndarray
    hits: int
    last_frame: int
    class_id: int
    confidence: float


class KalmanBoxSmoother:
    """Smooth ``xyxy`` of tracked detections around the object centre.

    * The box follows the filtered centre; width and height are smoothed more
      strongly (``size_scale``) than the position, so the box does not breathe
      with YOLO's size noise.
    * Tracks seen fewer than ``min_hits`` frames are hidden (one-frame false boxes).
    * When YOLO misses a confirmed object for up to ``coast_frames`` frames, its
      predicted box is still returned (flag ``data["coasting"]``), so boxes do
      not blink. Predicted boxes carry NaN raw boxes and are never measured.
    * When ByteTrack starts a new id where a recently lost track is predicted to
      be (same class, centre within ``stitch_distance`` box heights, similar
      size, lost ≤ ``stitch_frames``), the old id is kept. This repairs id
      switches after short occlusions; it is the tracker deciding identity, so
      two simultaneously visible objects are never merged.
    * A box of a brand-new track overlapping an existing track's box in the
      same frame (IoU > ``duplicate_iou``, same class) is a duplicate of that
      object (e.g. top-only box of a partly occluded car) and is dropped.
    """

    def __init__(
        self,
        min_hits: int = 3,
        measurement_scale: float = 6.0,
        max_missing: int = 30,
        size_scale: float = 2.5,
        coast_frames: int = 8,
        stitch_frames: int = 15,
        stitch_distance: float = 1.5,
        duplicate_iou: float = 0.6,
    ) -> None:
        """Create an empty filter bank."""
        if min_hits < 1:
            raise ValueError("min_hits phải ≥ 1")
        self._min_hits = min_hits
        self._measurement_scale = measurement_scale
        self._size_scale = size_scale
        self._max_missing = max(max_missing, stitch_frames, coast_frames)
        self._coast_frames = coast_frames
        self._stitch_frames = stitch_frames
        self._stitch_distance = stitch_distance
        self._duplicate_iou = duplicate_iou
        self._tracks: dict[int, _Track] = {}
        self._aliases: dict[int, int] = {}      # ByteTrack id -> id shown to the user
        self._present: set[int] = set()
        self._frame = 0

    @property
    def min_hits(self) -> int:
        """Frames a track must be seen before it is reported."""
        return self._min_hits

    @property
    def aliases(self) -> dict[int, int]:
        """Stitched ids: ByteTrack id → kept id."""
        return dict(self._aliases)

    def update(self, detections: sv.Detections, hide_unconfirmed: bool = True) -> sv.Detections:
        """Return detections with filtered boxes and stitched ids, plus coasting predictions.

        Every row has ``data["confirmed"]``; with ``hide_unconfirmed`` (default)
        rows of tracks seen fewer than ``min_hits`` frames are dropped.
        """
        self._frame += 1
        if detections.tracker_id is None:
            detections.tracker_id = np.empty(len(detections), dtype=int)
        boxes = detections.xyxy.astype(np.float64).copy()
        # Tracks already matched by ByteTrack in this frame are present, so they are not "lost" candidates.
        self._present = {self._aliases.get(int(r), int(r)) for r in detections.tracker_id}
        self._present &= set(self._tracks)
        ids = np.array([self._canonical_id(int(t), box, int(c)) for t, box, c in
                        zip(detections.tracker_id, boxes, detections.class_id)], dtype=int)
        duplicate = self._duplicates(ids, boxes, detections.class_id, detections.tracker_id)
        used = np.zeros(len(detections), dtype=bool)
        confirmed_flags = np.zeros(len(detections), dtype=bool)
        seen: set[int] = set()
        for index, box in enumerate(boxes):
            if duplicate[index]:
                continue
            tracker_id = int(ids[index])
            if tracker_id in seen:                      # two objects on one kept id: the second keeps its own id
                raw_id = int(detections.tracker_id[index])
                self._aliases.pop(raw_id, None)
                tracker_id = ids[index] = raw_id
                if tracker_id in seen:
                    continue
            seen.add(tracker_id)
            confidence = float(detections.confidence[index]) if detections.confidence is not None else 1.0
            track = self._step(tracker_id, box, int(detections.class_id[index]), confidence)
            boxes[index] = _to_xyxy(track.mean[:4])
            used[index] = True
            confirmed_flags[index] = track.hits >= self._min_hits
        measured = detections[used]
        measured.data = {                                 # only our keys, so coasting rows can be merged
            RAW_BOXES_KEY: measured.xyxy.astype(np.float32).copy(),
            COASTING_KEY: np.zeros(len(measured), dtype=bool),
            CONFIRMED_KEY: confirmed_flags[used],
        }
        measured.xyxy = boxes[used].astype(np.float32)
        measured.tracker_id = ids[used]
        coasting = self._coasting(seen)
        self._drop_missing()
        result = measured if len(coasting) == 0 else coasting if len(measured) == 0 else sv.Detections.merge([measured, coasting])
        return result[result.data[CONFIRMED_KEY]] if hide_unconfirmed and len(result) else result

    def reset(self) -> None:
        """Forget every track and alias."""
        self._tracks.clear()
        self._aliases.clear()
        self._frame = 0

    # ------------------------------------------------------------------ helpers
    def _canonical_id(self, raw_id: int, box: np.ndarray, class_id: int) -> int:
        """Map a ByteTrack id to the id shown, stitching a new id onto a recently lost track."""
        if raw_id in self._aliases:
            return self._aliases[raw_id]
        if raw_id in self._tracks:
            return raw_id
        measurement = _to_cxcywh(box)
        best, best_distance = None, float("inf")
        for tracker_id, track in self._tracks.items():
            missed = self._frame - track.last_frame
            if (tracker_id in self._present or missed < 1 or missed > self._stitch_frames
                    or track.hits < self._min_hits or track.class_id != class_id):
                continue
            predicted = track.mean[:4] + track.mean[4:] * missed
            height = max(predicted[3], 1.0)
            # A half-occluded car has a box about half as tall as when it reappears in full.
            distance = float(np.hypot(*(measurement[:2] - predicted[:2]))) / max(height, measurement[3])
            ratio = measurement[3] / height
            if distance <= self._stitch_distance and 0.4 <= ratio <= 2.5 and distance < best_distance:
                best, best_distance = tracker_id, distance
        if best is None:
            return raw_id
        self._aliases[raw_id] = best
        return best

    def _duplicates(self, ids: np.ndarray, boxes: np.ndarray, class_ids: np.ndarray, raw_ids: np.ndarray) -> np.ndarray:
        """Mark boxes of brand-new tracks that overlap an existing track's box in this frame.

        An occluded car can get two boxes (its visible top and a fuller box); the
        new one then becomes a second id. Such a box is dropped and its ByteTrack
        id is aliased to the existing track, so later frames keep the old id.
        """
        duplicate = np.zeros(len(ids), dtype=bool)
        existing = [j for j, tid in enumerate(ids) if int(tid) in self._tracks]
        for i, tid in enumerate(ids):
            if int(tid) in self._tracks:
                continue
            for j in existing:
                if class_ids[i] == class_ids[j] and _iou(boxes[i], boxes[j]) > self._duplicate_iou:
                    duplicate[i] = True
                    self._aliases[int(raw_ids[i])] = int(ids[j])
                    break
        return duplicate

    def _step(self, tracker_id: int, box: np.ndarray, class_id: int, confidence: float) -> _Track:
        measurement = _to_cxcywh(box)
        track = self._tracks.get(tracker_id)
        if track is None:
            track = self._initiate(measurement, class_id, confidence)
            self._tracks[tracker_id] = track
            return track
        for _ in range(max(1, self._frame - track.last_frame)):  # predict over missed frames
            self._predict(track)
        self._correct(track, measurement)
        track.hits += 1
        track.last_frame = self._frame
        track.class_id = class_id
        track.confidence = confidence
        return track

    def _coasting(self, seen: set[int]) -> sv.Detections:
        """Predicted boxes of confirmed tracks missed for at most ``coast_frames`` frames."""
        rows = []
        for tracker_id, track in self._tracks.items():
            missed = self._frame - track.last_frame
            if tracker_id in seen or track.hits < self._min_hits or not 1 <= missed <= self._coast_frames:
                continue
            predicted = track.mean[:4] + track.mean[4:] * missed
            rows.append((tracker_id, _to_xyxy(predicted), track.class_id, track.confidence))
        if not rows:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.array([r[1] for r in rows], dtype=np.float32),
            confidence=np.array([r[3] for r in rows], dtype=np.float32),
            class_id=np.array([r[2] for r in rows], dtype=int),
            tracker_id=np.array([r[0] for r in rows], dtype=int),
            data={RAW_BOXES_KEY: np.full((len(rows), 4), np.nan, dtype=np.float32),
                  COASTING_KEY: np.ones(len(rows), dtype=bool),
                  CONFIRMED_KEY: np.ones(len(rows), dtype=bool)},
        )

    def _initiate(self, measurement: np.ndarray, class_id: int, confidence: float) -> _Track:
        height = max(measurement[3], 1.0)
        std = np.r_[[2 * _STD_POSITION * height] * 4, [10 * _STD_VELOCITY * height] * 4]
        mean = np.r_[measurement, np.zeros(4)]
        return _Track(mean, np.diag(std**2), hits=1, last_frame=self._frame, class_id=class_id, confidence=confidence)

    @staticmethod
    def _predict(track: _Track) -> None:
        height = max(track.mean[3], 1.0)
        std = np.r_[[_STD_POSITION * height] * 4, [_STD_VELOCITY * height] * 4]
        track.mean = _F @ track.mean
        track.covariance = _F @ track.covariance @ _F.T + np.diag(std**2)

    def _correct(self, track: _Track, measurement: np.ndarray) -> None:
        height = max(track.mean[3], 1.0)
        position = _STD_POSITION * self._measurement_scale * height
        noise = np.diag([position, position, position * self._size_scale, position * self._size_scale]) ** 2
        projected_cov = _H @ track.covariance @ _H.T + noise
        gain = track.covariance @ _H.T @ np.linalg.inv(projected_cov)
        track.mean = track.mean + gain @ (measurement - _H @ track.mean)
        track.covariance = (np.eye(8) - gain @ _H) @ track.covariance

    def _drop_missing(self) -> None:
        for tracker_id in [t for t, track in self._tracks.items() if self._frame - track.last_frame > self._max_missing]:
            del self._tracks[tracker_id]
        live = set(self._tracks)
        for raw_id in [r for r, kept in self._aliases.items() if kept not in live]:
            del self._aliases[raw_id]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float64)


def _to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state
    w, h = max(w, 1.0), max(h, 1.0)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
