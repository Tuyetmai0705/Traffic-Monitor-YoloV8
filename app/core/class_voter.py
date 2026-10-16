"""Keep one class per track by voting over recent frames."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class ClassVoter:
    """Confidence-weighted majority vote of class ids per tracker id.

    YOLO may label the same van ``car`` in one frame and ``truck`` in the next.
    Counting, speed violations and region sessions all use the voted class, so
    one vehicle never appears under two classes in the report.
    """

    def __init__(self, window: int = 30, max_missing: int = 60) -> None:
        """Vote over the last ``window`` observations; forget tracks unseen for ``max_missing`` frames."""
        if window < 1:
            raise ValueError("window phải ≥ 1")
        self._window = window
        self._max_missing = max_missing
        self._history: dict[int, deque[tuple[int, float]]] = {}
        self._last_frame: dict[int, int] = {}
        self._frame = 0

    def vote(self, tracker_ids: np.ndarray, class_ids: np.ndarray, confidences: np.ndarray | None) -> np.ndarray:
        """Record this frame and return the voted class id of every detection."""
        self._frame += 1
        if confidences is None:
            confidences = np.ones(len(class_ids))
        voted = np.asarray(class_ids, dtype=int).copy()
        for index, (tracker_id, class_id, confidence) in enumerate(zip(tracker_ids, class_ids, confidences)):
            tid = int(tracker_id)
            history = self._history.setdefault(tid, deque(maxlen=self._window))
            history.append((int(class_id), float(confidence)))
            self._last_frame[tid] = self._frame
            scores: dict[int, float] = defaultdict(float)
            for past_class, past_confidence in history:
                scores[past_class] += past_confidence
            voted[index] = max(scores, key=scores.__getitem__)
        for tid in [t for t, seen in self._last_frame.items() if self._frame - seen > self._max_missing]:
            del self._history[tid]
            del self._last_frame[tid]
        return voted

    def reset(self) -> None:
        """Forget every track."""
        self._history.clear()
        self._last_frame.clear()
        self._frame = 0
