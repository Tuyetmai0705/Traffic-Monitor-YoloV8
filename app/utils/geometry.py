"""Geometry validation shared by configuration and speed estimation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_MIN_AREA = 1.0


def validate_quadrilateral(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return points as a (4, 2) float32 array or raise ValueError with a reason.

    The quadrilateral must have 4 finite points, be convex, have no three
    collinear consecutive points and a non-zero area.
    """
    try:
        array = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("tọa độ phải là số") from exc
    if array.shape != (4, 2):
        raise ValueError("cần đúng 4 điểm (x, y)")
    if not np.all(np.isfinite(array)):
        raise ValueError("tọa độ phải là số hữu hạn")

    signs = []
    for i in range(4):
        a, b, c = array[i], array[(i + 1) % 4], array[(i + 2) % 4]
        cross = float((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if abs(cross) < 1e-6:
            raise ValueError("có 3 điểm thẳng hàng")
        signs.append(np.sign(cross))
    if len(set(signs)) != 1:
        raise ValueError("4 điểm phải tạo thành tứ giác lồi theo thứ tự A, B, C, D")

    x, y = array[:, 0], array[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < _MIN_AREA:
        raise ValueError("diện tích tứ giác quá nhỏ")
    return array


def validate_line(start: Sequence[float], end: Sequence[float]) -> None:
    """Raise ValueError when a line segment is malformed or has zero length."""
    if len(start) != 2 or len(end) != 2:
        raise ValueError("mỗi điểm cần 2 tọa độ (x, y)")
    if tuple(start) == tuple(end):
        raise ValueError("điểm đầu và điểm cuối trùng nhau")


def line_inside_frame(start: Sequence[float], end: Sequence[float], width: int, height: int) -> bool:
    """Return True when both endpoints lie within the frame bounds."""
    return all(0 <= p[0] <= width and 0 <= p[1] <= height for p in (start, end))
