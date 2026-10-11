"""Geometry helpers shared by configuration, speed estimation and region monitoring."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
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


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """True when the open segments p1-p2 and q1-q2 properly cross."""

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Shoelace area of a polygon in square pixels."""
    array = np.asarray(points, dtype=np.float64)
    x, y = array[:, 0], array[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def validate_polygon(points: Sequence[Sequence[float]], min_area: float = 100.0) -> np.ndarray:
    """Return an ROI polygon as an (N, 2) int32 array or raise ValueError with a reason.

    The polygon needs at least 3 finite points, a minimum area and must not
    cross itself (concave shapes are allowed).
    """
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("tọa độ phải là số") from exc
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("mỗi điểm cần 2 tọa độ (x, y)")
    if len(array) < 3:
        raise ValueError("cần ít nhất 3 điểm")
    if not np.all(np.isfinite(array)):
        raise ValueError("tọa độ phải là số hữu hạn")
    count = len(array)
    for i in range(count):
        for j in range(i + 1, count):
            if abs(i - j) == 1 or {i, j} == {0, count - 1}:
                continue  # adjacent edges share a vertex
            if _segments_intersect(array[i], array[(i + 1) % count], array[j], array[(j + 1) % count]):
                raise ValueError("các cạnh của vùng cắt nhau")
    if polygon_area(array) < min_area:  # after the crossing check: a bow-tie has ~zero signed area
        raise ValueError("diện tích vùng quá nhỏ")
    return np.round(array).astype(np.int32)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Boolean mask of ``points`` (N, 2) inside ``polygon``; points on the boundary count as inside."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    contour = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    return np.array(
        [cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0 for x, y in points], dtype=bool
    )


def polygon_inside_frame(points: Sequence[Sequence[float]], width: int, height: int) -> bool:
    """Return True when every vertex lies within the frame bounds."""
    return all(0 <= x <= width and 0 <= y <= height for x, y in points)
