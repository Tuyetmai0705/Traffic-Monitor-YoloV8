"""ROI polygon validation and anchor-in-polygon membership."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.roi_manager import ROIManager
from app.models.roi import ROI
from app.utils.geometry import points_in_polygon, polygon_inside_frame, validate_polygon
from tests.conftest import make_detections

SQUARE = [[100, 100], [300, 100], [300, 300], [100, 300]]


def zone(roi_id: str = "zone_a", polygon=SQUARE, classes=(), max_dwell: float = 0.0) -> ROI:
    return ROI(roi_id=roi_id, name=roi_id, polygon=[list(p) for p in polygon], allowed_classes=list(classes), max_dwell_seconds=max_dwell)


def test_points_in_polygon_counts_boundary_as_inside() -> None:
    points = np.array([[200, 200], [100, 200], [300, 300], [301, 200], [50, 50]], dtype=float)
    assert points_in_polygon(points, np.array(SQUARE)).tolist() == [True, True, True, False, False]
    assert points_in_polygon(np.empty((0, 2)), np.array(SQUARE)).shape == (0,)


def test_concave_polygon_membership() -> None:
    l_shape = [[0, 0], [200, 0], [200, 100], [100, 100], [100, 200], [0, 200]]
    validate_polygon(l_shape)
    inside = points_in_polygon(np.array([[50, 150], [150, 150], [150, 50]], dtype=float), np.array(l_shape))
    assert inside.tolist() == [True, False, True]


@pytest.mark.parametrize(
    ("points", "reason"),
    [
        ([[0, 0], [100, 0]], "ít nhất 3 điểm"),
        ([[0, 0], [100, 0], [200, 0]], "diện tích"),
        ([[0, 0], [100, 100], [100, 0], [0, 100]], "cắt nhau"),
        ([[0, 0], [1, 0], [float("nan"), 1]], "hữu hạn"),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "2 tọa độ"),
    ],
)
def test_validate_polygon_rejects(points, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        validate_polygon(points)


def test_polygon_inside_frame() -> None:
    assert polygon_inside_frame(SQUARE, 640, 480)
    assert not polygon_inside_frame([[0, 0], [700, 0], [0, 100]], 640, 480)


def test_bottom_center_anchor_decides_membership() -> None:
    # Box centre (200, 50) is outside the square; its bottom centre (200, 150) is inside.
    detections = make_detections([[150, 0, 250, 150]], [2], tracker_ids=[1])
    assert ROIManager([zone()], "bottom_center").evaluate(detections, ["car"])["zone_a"].tolist() == [True]
    assert ROIManager([zone()], "center").evaluate(detections, ["car"])["zone_a"].tolist() == [False]


def test_allowed_classes_filter() -> None:
    detections = make_detections([[150, 150, 250, 250], [150, 150, 250, 250]], [0, 2], tracker_ids=[1, 2])
    people_only = ROIManager([zone(classes=["person"])])
    assert people_only.evaluate(detections, ["person", "car"])["zone_a"].tolist() == [True, False]
    every_class = ROIManager([zone()])
    assert every_class.evaluate(detections, ["person", "car"])["zone_a"].tolist() == [True, True]


def test_overlapping_regions_are_independent() -> None:
    zones = [zone("zone_a"), zone("zone_b", polygon=[[200, 200], [400, 200], [400, 400], [200, 400]])]
    detections = make_detections([[240, 200, 260, 250], [120, 200, 140, 250]], [2, 2], tracker_ids=[1, 2])
    result = ROIManager(zones).evaluate(detections, ["car", "car"])
    assert result["zone_a"].tolist() == [True, True]
    assert result["zone_b"].tolist() == [True, False]


def test_empty_detections_and_invalid_setup() -> None:
    manager = ROIManager([zone()])
    empty = make_detections(np.empty((0, 4)), [], tracker_ids=[])
    assert manager.evaluate(empty, [])["zone_a"].shape == (0,)
    assert manager.get_zone("zone_a").name == "zone_a"
    with pytest.raises(ValueError):
        ROIManager([zone()], "top_left")
    with pytest.raises(ValueError):
        ROIManager([zone(polygon=[[0, 0], [10, 0]])])


def test_roi_from_dict_and_scaling() -> None:
    roi = ROI.from_dict({"roi_id": "zone_a", "name": "Khu vực A", "polygon": [[100.4, 100], [400, 100], [450, 350]]})
    assert roi.polygon[0] == [100, 100]
    assert roi.allowed_classes == [] and roi.max_dwell_seconds == 0.0
    assert roi.allows("person") and roi.allows("car")
    assert roi.scaled(0.5).polygon == [[50, 50], [200, 50], [225, 175]]
    with pytest.raises(ValueError):
        ROI.from_dict({"roi_id": "x", "name": "x"})
