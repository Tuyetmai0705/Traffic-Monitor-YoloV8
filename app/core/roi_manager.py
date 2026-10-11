"""Region membership: which tracked objects are inside which ROI."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import supervision as sv

from app.models.roi import ANCHORS, ROI
from app.utils.geometry import points_in_polygon, validate_polygon

_ANCHOR_POSITIONS: dict[str, sv.Position] = {
    "bottom_center": sv.Position.BOTTOM_CENTER,
    "center": sv.Position.CENTER,
}


class ROIManager:
    """Hold ROI polygons and evaluate anchor-in-polygon membership.

    Rule (kept consistent everywhere): an object is inside a region when its
    anchor point lies inside or on the boundary of the polygon AND its class is
    allowed by that region. Regions are independent, so one object may be
    inside several overlapping regions at the same time.
    """

    def __init__(self, zones: Sequence[ROI], anchor: str = "bottom_center") -> None:
        """Validate polygons (processing-frame pixels) and the anchor name."""
        if anchor not in ANCHORS:
            raise ValueError(f"anchor phải là một trong {ANCHORS}")
        self._zones = list(zones)
        self._anchor = anchor
        self._polygons = {zone.roi_id: validate_polygon(zone.polygon) for zone in self._zones}

    @property
    def zones(self) -> list[ROI]:
        """Configured regions in display order."""
        return list(self._zones)

    @property
    def anchor(self) -> str:
        """Anchor used for membership (``bottom_center`` or ``center``)."""
        return self._anchor

    def get_zone(self, roi_id: str) -> ROI:
        """Return the region with ``roi_id``. Raises KeyError if unknown."""
        for zone in self._zones:
            if zone.roi_id == roi_id:
                return zone
        raise KeyError(roi_id)

    def anchor_points(self, detections: sv.Detections) -> np.ndarray:
        """Return the (N, 2) anchor point of every detection."""
        if len(detections) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return detections.get_anchors_coordinates(_ANCHOR_POSITIONS[self._anchor])

    def evaluate(self, detections: sv.Detections, class_names: Sequence[str]) -> dict[str, np.ndarray]:
        """Return ``{roi_id: bool mask}`` aligned with ``detections``."""
        anchors = self.anchor_points(detections)
        names = list(class_names)
        memberships: dict[str, np.ndarray] = {}
        for zone in self._zones:
            inside = points_in_polygon(anchors, self._polygons[zone.roi_id])
            allowed = np.array([zone.allows(name) for name in names], dtype=bool)
            memberships[zone.roi_id] = inside & allowed if len(names) else inside
        return memberships
