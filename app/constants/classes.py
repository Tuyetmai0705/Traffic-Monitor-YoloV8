"""COCO class ids used by the application and their display colours."""

PERSON = 0
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7

VEHICLE_CLASSES: dict[int, str] = {
    CAR: "car",
    MOTORCYCLE: "motorcycle",
    BUS: "bus",
    TRUCK: "truck",
}

# Every class the detector may keep: vehicles plus people for region monitoring.
OBJECT_CLASSES: dict[int, str] = {PERSON: "person", **VEHICLE_CLASSES}
CLASS_IDS_BY_NAME: dict[str, int] = {name: class_id for class_id, name in OBJECT_CLASSES.items()}

# Hex colours per class, shared by frame annotation and the dashboard.
CLASS_COLORS: dict[int, str] = {
    PERSON: "#F28DB2",
    CAR: "#5AA2FF",
    MOTORCYCLE: "#C39BFF",
    BUS: "#F7BE45",
    TRUCK: "#45CFC6",
}

VIOLATION_COLOR = "#FF4B3E"
OVERSTAY_COLOR = "#FF9F1C"
LINE_COLOR = "#FFD23F"

# Colours assigned to regions of interest in configuration order.
ROI_COLORS: tuple[str, ...] = ("#3DDC97", "#FF7AC6", "#4CC9F0", "#F9C74F", "#B5179E", "#90BE6D")


def roi_color(index: int) -> str:
    """Return the display colour of the ``index``-th region."""
    return ROI_COLORS[index % len(ROI_COLORS)]
