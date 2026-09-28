"""COCO class ids used for vehicles and their display colours."""

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

# Hex colours per class, shared by frame annotation and the dashboard.
CLASS_COLORS: dict[int, str] = {
    CAR: "#5AA2FF",
    MOTORCYCLE: "#C39BFF",
    BUS: "#F7BE45",
    TRUCK: "#45CFC6",
}

VIOLATION_COLOR = "#FF4B3E"
LINE_COLOR = "#FFD23F"
