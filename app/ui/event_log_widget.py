"""Event log: ENTER / EXIT / LOST / OVERSTAY with a type filter."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.constants.classes import OVERSTAY_COLOR
from app.models.event import EventType, RegionEvent
from app.models.traffic_statistics import TrafficStatistics
from app.utils.time_utils import format_timestamp

COLUMNS = ("Thời điểm", "Sự kiện", "Vùng", "Track", "Lớp", "Lưu trú")
EVENT_COLORS = {
    EventType.ENTER: "#4FD69C",
    EventType.EXIT: "#9AA5B2",
    EventType.LOST: "#FF6B63",
    EventType.OVERSTAY: OVERSTAY_COLOR,
}
MAX_ROWS = 2000


class EventLogWidget(QFrame):
    """Newest events first. Only events not shown yet are appended on each update."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the filter and the table."""
        super().__init__(parent)
        self.setObjectName("dashboard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        top = QHBoxLayout()
        title = QLabel("NHẬT KÝ SỰ KIỆN")
        title.setObjectName("sectionTitle")
        top.addWidget(title)
        top.addStretch(1)
        self.filter = QComboBox()
        self.filter.addItem("Tất cả", None)
        for event_type in EventType:
            self.filter.addItem(event_type.value, event_type)
        self.filter.currentIndexChanged.connect(self._rebuild)
        top.addWidget(self.filter)
        layout.addLayout(top)
        self.counter = QLabel()
        layout.addWidget(self.counter)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.verticalHeader().setDefaultSectionSize(24)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        self._events: list[RegionEvent] = []
        self._seen: set[str] = set()
        self._update_counter()

    @property
    def events(self) -> list[RegionEvent]:
        """Every event received in this run, oldest first."""
        return list(self._events)

    def update_statistics(self, stats: TrafficStatistics) -> list[RegionEvent]:
        """Append events not shown yet and return them (the window uses them for alerts)."""
        new = [event for event in stats.recent_events if event.event_id not in self._seen]
        for event in new:
            self._seen.add(event.event_id)
            self._events.append(event)
            if self._matches(event):
                self._insert(event)
        if new:
            self._update_counter()
        return new

    def reset(self) -> None:
        """Clear the log before a new run."""
        self._events.clear()
        self._seen.clear()
        self.table.setRowCount(0)
        self._update_counter()

    # ------------------------------------------------------------------ helpers
    def _matches(self, event: RegionEvent) -> bool:
        wanted = self.filter.currentData()
        return wanted is None or event.event_type is wanted

    def _insert(self, event: RegionEvent) -> None:
        self.table.insertRow(0)
        dwell = "" if event.dwell_seconds is None else f"{event.dwell_seconds:.1f} s"
        values = (format_timestamp(event.timestamp), event.event_type.value, event.roi_id, f"#{event.track_id}", event.class_name, dwell)
        color = QColor(EVENT_COLORS[event.event_type])
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 1:
                item.setForeground(color)
            if column in (3, 5):
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(0, column, item)
        if self.table.rowCount() > MAX_ROWS:
            self.table.setRowCount(MAX_ROWS)

    def _rebuild(self) -> None:
        self.table.setRowCount(0)
        for event in self._events[-MAX_ROWS:]:
            if self._matches(event):
                self._insert(event)

    def _update_counter(self) -> None:
        counts = {kind: sum(1 for e in self._events if e.event_type is kind) for kind in EventType}
        self.counter.setText("  ·  ".join(f"{kind.value} {counts[kind]}" for kind in EventType))
