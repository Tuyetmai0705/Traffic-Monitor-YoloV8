"""Table of objects currently inside regions: track, class, region, entry and dwell time."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QWidget

from app.constants.classes import OVERSTAY_COLOR
from app.models.traffic_statistics import TrafficStatistics
from app.utils.time_utils import format_timestamp

COLUMNS = ("Track", "Lớp", "Vùng", "Vào lúc", "Lưu trú", "Trạng thái")


class ObjectTable(QTableWidget):
    """One row per open session; overstaying rows are highlighted."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty, read-only table."""
        super().__init__(0, len(COLUMNS), parent)
        self.setHorizontalHeaderLabels(COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.verticalHeader().setDefaultSectionSize(24)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

    def update_statistics(self, stats: TrafficStatistics) -> None:
        """Show every open session of every region, longest dwell first."""
        sessions = [s for region in stats.regions for s in region.active_sessions]
        sessions.sort(key=lambda s: s.dwell_seconds, reverse=True)
        self.setRowCount(len(sessions))
        for row, session in enumerate(sessions):
            status = "QUÁ GIỜ" if session.overstay else "Trong vùng"
            values = (
                f"#{session.track_id}", session.class_name, session.roi_id,
                format_timestamp(session.entry_time), f"{session.dwell_seconds:.1f} s", status,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (0, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if session.overstay:
                    item.setForeground(QColor(OVERSTAY_COLOR))
                self.setItem(row, column, item)

    def reset(self) -> None:
        """Remove every row."""
        self.setRowCount(0)
