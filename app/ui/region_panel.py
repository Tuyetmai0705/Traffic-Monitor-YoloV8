"""Region dashboard: occupancy and counters per ROI, plus the objects inside."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.constants.classes import OVERSTAY_COLOR, roi_color
from app.models.roi import ROI
from app.models.traffic_statistics import TrafficStatistics
from app.ui.object_table import ObjectTable

COLUMNS = ("Vùng", "Hiện", "Đỉnh", "Vào", "Ra", "Mất", "Quá giờ", "TB (s)")
COLUMN_TIPS = ("Mã vùng", "Occupancy: số đối tượng đang trong vùng", "Occupancy cao nhất",
               "Số lượt vào (ENTER)", "Số lượt ra (EXIT)", "Mất dấu khi đang trong vùng (LOST)",
               "Số cảnh báo quá thời gian lưu trú (OVERSTAY)", "Thời gian lưu trú trung bình của các lượt đã kết thúc")


class RegionPanel(QFrame):
    """Shows Occupancy (now) separately from Entry count (cumulative), as the spec requires."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the occupancy table and the object table."""
        super().__init__(parent)
        self.setObjectName("dashboard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        title = QLabel("OCCUPANCY THEO VÙNG")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.summary = QLabel()
        self.summary.setObjectName("metricValue")
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.verticalHeader().setDefaultSectionSize(26)
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(30)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column, width in enumerate((0, 42, 42, 40, 36, 40, 58, 54)):
            if width:
                self.table.setColumnWidth(column, width)
            self.table.horizontalHeaderItem(column).setToolTip(COLUMN_TIPS[column])
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone)
        layout.addWidget(self.table)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        objects_title = QLabel("ĐỐI TƯỢNG TRONG VÙNG")
        objects_title.setObjectName("sectionTitle")
        layout.addWidget(objects_title)
        self.objects = ObjectTable()
        layout.addWidget(self.objects, 1)
        self.set_zones([])

    def set_zones(self, zones: list[ROI]) -> None:
        """Show one zeroed row per configured region (or a hint when there is none)."""
        self._zones = list(zones)
        self.table.setRowCount(len(zones))
        for row, zone in enumerate(zones):
            name = QTableWidgetItem(f"■ {zone.roi_id}")
            name.setForeground(QColor(roi_color(row)))
            name.setToolTip(zone.name)
            self.table.setItem(row, 0, name)
            for column in range(1, len(COLUMNS)):
                self.table.setItem(row, column, self._number("0" if column < 7 else "—"))
        self.table.setFixedHeight(self.table.horizontalHeader().sizeHint().height() + 26 * max(1, len(zones)) + 6)
        self.legend.setText("  ".join(f"<span style='color:{roi_color(i)}'>■</span> {z.roi_id}: {z.name}" for i, z in enumerate(zones)))
        self.objects.reset()
        self.summary.setText("Chưa có vùng ROI. Bấm ROI Zones để vẽ vùng." if not zones else
                             f"{len(zones)} vùng · đang có 0 đối tượng · 0 cảnh báo")

    def update_statistics(self, stats: TrafficStatistics) -> None:
        """Refresh every region row and the object table."""
        by_id = {region.roi_id: region for region in stats.regions}
        for row, zone in enumerate(self._zones):
            region = by_id.get(zone.roi_id)
            if region is None:
                continue
            values = (region.occupancy, region.peak_occupancy, region.entries, region.exits, region.lost, region.overstays)
            for column, value in enumerate(values, start=1):
                self.table.item(row, column).setText(str(value))
            self.table.item(row, 6).setForeground(QColor(OVERSTAY_COLOR if region.overstays else "#E7EBF0"))
            self.table.item(row, 7).setText(f"{region.average_dwell_seconds:.1f}" if region.exits + region.lost else "—")
        inside = sum(region.occupancy for region in stats.regions)
        if self._zones:
            self.summary.setText(f"{len(self._zones)} vùng · đang có {inside} đối tượng · {stats.overstay_alerts} cảnh báo")
        self.objects.update_statistics(stats)

    def reset(self) -> None:
        """Zero every row."""
        self.set_zones(self._zones)

    @staticmethod
    def _number(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return item
