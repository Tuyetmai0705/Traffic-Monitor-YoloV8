"""Realtime traffic statistics panel. Displays values only; computes nothing."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.constants.classes import CLASS_COLORS, VEHICLE_CLASSES
from app.models.traffic_statistics import TrafficStatistics
from app.utils.time_utils import frame_to_timestamp


class DashboardWidget(QFrame):
    """Totals, per-class IN/OUT, violations and speed metrics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the static layout."""
        super().__init__(parent)
        self.setObjectName("dashboard")
        self.setFixedWidth(330)
        self._fps = 25.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(self._section("TRAFFIC STATISTICS"))
        tiles = QGridLayout()
        tiles.setSpacing(8)
        self.total_in = self._tile(tiles, 0, "TOTAL IN", "tileIn")
        self.total_out = self._tile(tiles, 1, "TOTAL OUT", "tileOut")
        layout.addLayout(tiles)

        self.table = QTableWidget(len(VEHICLE_CLASSES), 3)
        self.table.setHorizontalHeaderLabels(["CLASS", "IN", "OUT"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, 64)
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for row, (class_id, name) in enumerate(VEHICLE_CLASSES.items()):
            label = QTableWidgetItem(f"■ {name}")
            label.setForeground(QColor(CLASS_COLORS[class_id]))
            self.table.setItem(row, 0, label)
            for column in (1, 2):
                item = QTableWidgetItem("0")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, column, item)
        self.table.setFixedHeight(
            self.table.horizontalHeader().sizeHint().height() + 26 * len(VEHICLE_CLASSES) + 2 * self.table.frameWidth() + 2
        )
        layout.addWidget(self.table)

        metrics = QGridLayout()
        metrics.setVerticalSpacing(4)
        self.violations = self._metric(metrics, 0, "Violations", "metricBad")
        self.average_speed = self._metric(metrics, 1, "Average speed", "metricValue")
        self.max_speed = self._metric(metrics, 2, "Maximum speed", "metricValue")
        self.speed_limit = self._metric(metrics, 3, "Speed limit", "metricValue")
        self.active_tracks = self._metric(metrics, 4, "Vehicles in frame", "metricValue")
        layout.addLayout(metrics)

        layout.addWidget(self._section("SPEED VIOLATIONS"))
        self.violation_list = QListWidget()
        self.violation_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self.violation_list, 1)
        self.reset()

    def set_context(self, speed_limit: float, fps: float) -> None:
        """Show the configured limit and remember fps for violation timestamps."""
        self.speed_limit.setText(f"{speed_limit:.0f} km/h")
        self._fps = fps if fps > 0 else 25.0

    def update_statistics(self, stats: TrafficStatistics) -> None:
        """Refresh every value from a statistics snapshot."""
        self.total_in.setText(str(stats.total_in))
        self.total_out.setText(str(stats.total_out))
        for row, name in enumerate(VEHICLE_CLASSES.values()):
            counts = stats.by_class.get(name, {"in": 0, "out": 0})
            self.table.item(row, 1).setText(str(counts["in"]))
            self.table.item(row, 2).setText(str(counts["out"]))
        self.violations.setText(str(stats.violations_count))
        self.average_speed.setText(f"{stats.average_speed_kmh:.1f} km/h" if stats.average_speed_kmh else "—")
        self.max_speed.setText(f"{stats.max_speed_kmh:.1f} km/h" if stats.max_speed_kmh else "—")
        self.active_tracks.setText(str(stats.active_tracks))
        if self.violation_list.count() != len(stats.violations) or stats.violations:
            self._fill_violations(stats)

    def reset(self) -> None:
        """Show zeros before a run."""
        self.update_statistics(TrafficStatistics(by_class={name: {"in": 0, "out": 0} for name in VEHICLE_CLASSES.values()}))
        self.violation_list.clear()
        self.violation_list.addItem("Chưa có vi phạm")

    def _fill_violations(self, stats: TrafficStatistics) -> None:
        self.violation_list.clear()
        if not stats.violations:
            self.violation_list.addItem("Chưa có vi phạm")
            return
        for violation in reversed(stats.violations):
            self.violation_list.addItem(
                f"#{violation.tracker_id:<4} {violation.class_name:<10} {violation.max_speed_kmh:6.1f} km/h  "
                f"{frame_to_timestamp(violation.first_frame, self._fps)}"
            )

    @staticmethod
    def _section(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _tile(grid: QGridLayout, column: int, caption: str, value_name: str) -> QLabel:
        frame = QFrame()
        frame.setObjectName("tile")
        box = QVBoxLayout(frame)
        box.setContentsMargins(10, 8, 10, 8)
        caption_label = QLabel(caption)
        caption_label.setObjectName("tileCaption")
        value = QLabel("0")
        value.setObjectName(value_name)
        box.addWidget(caption_label)
        box.addWidget(value)
        grid.addWidget(frame, 0, column)
        return value

    @staticmethod
    def _metric(grid: QGridLayout, row: int, caption: str, value_name: str) -> QLabel:
        grid.addWidget(QLabel(caption), row, 0)
        value = QLabel("—")
        value.setObjectName(value_name)
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(value, row, 1)
        return value
