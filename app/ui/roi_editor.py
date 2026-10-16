"""ROI editor: draw region polygons on the video frame and set their rules."""

from __future__ import annotations

import string

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import AppConfig, RegionsConfig
from app.constants import messages as msg
from app.constants.classes import CLASS_IDS_BY_NAME, OBJECT_CLASSES, roi_color
from app.core.exceptions import InvalidConfigError
from app.models.roi import ROI
from app.services.config_service import ConfigService
from app.utils.display import to_qimage

_HANDLE_RADIUS = 7
_ANCHOR_LABELS = {"center": "Tâm box (center)", "bottom_center": "Điểm đáy giữa box (bottom_center)"}


class PolygonCanvas(QWidget):
    """Shows a frame and lets the user add, drag and remove vertices of the selected polygon.

    All coordinates handed in and out are original-frame pixels; the widget
    maps them to its own scaled view, so drawing on a resized preview stays
    correct (technical document §8, "ROI drawn on preview").
    """

    changed = Signal()

    def __init__(self, frame: np.ndarray | None, parent: QWidget | None = None) -> None:
        """Display ``frame`` (BGR) or a hint when no video is open."""
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._image: QImage | None = to_qimage(frame) if frame is not None else None
        self._zones: list[ROI] = []
        self._selected = -1
        self._drag_index: int | None = None

    @property
    def has_frame(self) -> bool:
        """True when a frame is available to draw on."""
        return self._image is not None

    def set_zones(self, zones: list[ROI], selected: int) -> None:
        """Show ``zones`` (edited in place) and highlight ``selected``."""
        self._zones = zones
        self._selected = selected
        self.update()

    # ------------------------------------------------------------- coordinates
    def _view_rect(self) -> QRectF:
        if self._image is None:
            return QRectF(self.rect())
        scale = min(self.width() / self._image.width(), self.height() / self._image.height())
        width, height = self._image.width() * scale, self._image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _scale(self) -> float:
        return self._view_rect().width() / self._image.width() if self._image is not None else 1.0

    def to_widget(self, point: list[int] | tuple[float, float]) -> QPointF:
        """Map a frame pixel to widget coordinates."""
        rect = self._view_rect()
        return QPointF(rect.x() + point[0] * self._scale(), rect.y() + point[1] * self._scale())

    def to_frame(self, position: QPointF) -> list[int]:
        """Map a widget position to a frame pixel, clamped to the frame."""
        rect = self._view_rect()
        assert self._image is not None
        x = (position.x() - rect.x()) / self._scale()
        y = (position.y() - rect.y()) / self._scale()
        return [int(round(min(max(x, 0), self._image.width() - 1))), int(round(min(max(y, 0), self._image.height() - 1)))]

    # ------------------------------------------------------------------ editing
    def _current(self) -> ROI | None:
        return self._zones[self._selected] if 0 <= self._selected < len(self._zones) else None

    def _vertex_at(self, position: QPointF) -> int | None:
        zone = self._current()
        if zone is None:
            return None
        for index, point in enumerate(zone.polygon):
            if (self.to_widget(point) - position).manhattanLength() <= _HANDLE_RADIUS * 2:
                return index
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        """Left: grab a vertex or add a point. Right: remove the last point."""
        zone = self._current()
        if self._image is None or zone is None or not self._view_rect().contains(event.position()):
            return
        if event.button() == Qt.MouseButton.RightButton:
            if zone.polygon:
                zone.polygon.pop()
                self.changed.emit()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._drag_index = self._vertex_at(event.position())
            if self._drag_index is None:
                zone.polygon.append(self.to_frame(event.position()))
                self.changed.emit()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        """Drag the grabbed vertex."""
        zone = self._current()
        if zone is not None and self._drag_index is not None and self._image is not None:
            zone.polygon[self._drag_index] = self.to_frame(event.position())
            self.changed.emit()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        """Drop the grabbed vertex."""
        self._drag_index = None

    # ------------------------------------------------------------------ drawing
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        """Draw the frame, every polygon and the selected polygon's handles."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0B0D10"))
        if self._image is None:
            painter.setPen(QColor("#9AA5B2"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, msg.ROI_NEED_VIDEO)
            return
        painter.drawImage(self._view_rect(), self._image)
        for index, zone in enumerate(self._zones):
            color = QColor(roi_color(index))
            selected = index == self._selected
            points = [self.to_widget(point) for point in zone.polygon]
            if len(points) >= 3:
                fill = QColor(color)
                fill.setAlpha(70 if selected else 40)
                painter.setBrush(QBrush(fill))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color, 3 if selected else 2, Qt.PenStyle.SolidLine if len(points) >= 3 else Qt.PenStyle.DashLine))
            if len(points) >= 3:
                painter.drawPolygon(QPolygonF(points))
            elif len(points) == 2:
                painter.drawLine(points[0], points[1])
            if points:
                painter.setPen(QColor("#FFFFFF"))
                top = min(points, key=lambda p: p.y())
                painter.drawText(top + QPointF(6, -8), f"{zone.roi_id} · {zone.name}")
            if selected:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor("#0B0D10"), 1))
                for number, point in enumerate(points, start=1):
                    painter.drawEllipse(point, _HANDLE_RADIUS, _HANDLE_RADIUS)
                    painter.drawText(point + QPointF(9, 4), str(number))
        painter.setPen(QColor("#E7EBF0"))
        painter.drawText(QRectF(8, self.height() - 26, self.width() - 16, 20), Qt.AlignmentFlag.AlignLeft, msg.ROI_POINT_HINT)


class ROIEditorDialog(QDialog):
    """Add, edit and delete regions: polygon, name, classes and dwell limit."""

    def __init__(
        self,
        config: AppConfig,
        config_service: ConfigService,
        preview_frame: np.ndarray | None,
        video_name: str | None,
        parent: QWidget | None = None,
    ) -> None:
        """Edit a copy of ``config.regions``; nothing changes until Save."""
        super().__init__(parent)
        self.setWindowTitle("ROI Zones — Giám sát đối tượng trong vùng")
        self.resize(1320, 780)
        self._base = config.copy()
        self._service = config_service
        self._zones: list[ROI] = [ROI.from_dict(zone.to_dict()) for zone in config.regions.zones]
        self._result: AppConfig | None = None
        self._loading = False

        self.canvas = PolygonCanvas(preview_frame)
        self.canvas.changed.connect(self._on_polygon_changed)
        side = QVBoxLayout()
        side.addWidget(self._general_group(config.regions))
        side.addWidget(self._zone_list_group())
        side.addWidget(self._zone_group())
        self.class_warning = QLabel()
        self.class_warning.setWordWrap(True)
        self.class_warning.setStyleSheet("color: #E6B32E;")
        side.addWidget(self.class_warning)
        self.save_for_video = QCheckBox(f"Lưu vùng riêng cho video {video_name}" if video_name else "Lưu vùng riêng cho video (chưa mở video)")
        self.save_for_video.setEnabled(video_name is not None)
        self.save_for_video.setChecked(video_name is not None)
        side.addWidget(self.save_for_video)
        side.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        side.addWidget(buttons)

        panel = QWidget()
        panel.setFixedWidth(380)
        panel.setLayout(side)
        root = QHBoxLayout(self)
        root.addWidget(self.canvas, 1)
        root.addWidget(panel)
        self._refresh_list(0 if self._zones else -1)

    # ------------------------------------------------------------------ public
    def result_config(self) -> AppConfig | None:
        """The validated configuration after Save, otherwise None."""
        return self._result

    def save_for_video_requested(self) -> bool:
        """True when the regions should also be stored in the video's calibration file."""
        return self.save_for_video.isEnabled() and self.save_for_video.isChecked()

    @property
    def zones(self) -> list[ROI]:
        """Regions being edited."""
        return self._zones

    def add_zone(self) -> ROI:
        """Append a new empty region with a unique id and select it."""
        used = {zone.roi_id for zone in self._zones}
        letter = next((c for c in string.ascii_lowercase if f"zone_{c}" not in used), str(len(self._zones) + 1))
        zone = ROI(roi_id=f"zone_{letter}", name=f"Vùng {letter.upper()}", polygon=[], allowed_classes=[], max_dwell_seconds=0.0)
        self._zones.append(zone)
        self._refresh_list(len(self._zones) - 1)
        return zone

    def delete_zone(self) -> None:
        """Remove the selected region."""
        index = self.zone_list.currentRow()
        if 0 <= index < len(self._zones):
            del self._zones[index]
            self._refresh_list(min(index, len(self._zones) - 1))

    def accept(self) -> None:  # noqa: D401 (Qt override)
        """Validate every region and the whole config before closing."""
        config = self._base.copy()
        config.regions = RegionsConfig(
            enabled=self.enabled.isChecked(),
            anchor=self.anchor.currentData(),
            grace_seconds=self.grace.value(),
            min_confirm_frames=self.confirm.value(),
            zones=[ROI.from_dict(zone.to_dict()) for zone in self._zones],
        )
        try:
            self._service.validate(config)
        except InvalidConfigError as exc:
            self._select_zone_of_field(exc.field)
            QMessageBox.warning(self, "ROI không hợp lệ", str(exc))
            return
        self._result = config
        super().accept()

    # ------------------------------------------------------------------ layout
    def _general_group(self, regions: RegionsConfig) -> QGroupBox:
        group = QGroupBox("Thiết lập chung")
        form = QFormLayout(group)
        self.enabled = QCheckBox("Bật giám sát vùng")
        self.enabled.setChecked(regions.enabled)
        self.anchor = QComboBox()
        for key, label in _ANCHOR_LABELS.items():
            self.anchor.addItem(label, key)
        self.anchor.setCurrentIndex(max(0, self.anchor.findData(regions.anchor)))
        self.grace = QDoubleSpinBox()
        self.grace.setRange(0.0, 10.0)
        self.grace.setSingleStep(0.1)
        self.grace.setSuffix(" s")
        self.grace.setValue(regions.grace_seconds)
        self.grace.setToolTip("Đối tượng phải ở ngoài vùng (hoặc mất dấu) lâu hơn thời gian này mới tính là đã ra.")
        self.confirm = QSpinBox()
        self.confirm.setRange(1, 50)
        self.confirm.setSuffix(" frame")
        self.confirm.setValue(regions.min_confirm_frames)
        self.confirm.setToolTip("Đối tượng phải ở trong vùng liên tiếp số frame này mới tính là đã vào (lọc box nhiễu).")
        form.addRow(self.enabled)
        form.addRow("Điểm xét trong vùng", self.anchor)
        form.addRow("Xác nhận vào vùng", self.confirm)
        form.addRow("Grace period", self.grace)
        return group

    def _zone_list_group(self) -> QGroupBox:
        group = QGroupBox("Danh sách vùng")
        box = QVBoxLayout(group)
        self.zone_list = QListWidget()
        self.zone_list.setFixedHeight(110)
        self.zone_list.currentRowChanged.connect(self._on_zone_selected)
        box.addWidget(self.zone_list)
        row = QHBoxLayout()
        self.add_button = QPushButton("Thêm vùng")
        self.add_button.clicked.connect(self.add_zone)
        self.delete_button = QPushButton("Xóa vùng")
        self.delete_button.clicked.connect(self.delete_zone)
        row.addWidget(self.add_button)
        row.addWidget(self.delete_button)
        box.addLayout(row)
        return group

    def _zone_group(self) -> QGroupBox:
        self.zone_group = QGroupBox("Vùng đang chọn")
        form = QFormLayout(self.zone_group)
        self.roi_id = QLineEdit()
        self.roi_id.setToolTip("Mã ASCII, được vẽ lên video (cv2 không vẽ được tiếng Việt).")
        self.roi_id.textEdited.connect(self._on_fields_changed)
        self.name = QLineEdit()
        self.name.textEdited.connect(self._on_fields_changed)
        classes = QWidget()
        grid = QGridLayout(classes)
        grid.setContentsMargins(0, 0, 0, 0)
        self.class_boxes: dict[str, QCheckBox] = {}
        for index, name in enumerate(OBJECT_CLASSES.values()):
            box = QCheckBox(name)
            box.toggled.connect(self._on_fields_changed)
            grid.addWidget(box, index // 3, index % 3)
            self.class_boxes[name] = box
        self.max_dwell = QDoubleSpinBox()
        self.max_dwell.setRange(0.0, 86400.0)
        self.max_dwell.setDecimals(1)
        self.max_dwell.setSuffix(" s")
        self.max_dwell.setSpecialValueText("Tắt cảnh báo")
        self.max_dwell.valueChanged.connect(self._on_fields_changed)
        self.points_label = QLabel()
        point_buttons = QHBoxLayout()
        self.undo_button = QPushButton("Xóa điểm cuối")
        self.undo_button.clicked.connect(self._undo_point)
        self.clear_button = QPushButton("Vẽ lại")
        self.clear_button.clicked.connect(self._clear_points)
        point_buttons.addWidget(self.undo_button)
        point_buttons.addWidget(self.clear_button)
        form.addRow("Mã vùng", self.roi_id)
        form.addRow("Tên vùng", self.name)
        form.addRow("Lớp theo dõi", classes)
        form.addRow("", QLabel("(không chọn lớp nào = theo dõi mọi lớp)"))
        form.addRow("Ngưỡng lưu trú", self.max_dwell)
        form.addRow("Polygon", self.points_label)
        form.addRow(point_buttons)
        return self.zone_group

    # --------------------------------------------------------------- behaviour
    def _current(self) -> ROI | None:
        index = self.zone_list.currentRow()
        return self._zones[index] if 0 <= index < len(self._zones) else None

    def _refresh_list(self, select: int) -> None:
        self.zone_list.blockSignals(True)
        self.zone_list.clear()
        for index, zone in enumerate(self._zones):
            item = QListWidgetItem(f"■ {zone.roi_id} — {zone.name}  ({len(zone.polygon)} điểm)")
            item.setForeground(QColor(roi_color(index)))
            self.zone_list.addItem(item)
        self.zone_list.setCurrentRow(select)
        self.zone_list.blockSignals(False)
        self._on_zone_selected(select)

    def _on_zone_selected(self, index: int) -> None:
        zone = self._zones[index] if 0 <= index < len(self._zones) else None
        self.zone_group.setEnabled(zone is not None)
        self.delete_button.setEnabled(zone is not None)
        self._loading = True
        self.roi_id.setText(zone.roi_id if zone else "")
        self.name.setText(zone.name if zone else "")
        for name, box in self.class_boxes.items():
            box.setChecked(bool(zone and name in zone.allowed_classes))
        self.max_dwell.setValue(zone.max_dwell_seconds if zone else 0.0)
        self._loading = False
        self._update_points_label()
        self._update_class_warning()
        self.canvas.set_zones(self._zones, index)

    def _on_fields_changed(self) -> None:
        zone = self._current()
        if zone is None or self._loading:
            return
        zone.roi_id = self.roi_id.text().strip()
        zone.name = self.name.text().strip()
        zone.allowed_classes = [name for name, box in self.class_boxes.items() if box.isChecked()]
        zone.max_dwell_seconds = self.max_dwell.value()
        self._update_list_item()
        self._update_class_warning()
        self.canvas.update()

    def _on_polygon_changed(self) -> None:
        self._update_points_label()
        self._update_list_item()

    def _undo_point(self) -> None:
        zone = self._current()
        if zone and zone.polygon:
            zone.polygon.pop()
            self._on_polygon_changed()
            self.canvas.update()

    def _clear_points(self) -> None:
        zone = self._current()
        if zone:
            zone.polygon.clear()
            self._on_polygon_changed()
            self.canvas.update()

    def _update_list_item(self) -> None:
        index = self.zone_list.currentRow()
        zone = self._current()
        if zone is not None:
            self.zone_list.item(index).setText(f"■ {zone.roi_id} — {zone.name}  ({len(zone.polygon)} điểm)")

    def _update_points_label(self) -> None:
        zone = self._current()
        if zone is None:
            self.points_label.setText("—")
        elif len(zone.polygon) < 3:
            self.points_label.setText(f"{len(zone.polygon)} điểm — cần ít nhất 3 điểm")
        else:
            self.points_label.setText(f"{len(zone.polygon)} điểm: " + " ".join(f"({x},{y})" for x, y in zone.polygon[:4])
                                      + (" …" if len(zone.polygon) > 4 else ""))

    def _update_class_warning(self) -> None:
        """Warn when a region watches a class the detector does not keep (Settings → classes)."""
        detected = {OBJECT_CLASSES[class_id] for class_id in self._base.classes}
        wanted = {name for zone in self._zones for name in zone.allowed_classes}
        missing = sorted(wanted - detected, key=lambda name: CLASS_IDS_BY_NAME[name])
        self.class_warning.setText(msg.ROI_CLASS_NOT_DETECTED.format(names=", ".join(missing)) if missing else "")

    def _select_zone_of_field(self, field: str) -> None:
        if field.startswith("regions.zones[") and "]" in field:
            index = int(field[len("regions.zones["):field.index("]")])
            if 0 <= index < len(self._zones):
                self.zone_list.setCurrentRow(index)
