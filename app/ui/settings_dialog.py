"""Settings form for model, detection, speed, calibration and outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import AppConfig
from app.constants.classes import OBJECT_CLASSES
from app.core.exceptions import InvalidConfigError, TrafficMonitorError
from app.services.config_service import ConfigService
from app.utils.device import available_devices
from app.utils.display import draw_calibration, resize_to_width, to_qimage
from app.utils.file_utils import PROJECT_ROOT, resolve_path, to_display_path

_COORD_RANGE = 20000.0


def _spin(minimum: float, maximum: float, value: float, step: float = 1.0, decimals: int = 0) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    return box


class SettingsDialog(QDialog):
    """Edit every configurable parameter; validation is delegated to ConfigService."""

    def __init__(
        self,
        config: AppConfig,
        config_service: ConfigService,
        preview_frame: np.ndarray | None = None,
        video_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the form populated from ``config``."""
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(760)
        self._service = config_service
        self._base = config.copy()
        self._result: AppConfig | None = None
        self._preview_frame = preview_frame
        self._fields: dict[str, QWidget] = {}

        tabs = QTabWidget()
        tabs.addTab(self._model_tab(), "Model")
        tabs.addTab(self._detection_tab(), "Phát hiện && tốc độ")
        tabs.addTab(self._calibration_tab(video_name), "Hiệu chỉnh")
        tabs.addTab(self._output_tab(), "Đầu ra")
        self.tabs = tabs

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.restore_button = buttons.addButton("Khôi phục mặc định", QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.restore_button.clicked.connect(self._restore_defaults)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)
        self._load(config)

    # ------------------------------------------------------------------ public
    def result_config(self) -> AppConfig | None:
        """Validated config after Save, otherwise None."""
        return self._result

    def save_calibration_requested(self) -> bool:
        """True when the user asked to store calibration for the current video."""
        return self.save_calibration.isEnabled() and self.save_calibration.isChecked()

    def collect(self) -> AppConfig:
        """Build a config from the form values (not yet validated)."""
        config = self._base.copy()
        config.model.path = self.model_path.text().strip()
        config.model.device = self.device.currentData()
        config.model.half = self.half.isChecked()
        config.model.agnostic_nms = self.agnostic_nms.isChecked()
        config.model.imgsz = self.imgsz.value()
        config.model.confidence = round(self.confidence.value(), 2)
        config.model.iou = round(self.iou.value(), 2)
        config.classes = [class_id for class_id, box in self.class_boxes.items() if box.isChecked()]
        config.speed.limit_kmh = self.speed_limit.value()
        config.speed.window_seconds = self.window_seconds.value()
        config.perspective.source_points = [[x.value(), y.value()] for x, y in self.points]
        config.perspective.road_width_m = self.road_width.value()
        config.perspective.road_length_m = self.road_length.value()
        config.counting_line.start = [int(self.line[0].value()), int(self.line[1].value())]
        config.counting_line.end = [int(self.line[2].value()), int(self.line[3].value())]
        config.output.video_path = self.video_path.text().strip()
        config.output.report_path = self.report_path.text().strip()
        config.processing.max_width = self.max_width.value()
        return config

    def accept(self) -> None:  # noqa: D102 (Qt override)
        config = self.collect()
        self._clear_invalid()
        try:
            self._service.validate(config)
        except InvalidConfigError as exc:
            self._mark_invalid(exc.field)
            QMessageBox.warning(self, "Cấu hình không hợp lệ", str(exc))
            return
        except TrafficMonitorError as exc:
            QMessageBox.warning(self, "Cấu hình không hợp lệ", str(exc))
            return
        self._result = config
        super().accept()

    # -------------------------------------------------------------------- tabs
    def _model_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.model_path = QLineEdit()
        browse = QPushButton("Chọn…")
        browse.clicked.connect(self._browse_model)
        row = QHBoxLayout()
        row.addWidget(self.model_path, 1)
        row.addWidget(browse)
        form.addRow("YOLO model (.pt)", row)
        self.device = QComboBox()
        for device_id, label in available_devices():
            self.device.addItem(label, device_id)
        form.addRow("Thiết bị suy luận", self.device)
        self.half = QCheckBox("Dùng FP16 trên GPU (nhanh hơn)")
        form.addRow("", self.half)
        self.agnostic_nms = QCheckBox("Gộp box trùng khác lớp (agnostic NMS)")
        self.agnostic_nms.setToolTip("Một xe được nhận vừa là car vừa là truck sẽ chỉ giữ 1 box, tránh đếm trùng.")
        form.addRow("", self.agnostic_nms)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(160, 1920)
        self.imgsz.setSingleStep(32)
        form.addRow("Kích thước ảnh suy luận", self.imgsz)
        self.max_width = QSpinBox()
        self.max_width.setRange(640, 7680)
        self.max_width.setSingleStep(160)
        self.max_width.setSuffix(" px")
        form.addRow("Độ rộng xử lý tối đa", self.max_width)
        self._fields.update({"model.path": self.model_path, "model.device": self.device, "model.imgsz": self.imgsz,
                             "processing.max_width": self.max_width})
        return tab

    def _detection_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.confidence = _spin(0.05, 0.95, 0.3, 0.05, 2)
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(5, 95)
        self.confidence_slider.valueChanged.connect(lambda v: self.confidence.setValue(v / 100))
        self.confidence.valueChanged.connect(lambda v: self.confidence_slider.setValue(round(v * 100)))
        row = QHBoxLayout()
        row.addWidget(self.confidence_slider, 1)
        row.addWidget(self.confidence)
        form.addRow("Confidence threshold", row)
        self.iou = _spin(0.05, 1.0, 0.7, 0.05, 2)
        form.addRow("IoU (NMS)", self.iou)
        classes = QHBoxLayout()
        self.class_boxes: dict[int, QCheckBox] = {}
        for class_id, name in OBJECT_CLASSES.items():
            box = QCheckBox(f"{name} ({class_id})")
            self.class_boxes[class_id] = box
            classes.addWidget(box)
        form.addRow("Object classes", classes)
        self.speed_limit = _spin(10, 200, 80, 5, 0)
        self.speed_limit.setSuffix(" km/h")
        form.addRow("Speed limit", self.speed_limit)
        self.window_seconds = _spin(0.2, 5.0, 1.0, 0.1, 1)
        self.window_seconds.setSuffix(" s")
        form.addRow("Cửa sổ tính tốc độ", self.window_seconds)
        self._fields.update({"model.confidence": self.confidence, "model.iou": self.iou,
                             "speed.limit_kmh": self.speed_limit, "speed.window_seconds": self.window_seconds})
        return tab

    def _calibration_tab(self, video_name: str | None) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QVBoxLayout()
        points_box = QGroupBox("Perspective source points (pixel gốc của video)")
        grid = QGridLayout(points_box)
        self.points: list[tuple[QDoubleSpinBox, QDoubleSpinBox]] = []
        for row, name in enumerate(("A (trên-trái)", "B (trên-phải)", "C (dưới-phải)", "D (dưới-trái)")):
            x, y = _spin(-_COORD_RANGE, _COORD_RANGE, 0), _spin(-_COORD_RANGE, _COORD_RANGE, 0)
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(QLabel("x"), row, 1)
            grid.addWidget(x, row, 2)
            grid.addWidget(QLabel("y"), row, 3)
            grid.addWidget(y, row, 4)
            self.points.append((x, y))
        left.addWidget(points_box)

        road = QGroupBox("Kích thước thực của vùng ABCD")
        road_form = QFormLayout(road)
        self.road_width = _spin(0.5, 500, 25, 0.5, 1)
        self.road_width.setSuffix(" m")
        self.road_length = _spin(1, 2000, 250, 1, 1)
        self.road_length.setSuffix(" m")
        road_form.addRow("Road width", self.road_width)
        road_form.addRow("Road length", self.road_length)
        left.addWidget(road)

        line_box = QGroupBox("Vạch đếm (đảo Start/End để đổi chiều IN/OUT)")
        line_grid = QGridLayout(line_box)
        self.line = [_spin(-_COORD_RANGE, _COORD_RANGE, 0) for _ in range(4)]
        for index, caption in enumerate(("Start x", "Start y", "End x", "End y")):
            line_grid.addWidget(QLabel(caption), index // 2, (index % 2) * 2)
            line_grid.addWidget(self.line[index], index // 2, (index % 2) * 2 + 1)
        left.addWidget(line_box)

        self.save_calibration = QCheckBox(
            f"Lưu hiệu chỉnh riêng cho video {video_name}" if video_name else "Chọn video để lưu hiệu chỉnh riêng"
        )
        self.save_calibration.setEnabled(video_name is not None)
        self.save_calibration.setChecked(video_name is not None)
        left.addWidget(self.save_calibration)
        left.addStretch(1)
        layout.addLayout(left, 1)

        self.preview = QLabel("Chọn video để xem trước vạch đếm và vùng ABCD")
        self.preview.setObjectName("videoView")
        self.preview.setMinimumSize(420, 240)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview, 1)
        for spin in [*self.line, self.road_width, self.road_length, *[c for p in self.points for c in p]]:
            spin.valueChanged.connect(self._update_preview)
        self._fields.update({"perspective.source_points": points_box, "perspective.road_width_m": self.road_width,
                             "perspective.road_length_m": self.road_length, "counting_line": line_box})
        return tab

    def _output_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.video_path = QLineEdit()
        self.report_path = QLineEdit()
        for label, edit, caption, pattern in (
            ("Output video path", self.video_path, "Lưu video kết quả", "Video (*.mp4 *.avi)"),
            ("Report path", self.report_path, "Lưu báo cáo", "JSON (*.json)"),
        ):
            browse = QPushButton("Chọn…")
            browse.clicked.connect(lambda _c=False, e=edit, t=caption, p=pattern: self._browse_save(e, t, p))
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(browse)
            form.addRow(label, row)
        self._fields.update({"output.video_path": self.video_path, "output.report_path": self.report_path})
        return tab

    # ----------------------------------------------------------------- helpers
    def _load(self, config: AppConfig) -> None:
        self.model_path.setText(config.model.path)
        index = self.device.findData(config.model.device)
        self.device.setCurrentIndex(index if index >= 0 else 0)
        self.half.setChecked(config.model.half)
        self.agnostic_nms.setChecked(config.model.agnostic_nms)
        self.imgsz.setValue(config.model.imgsz)
        self.max_width.setValue(config.processing.max_width)
        self.confidence.setValue(config.model.confidence)
        self.confidence_slider.setValue(round(config.model.confidence * 100))
        self.iou.setValue(config.model.iou)
        for class_id, box in self.class_boxes.items():
            box.setChecked(class_id in config.classes)
        self.speed_limit.setValue(config.speed.limit_kmh)
        self.window_seconds.setValue(config.speed.window_seconds)
        for (x, y), (px, py) in zip(self.points, config.perspective.source_points):
            x.setValue(px)
            y.setValue(py)
        self.road_width.setValue(config.perspective.road_width_m)
        self.road_length.setValue(config.perspective.road_length_m)
        for spin, value in zip(self.line, [*config.counting_line.start, *config.counting_line.end]):
            spin.setValue(value)
        self.video_path.setText(config.output.video_path)
        self.report_path.setText(config.output.report_path)
        self._update_preview()

    def _restore_defaults(self) -> None:
        try:
            self._load(self._service.load_defaults())
        except TrafficMonitorError as exc:
            QMessageBox.warning(self, "Không đọc được cấu hình mặc định", str(exc))

    def _update_preview(self) -> None:
        if self._preview_frame is None:
            return
        config = self.collect()
        try:
            canvas = draw_calibration(
                self._preview_frame, config.counting_line.start, config.counting_line.end, config.perspective.source_points,
                config.regions.zones if config.regions.enabled else (), config.annotation,
            )
        except Exception:  # preview must never break the form
            return
        image = to_qimage(resize_to_width(canvas, 800))
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _browse_model(self) -> None:
        start = resolve_path(self.model_path.text() or "models").parent
        path, _ = QFileDialog.getOpenFileName(self, "Chọn model YOLO", str(start), "YOLO weights (*.pt)")
        if path:
            self.model_path.setText(to_display_path(path))

    def _browse_save(self, edit: QLineEdit, caption: str, pattern: str) -> None:
        start = resolve_path(edit.text()) if edit.text() else PROJECT_ROOT
        path, _ = QFileDialog.getSaveFileName(self, caption, str(start), pattern)
        if path:
            edit.setText(to_display_path(Path(path)))

    def _mark_invalid(self, field: str) -> None:
        for name, widget in self._fields.items():
            if field == name or field.startswith(name):
                widget.setProperty("invalid", True)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                for tab_index in range(self.tabs.count()):
                    if self.tabs.widget(tab_index).isAncestorOf(widget):
                        self.tabs.setCurrentIndex(tab_index)

    def _clear_invalid(self) -> None:
        for widget in self._fields.values():
            widget.setProperty("invalid", False)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
