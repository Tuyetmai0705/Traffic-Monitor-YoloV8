"""Widget that shows processed frames scaled to fit."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

PLACEHOLDER = "NO VIDEO\nBấm Open Video để chọn file trong data/input/"


class VideoWidget(QLabel):
    """Display frames, keep aspect ratio on resize, clear to a placeholder."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty video view."""
        super().__init__(parent)
        self.setObjectName("videoView")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap: QPixmap | None = None
        self.clear_frame()

    @property
    def has_frame(self) -> bool:
        """True when a frame is displayed."""
        return self._pixmap is not None

    def show_frame(self, image: QImage) -> None:
        """Display a frame scaled to the widget."""
        self._pixmap = QPixmap.fromImage(image)
        self._render()

    def clear_frame(self, message: str = PLACEHOLDER) -> None:
        """Remove the frame and show a message."""
        self._pixmap = None
        super().clear()
        self.setText(message)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        """Rescale the current frame when the widget is resized."""
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
