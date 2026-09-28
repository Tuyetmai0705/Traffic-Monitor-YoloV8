"""Control buttons. Emits user intent only; MainWindow decides what is allowed."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QWidget

from app.models.app_state import Action

BUTTON_LABELS: dict[Action, str] = {
    Action.OPEN: "Open Video",
    Action.START: "Start",
    Action.PAUSE: "Pause",
    Action.RESUME: "Resume",
    Action.STOP: "Stop",
    Action.EXPORT: "Export Report",
    Action.SETTINGS: "Settings",
}

SHORTCUTS: dict[Action, str] = {
    Action.OPEN: "Ctrl+O",
    Action.START: "F5",
    Action.PAUSE: "F6",
    Action.RESUME: "F7",
    Action.STOP: "F8",
    Action.EXPORT: "Ctrl+E",
    Action.SETTINGS: "Ctrl+,",
}


class ControlPanel(QFrame):
    """Seven action buttons. Blocked buttons stay clickable so the user gets an explanation."""

    action_triggered = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the buttons in spec order."""
        super().__init__(parent)
        self.setObjectName("controlBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        self.buttons: dict[Action, QPushButton] = {}
        for action, label in BUTTON_LABELS.items():
            if action is Action.EXPORT:
                layout.addStretch(1)
            button = QPushButton(label)
            button.setObjectName("primary" if action is Action.START else f"btn_{action.value}")
            button.setShortcut(SHORTCUTS[action])
            button.clicked.connect(lambda _checked=False, a=action: self.action_triggered.emit(a.value))
            layout.addWidget(button)
            self.buttons[action] = button

    def set_allowed(self, allowed: dict[Action, bool], reasons: dict[Action, str] | None = None) -> None:
        """Style each button as allowed or blocked and set a tooltip."""
        reasons = reasons or {}
        for action, button in self.buttons.items():
            is_allowed = allowed.get(action, False)
            button.setProperty("blocked", not is_allowed)
            tip = f"{BUTTON_LABELS[action]} ({SHORTCUTS[action]})"
            button.setToolTip(tip if is_allowed else f"{tip}\n{reasons.get(action, '')}")
            button.style().unpolish(button)
            button.style().polish(button)

    def is_allowed(self, action: Action) -> bool:
        """True when the button is currently shown as allowed."""
        return not bool(self.buttons[action].property("blocked"))
