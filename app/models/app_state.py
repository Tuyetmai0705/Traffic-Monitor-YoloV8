"""Application states and the business rules for which action is allowed when."""

from __future__ import annotations

from enum import Enum

from app.constants import messages as msg


class AppState(str, Enum):
    """Lifecycle of the desktop application."""

    IDLE = "IDLE"
    VIDEO_SELECTED = "VIDEO_SELECTED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class Action(str, Enum):
    """User actions exposed by the control panel."""

    OPEN = "open"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    EXPORT = "export"
    SETTINGS = "settings"


_IDLE_LIKE = frozenset({AppState.IDLE, AppState.VIDEO_SELECTED, AppState.STOPPED, AppState.COMPLETED, AppState.ERROR})

ALLOWED_STATES: dict[Action, frozenset[AppState]] = {
    Action.OPEN: _IDLE_LIKE,
    Action.START: frozenset({AppState.VIDEO_SELECTED, AppState.STOPPED, AppState.COMPLETED}),
    Action.PAUSE: frozenset({AppState.RUNNING}),
    Action.RESUME: frozenset({AppState.PAUSED}),
    Action.STOP: frozenset({AppState.RUNNING, AppState.PAUSED}),
    Action.EXPORT: frozenset({AppState.STOPPED, AppState.COMPLETED}),
    Action.SETTINGS: _IDLE_LIKE,
}

# Target state of each successful user action (None keeps the current state).
ACTION_TARGET: dict[Action, AppState | None] = {
    Action.OPEN: AppState.VIDEO_SELECTED,
    Action.START: AppState.RUNNING,
    Action.PAUSE: AppState.PAUSED,
    Action.RESUME: AppState.RUNNING,
    Action.STOP: AppState.STOPPED,
    Action.EXPORT: None,
    Action.SETTINGS: None,
}


def can_perform(state: AppState, action: Action) -> bool:
    """Return True when ``action`` is allowed in ``state``."""
    return state in ALLOWED_STATES[action]


def blocked_message(state: AppState, action: Action) -> str:
    """Return the user message explaining why ``action`` is blocked in ``state``."""
    if action is Action.START:
        return msg.BLOCK_START_RUNNING if state in (AppState.RUNNING, AppState.PAUSED) else msg.BLOCK_START_NO_VIDEO
    return {
        Action.OPEN: msg.BLOCK_OPEN,
        Action.PAUSE: msg.BLOCK_PAUSE,
        Action.RESUME: msg.BLOCK_RESUME,
        Action.STOP: msg.BLOCK_STOP,
        Action.EXPORT: msg.BLOCK_EXPORT,
        Action.SETTINGS: msg.BLOCK_SETTINGS,
    }[action]
