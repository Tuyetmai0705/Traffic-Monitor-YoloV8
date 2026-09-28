"""Business rules for application states (spec §19)."""

from __future__ import annotations

import pytest

from app.constants import messages as msg
from app.models.app_state import Action, AppState, blocked_message, can_perform


@pytest.mark.parametrize(
    ("state", "action", "allowed"),
    [
        (AppState.IDLE, Action.START, False),        # Start khi chưa chọn video
        (AppState.VIDEO_SELECTED, Action.START, True),
        (AppState.VIDEO_SELECTED, Action.PAUSE, False),  # Pause khi chưa chạy
        (AppState.RUNNING, Action.PAUSE, True),
        (AppState.RUNNING, Action.RESUME, False),    # Resume khi chưa pause
        (AppState.PAUSED, Action.RESUME, True),
        (AppState.RUNNING, Action.EXPORT, False),    # Export khi chưa có result
        (AppState.VIDEO_SELECTED, Action.EXPORT, False),
        (AppState.STOPPED, Action.EXPORT, True),
        (AppState.COMPLETED, Action.EXPORT, True),
        (AppState.PAUSED, Action.STOP, True),
        (AppState.IDLE, Action.STOP, False),
        (AppState.RUNNING, Action.SETTINGS, False),
        (AppState.RUNNING, Action.OPEN, False),
        (AppState.COMPLETED, Action.START, True),
        (AppState.ERROR, Action.OPEN, True),
        (AppState.ERROR, Action.START, False),
    ],
)
def test_allowed_actions(state: AppState, action: Action, allowed: bool) -> None:
    assert can_perform(state, action) is allowed


def test_blocked_messages() -> None:
    assert blocked_message(AppState.IDLE, Action.START) == msg.BLOCK_START_NO_VIDEO
    assert blocked_message(AppState.RUNNING, Action.START) == msg.BLOCK_START_RUNNING
    assert blocked_message(AppState.VIDEO_SELECTED, Action.PAUSE) == msg.BLOCK_PAUSE
    assert blocked_message(AppState.RUNNING, Action.RESUME) == msg.BLOCK_RESUME
    assert blocked_message(AppState.RUNNING, Action.EXPORT) == msg.BLOCK_EXPORT


def test_every_action_has_a_message_for_every_blocked_state() -> None:
    for action in Action:
        for state in AppState:
            if not can_perform(state, action):
                assert blocked_message(state, action)
