"""Workaround for slow imports after PySide6 is loaded.

PySide6 hooks every import to decide whether the module uses PySide
(``shibokensupport.feature._mod_uses_pyside``). The stock check calls
``inspect.getsource`` which scans all of ``sys.modules`` per import, so
importing torch/ultralytics lazily (e.g. inside the worker thread) becomes
quadratic and takes tens of seconds. This replaces the check with an
equivalent file search that runs in microseconds.
"""

from __future__ import annotations

import sys
from types import ModuleType

from app.utils.logger import get_logger

logger = get_logger("qt_compat")


def _mod_uses_pyside(module: ModuleType) -> bool:
    """Same contract as shiboken's check: does the module source mention PySide6?"""
    path = getattr(module, "__file__", None)
    if not path or not str(path).endswith(".py"):
        return False
    try:
        with open(path, "rb") as handle:
            return b"PySide6" in handle.read()
    except OSError:
        return False


def speed_up_pyside_imports() -> bool:
    """Install the fast check. Returns True when the patch was applied."""
    import PySide6.QtCore  # noqa: F401  (loads shibokensupport)

    feature = sys.modules.get("shibokensupport.feature")
    if feature is None or not hasattr(feature, "_mod_uses_pyside"):
        logger.debug("shibokensupport.feature not found; no patch needed")
        return False
    feature._mod_uses_pyside = _mod_uses_pyside
    return True
