"""Desktop UI package.

torch and ultralytics are imported before any PySide6 module, and PySide6's
per-import source inspection is replaced by a fast equivalent, otherwise the
lazy imports done while loading YOLO in the worker thread take tens of seconds.
"""

import torch  # noqa: F401
import ultralytics  # noqa: F401

from app.utils.qt_compat import speed_up_pyside_imports

speed_up_pyside_imports()
