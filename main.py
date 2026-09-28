"""Traffic Monitor desktop application entry point: ``python main.py``."""

from __future__ import annotations

import sys
import warnings
from types import TracebackType

# app.ui must be imported before PySide6: it preloads torch/ultralytics and
# patches PySide6's slow per-import hook (see app/utils/qt_compat.py).
from app.ui.main_window import MainWindow
from PySide6.QtWidgets import QApplication, QMessageBox

from app.constants import messages as msg
from app.core.exceptions import TrafficMonitorError
from app.services.config_service import ConfigService
from app.ui.styles import STYLESHEET
from app.utils.file_utils import OUTPUT_DIRS, PROJECT_ROOT, ensure_dir, resolve_path
from app.utils.logger import get_logger, setup_logger

DEFAULT_LOG = PROJECT_ROOT / "outputs" / "logs" / "app.log"

warnings.filterwarnings("ignore", category=FutureWarning)


def install_excepthook() -> None:
    """Log unexpected exceptions and show a friendly dialog instead of a traceback."""
    logger = get_logger("main")

    def handle(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.error("Error: unhandled exception", exc_info=(exc_type, exc, tb))
        text = str(exc) if isinstance(exc, TrafficMonitorError) else msg.UNEXPECTED_ERROR
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "Lỗi", text)

    sys.excepthook = handle


def main() -> int:
    """Configure logging and config, then run the Qt event loop."""
    setup_logger(DEFAULT_LOG)
    logger = get_logger("main")
    app = QApplication(sys.argv)
    app.setApplicationName("Traffic Monitor")
    app.setStyleSheet(STYLESHEET)
    install_excepthook()

    config_service = ConfigService()
    try:
        config = config_service.load()
    except TrafficMonitorError as exc:
        logger.error("Error: %s", exc)
        QMessageBox.critical(None, "Không đọc được cấu hình", str(exc))
        return 1
    if resolve_path(config.output.log_path) != DEFAULT_LOG:
        setup_logger(resolve_path(config.output.log_path))
    for directory in OUTPUT_DIRS:
        ensure_dir(directory)
    logger.info("Application started (config=%s)", config_service.user_path.name if config_service.user_path.is_file() else "default_config.json")

    window = MainWindow(config, config_service)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
