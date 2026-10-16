"""Dark Qt stylesheet for the desktop application."""

from app.constants.classes import CLASS_COLORS

BACKGROUND = "#1A1E24"
PANEL = "#22272F"
PANEL_2 = "#2A303A"
LINE = "#363E4A"
INK = "#E7EBF0"
INK_2 = "#9AA5B2"
ACCENT = "#4F8FE0"
GOOD = "#4FD69C"
WARN = "#E6B32E"
BAD = "#FF6B63"
OUT = "#FFA863"

STATE_COLORS: dict[str, str] = {
    "IDLE": "#6E7885",
    "VIDEO_SELECTED": ACCENT,
    "RUNNING": GOOD,
    "PAUSED": WARN,
    "STOPPED": INK_2,
    "COMPLETED": GOOD,
    "ERROR": BAD,
}

NOTIFY_COLORS: dict[str, str] = {"info": INK, "success": GOOD, "warning": WARN, "error": BAD}

CLASS_HEX = CLASS_COLORS

# Same family as the text drawn on video frames (annotation.font_family).
FONT_FAMILY = '"Times New Roman", "Liberation Serif", "DejaVu Serif", serif'

STYLESHEET = f"""
QMainWindow, QDialog {{ background: {BACKGROUND}; }}
QWidget {{ color: {INK}; font-family: {FONT_FAMILY}; font-size: 14px; }}
QLabel#title {{ font-size: 18px; font-weight: 700; letter-spacing: 4px; padding: 8px 12px; background: #12151A; border-bottom: 1px solid {LINE}; }}
QLabel#sectionTitle {{ color: {INK_2}; font-size: 13px; font-weight: 600; letter-spacing: 2px; }}
QFrame#dashboard {{ background: {PANEL}; border-left: 1px solid {LINE}; }}
QFrame#tile {{ background: {PANEL_2}; border-radius: 6px; }}
QLabel#tileCaption {{ color: {INK_2}; font-size: 12px; letter-spacing: 1px; }}
QLabel#tileIn {{ color: {GOOD}; font-size: 30px; font-weight: 700; font-family: {FONT_FAMILY}; }}
QLabel#tileOut {{ color: {OUT}; font-size: 30px; font-weight: 700; font-family: {FONT_FAMILY}; }}
QLabel#metricValue {{ font-family: {FONT_FAMILY}; font-size: 15px; }}
QLabel#metricBad {{ font-family: {FONT_FAMILY}; font-size: 15px; color: {BAD}; font-weight: 700; }}
QLabel#videoView {{ background: #0B0D10; color: #6E7885; font-size: 15px; }}
QFrame#controlBar {{ background: {PANEL}; border-top: 1px solid {LINE}; }}
QPushButton {{ background: {PANEL_2}; border: 1px solid {LINE}; border-radius: 5px; padding: 7px 16px; font-weight: 500; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {LINE}; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
QPushButton[blocked="true"], QPushButton#primary[blocked="true"] {{ color: #5C6570; background: #23282F; border-color: #2C323B; }}
QPushButton[blocked="true"]:hover, QPushButton#primary[blocked="true"]:hover {{ border-color: {WARN}; }}
QTableWidget, QListWidget {{ background: {PANEL}; border: 1px solid {LINE}; gridline-color: {LINE}; font-family: {FONT_FAMILY}; }}
QHeaderView::section {{ background: {PANEL_2}; color: {INK_2}; border: 0; padding: 4px; font-size: 12px; }}
QListWidget::item {{ padding: 3px 4px; color: #FFB2AC; }}
QStatusBar {{ background: #12151A; border-top: 1px solid {LINE}; color: {INK_2}; font-family: {FONT_FAMILY}; }}
QStatusBar QLabel {{ color: {INK_2}; padding: 0 6px; font-family: {FONT_FAMILY}; }}
QLabel#statePill {{ border-radius: 9px; padding: 2px 10px; color: #0B0D10; font-weight: 700; }}
QProgressBar {{ background: {PANEL_2}; border: 0; border-radius: 3px; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: #15181D; border: 1px solid {LINE}; border-radius: 4px; padding: 4px 6px; font-family: {FONT_FAMILY}; }}
QLineEdit[invalid="true"], QSpinBox[invalid="true"], QDoubleSpinBox[invalid="true"], QComboBox[invalid="true"] {{ border-color: {BAD}; }}
QTabWidget::pane {{ border: 1px solid {LINE}; background: {PANEL}; }}
QTabBar::tab {{ background: {PANEL_2}; padding: 7px 14px; border: 1px solid {LINE}; border-bottom: 0; }}
QTabBar::tab:selected {{ background: {PANEL}; color: white; }}
QGroupBox {{ border: 1px solid {LINE}; border-radius: 6px; margin-top: 14px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {INK_2}; }}
QCheckBox::indicator {{ width: 15px; height: 15px; }}
QMessageBox {{ background: {PANEL}; }}
QToolTip {{ background: {PANEL_2}; color: {INK}; border: 1px solid {LINE}; }}
"""
