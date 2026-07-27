import os
import sys

# In a frozen build (PyInstaller AppImage/exe) there is no console, so any
# startup crash or our diagnostic prints (e.g. the GPU-pick CPU fallback) vanish
# — a friend just sees "it didn't open" with nothing to send back. Tee stdout/
# stderr to a log file next to the user's config so failures are recoverable.
# Only in the frozen build; a dev run keeps its normal terminal output.
if getattr(sys, "frozen", False):
    try:
        if os.name == "nt":
            _log_base = os.environ.get("APPDATA", os.path.expanduser("~"))
        else:
            _log_base = os.environ.get(
                "XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config"))
        _log_dir = os.path.join(_log_base, "cadapp")
        os.makedirs(_log_dir, exist_ok=True)
        _log_file = open(os.path.join(_log_dir, "slopcad.log"), "w",
                         buffering=1, encoding="utf-8")
        sys.stdout = _log_file
        sys.stderr = _log_file
        import faulthandler
        faulthandler.enable(_log_file)   # dump native tracebacks on hard crash
    except Exception:
        pass   # never let logging setup stop the app from launching

# Linux/Mesa only: on X11 WMs with no compositor (e.g. awesomewm), Qt6's GLX
# path swaps buffers via Mesa DRI3 (loader_dri3_swap_buffers_msc ->
# xcb_wait_for_special_event), which can block the GUI thread forever during the
# expose/flush after a window resize — the app freezes on its last frame. The
# vblank_mode env var disables that DRI3 vblank wait. It is a MESA-ONLY variable:
# the NVIDIA proprietary driver and Windows (WGL) ignore it, so it's a harmless
# no-op there. Keep the default (GLX) GL integration so PyOpenGL's GLX-based
# context detection still works (switching Qt to EGL breaks glVertexPointer's
# getContext()). Must be set before Qt is imported; setdefault lets any env
# override it. Only touch the environment on Linux so Windows stays pristine.
if sys.platform.startswith("linux"):
    os.environ.setdefault("vblank_mode", "0")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
from gui.mainwindow import MainWindow


def apply_dark_palette(app: QApplication):
    """
    Force a dark palette onto every Qt widget in the app.
    This covers menus, dialogs, input boxes, splitters — everything —
    without needing per-widget stylesheets.
    """
    palette = QPalette()

    # Base colours
    dark        = QColor(30,  30,  30)   # window / base background
    mid_dark    = QColor(42,  42,  42)   # alternate rows, panels
    mid         = QColor(55,  55,  55)   # buttons, inactive
    light       = QColor(68,  68,  68)   # borders, highlights
    text        = QColor(212, 212, 212)  # primary text
    text_dim    = QColor(130, 130, 130)  # disabled / placeholder
    highlight   = QColor(42,  100, 168)  # selection blue
    highlight_t = QColor(212, 212, 212)  # text on selection

    palette.setColor(QPalette.ColorRole.Window,          dark)
    palette.setColor(QPalette.ColorRole.WindowText,      text)
    palette.setColor(QPalette.ColorRole.Base,            mid_dark)
    palette.setColor(QPalette.ColorRole.AlternateBase,   dark)
    palette.setColor(QPalette.ColorRole.ToolTipBase,     mid_dark)
    palette.setColor(QPalette.ColorRole.ToolTipText,     text)
    palette.setColor(QPalette.ColorRole.Text,            text)
    palette.setColor(QPalette.ColorRole.Button,          mid)
    palette.setColor(QPalette.ColorRole.ButtonText,      text)
    palette.setColor(QPalette.ColorRole.BrightText,      QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Link,            QColor(86, 156, 214))
    palette.setColor(QPalette.ColorRole.Highlight,       highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, highlight_t)

    # Disabled state — visibly dimmer but still legible
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.WindowText, text_dim)
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.Text,       text_dim)
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.ButtonText, text_dim)
    palette.setColor(QPalette.ColorGroup.Disabled,
                     QPalette.ColorRole.Highlight,  QColor(60, 60, 60))

    app.setPalette(palette)

    # Minimal stylesheet — just enough to fix a few things QPalette can't reach
    # (QMenu borders, QInputDialog backgrounds, scrollbar width)
    app.setStyleSheet("""
        QMenu {
            background-color: #2a2a2a;
            border: 1px solid #444;
        }
        QMenu::item:selected {
            background-color: #2a64a8;
        }
        QMenu::separator {
            height: 1px;
            background: #444;
            margin: 2px 8px;
        }
        QScrollBar:vertical {
            background: #2a2a2a;
            width: 10px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #555;
            min-height: 20px;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical:hover {
            background: #777;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical { background: none; }
        QToolTip {
            background-color: #2a2a2a;
            color: #d4d4d4;
            border: 1px solid #555;
        }
    """)


def main():
    from PyQt6.QtGui import QSurfaceFormat
    from cad.prefs import prefs as _prefs
    _prefs.load()
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setRedBufferSize(8)
    fmt.setGreenBufferSize(8)
    fmt.setBlueBufferSize(8)
    # Multisample antialiasing — the single biggest win for crisp silhouettes
    # and sketch lines. Sample count is a pref (0 disables). Set at context
    # creation, so a change takes effect on the next launch.
    if _prefs.msaa_samples and _prefs.msaa_samples > 0:
        fmt.setSamples(int(_prefs.msaa_samples))
    QSurfaceFormat.setDefaultFormat(fmt)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")   # Fusion renders cleanly with a custom palette
    apply_dark_palette(app)

    from cad.prefs import prefs
    import cad.prefs as _prefs_mod
    _prefs_mod._BASE_FONT_PT = app.font().pointSize()

    if prefs.ui_scale_offset != 0:
        from PyQt6.QtGui import QFont
        f = app.font()
        f.setPointSize(max(1, _prefs_mod._BASE_FONT_PT + prefs.ui_scale_offset))
        app.setFont(f)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
