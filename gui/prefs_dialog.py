"""
gui/prefs_dialog.py

PrefsDialog — color/settings/keybinding editor for user preferences.

Each preference is shown as a labelled row.  Color values get a clickable
swatch that opens QColorDialog.  Bool values get a checkbox.  Float values
get a spinbox.  Keybindings get a QKeySequenceEdit.  Changes are written to
the prefs singleton immediately and saved to disk on OK.
"""

from __future__ import annotations
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QCheckBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QDialogButtonBox, QFrame, QSizePolicy, QKeySequenceEdit,
    QTabWidget,
)
from PyQt6.QtGui import QColor, QKeySequence, QFont
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from cad.prefs import prefs, KEYBIND_LABELS, KEYBIND_DEFAULTS


def _apply_ui_scale(old_offset: int = 0):
    """Update QApplication font and rescale all widget stylesheets."""
    import re
    app = QApplication.instance()
    if app is None:
        return

    import cad.prefs as _prefs_mod
    base_pt = getattr(_prefs_mod, '_BASE_FONT_PT', 9)
    f = app.font()
    f.setPointSize(max(1, base_pt + prefs.ui_scale_offset))
    app.setFont(f)

    new_offset = prefs.ui_scale_offset

    def _rebase(css: str) -> str:
        """Undo old offset, apply new offset to every font-size: Npx."""
        def _sub(m):
            current = int(m.group(1))
            original = current - old_offset
            return f"font-size: {max(1, original + new_offset)}px"
        return re.sub(r'font-size:\s*(\d+)px', _sub, css)

    all_widgets = list(app.allWidgets()) + list(app.topLevelWidgets())
    seen = set()
    for widget in all_widgets:
        if id(widget) in seen:
            continue
        seen.add(id(widget))
        ss = widget.styleSheet()
        if ss and 'font-size' in ss:
            widget.setStyleSheet(_rebase(ss))

    # Menu bars cache their font — force re-inherit from the app font
    from PyQt6.QtWidgets import QMenuBar, QMenu, QMainWindow
    for widget in app.allWidgets():
        if isinstance(widget, (QMenuBar, QMenu)):
            widget.setFont(app.font())

    # Rebuild toolbars so icon size updates immediately
    for widget in app.topLevelWidgets():
        if isinstance(widget, QMainWindow) and hasattr(widget, 'refresh_toolbars'):
            widget.refresh_toolbars()


class ClearableKeySequenceEdit(QKeySequenceEdit):
    """
    QKeySequenceEdit that treats Backspace as 'clear this binding'
    rather than binding to the Backspace key.  An empty sequence means
    the action is unbound.
    """
    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.clear()
        else:
            super().keyPressEvent(e)


# Human-readable labels for each pref key
_LABELS = {
    'default_unit':            ('Units',     'Default unit'),
    'display_decimals':        ('Units',     'Decimal places'),
    'background_color':        ('Viewport',  'Background color'),
    'body_color':              ('Bodies',    'Body color'),
    'body_color_active':       ('Bodies',    'Active body color'),
    'edge_color':              ('Edges',     'Edge color'),
    'edge_width':              ('Edges',     'Edge width'),
    'show_edges':              ('Edges',     'Show edges'),
    'face_selected_color':     ('Selection', 'Selected face color'),
    'edge_hovered_color':      ('Selection', 'Hovered edge color'),
    'edge_selected_color':     ('Selection', 'Selected edge color'),
    'vertex_hovered_color':    ('Selection', 'Hovered vertex color'),
    'vertex_selected_color':   ('Selection', 'Selected vertex color'),
    'op_preview_color':        ('Operations', '3D operation preview color'),
    'op_preview_opacity':      ('Operations', '3D operation preview opacity (0–1)'),
    'sketch_line_color':       ('Sketch',    'Sketch line color'),
    'sketch_reference_color':  ('Sketch',    'Reference geometry color'),
    'sketch_preview_color':    ('Sketch',    'Preview line color'),
    'sketch_cursor_color':     ('Sketch',    'Cursor color'),
    'sketch_axis_x_color':     ('Sketch',    'X axis color'),
    'sketch_axis_y_color':     ('Sketch',    'Y axis color'),
    'sketch_grid_major_color': ('Sketch',    'Grid major color'),
    'sketch_grid_minor_color': ('Sketch',    'Grid minor color'),
    'sketch_line_width':       ('Sketch',    'Sketch line width'),
    'sketch_reference_width':  ('Sketch',    'Reference line width'),
    'camera_rotate_speed':     ('Camera',    'Rotate speed'),
    'camera_pan_speed':        ('Camera',    'Pan speed'),
    'camera_mode':             ('Camera',    'Rotation mode'),
    'camera_invert_yaw':       ('Camera',    'Invert yaw'),
    'camera_invert_pitch':     ('Camera',    'Invert pitch'),
    'ui_scale_offset':         ('UI',        'Font size offset (steps)'),
}

_STR_OPTIONS = {
    'camera_mode':    ['trackball', 'arcball'],
    'default_unit':   ['mm', 'cm', 'm', 'in', 'ft'],
}


def _color_to_qcolor(c) -> QColor:
    r, g, b = c
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def _qcolor_to_tuple(q: QColor) -> tuple:
    return (q.red() / 255, q.green() / 255, q.blue() / 255)


def _swatch_style(color) -> str:
    r, g, b = color
    hr, hg, hb = int(r * 255), int(g * 255), int(b * 255)
    return (f"background: rgb({hr},{hg},{hb}); "
            f"border: 1px solid #555; border-radius: 3px;")


def _section_header(layout, title: str):
    lbl = QLabel(title.upper())
    lbl.setStyleSheet(prefs.scale_stylesheet(
        "color: #555; font-size: 10px; font-weight: bold; "
        "letter-spacing: 1px; padding-top: 12px; padding-bottom: 4px;"))
    layout.addWidget(lbl)
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #333;")
    layout.addWidget(line)


def _row_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(prefs.scale_stylesheet("color: #ccc; font-size: 12px;"))
    lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                      QSizePolicy.Policy.Preferred)
    return lbl


class PrefsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(460)
        self.setMinimumHeight(540)
        self._widgets: dict[str, QWidget] = {}
        self._keybind_widgets: dict[str, QKeySequenceEdit] = {}
        self._setup()

    def _setup(self):
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab {
                background: #222; color: #888;
                padding: 6px 16px; border: none;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected { color: #d4d4d4; border-bottom: 2px solid #4a90d9; }
        """)

        tabs.addTab(self._build_prefs_tab(),     "Display")
        tabs.addTab(self._build_keybinds_tab(),  "Keybindings")

        root.addWidget(tabs)
        root.addWidget(self._build_button_row())

    # ------------------------------------------------------------------
    # Display tab
    # ------------------------------------------------------------------

    def _build_prefs_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)

        current_section = None
        for key, (section, label) in _LABELS.items():
            if section != current_section:
                current_section = section
                _section_header(layout, section)

            val = getattr(prefs, key)
            row = QHBoxLayout()
            row.setContentsMargins(0, 3, 0, 3)
            row.addWidget(_row_label(label))

            if isinstance(val, tuple):
                btn = QPushButton()
                btn.setFixedSize(48, 22)
                btn.setStyleSheet(_swatch_style(val))
                btn.setProperty('pref_key', key)
                btn.clicked.connect(self._pick_color)
                self._widgets[key] = btn
                row.addWidget(btn)

            elif isinstance(val, bool):
                cb = QCheckBox()
                cb.setChecked(val)
                cb.setProperty('pref_key', key)
                self._widgets[key] = cb
                row.addWidget(cb)

            elif isinstance(val, int):
                sp = QSpinBox()
                if key == 'ui_scale_offset':
                    sp.setMinimum(-4)
                    sp.setMaximum(8)
                    sp.setPrefix("+" if val >= 0 else "")
                    sp.valueChanged.connect(
                        lambda v, s=sp: s.setPrefix("+" if v >= 0 else ""))
                else:
                    sp.setMinimum(0)
                    sp.setMaximum(6)
                sp.setValue(val)
                sp.setFixedWidth(60)
                sp.setProperty('pref_key', key)
                self._widgets[key] = sp
                row.addWidget(sp)

            elif isinstance(val, float):
                sp = QDoubleSpinBox()
                sp.setDecimals(2)
                if key.endswith('_opacity'):
                    sp.setMinimum(0.0)
                    sp.setMaximum(1.0)
                    sp.setSingleStep(0.05)
                else:
                    sp.setMinimum(0.1)
                    sp.setMaximum(20.0)
                    sp.setSingleStep(0.1)
                sp.setValue(val)
                sp.setFixedWidth(80)
                sp.setProperty('pref_key', key)
                self._widgets[key] = sp
                row.addWidget(sp)

            elif isinstance(val, str):
                cb = QComboBox()
                for option in _STR_OPTIONS.get(key, [val]):
                    cb.addItem(option)
                cb.setCurrentText(val)
                cb.setProperty('pref_key', key)
                self._widgets[key] = cb
                row.addWidget(cb)

            row_w = QWidget()
            row_w.setLayout(row)
            layout.addWidget(row_w)

        layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    # ------------------------------------------------------------------
    # Keybindings tab
    # ------------------------------------------------------------------

    def _build_keybinds_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)

        current_section = None
        for action, (section, label) in KEYBIND_LABELS.items():
            if section != current_section:
                current_section = section
                _section_header(layout, section)

            row = QHBoxLayout()
            row.setContentsMargins(0, 3, 0, 3)
            row.addWidget(_row_label(label))

            kse = ClearableKeySequenceEdit()
            kse.setKeySequence(QKeySequence(prefs.key(action)))
            kse.setFixedWidth(130)
            kse.setStyleSheet("""
                QKeySequenceEdit {
                    background: #2a2a2a; color: #d4d4d4;
                    border: 1px solid #444; border-radius: 3px;
                    padding: 2px 6px; font-size: 12px;
                }
            """)
            self._keybind_widgets[action] = kse
            row.addWidget(kse)

            # Per-row reset to default
            rst = QPushButton("↺")
            rst.setFixedSize(24, 24)
            rst.setToolTip(f"Reset to default ({KEYBIND_DEFAULTS.get(action, '')})")
            rst.setStyleSheet("""
                QPushButton {
                    background: #2a2a2a; color: #666;
                    border: 1px solid #444; border-radius: 3px;
                    font-size: 13px;
                }
                QPushButton:hover { color: #d4d4d4; }
            """)
            rst.setProperty('action', action)
            rst.clicked.connect(self._reset_keybind)
            row.addWidget(rst)

            row_w = QWidget()
            row_w.setLayout(row)
            layout.addWidget(row_w)

        layout.addStretch()

        # Reset all keybinds button
        reset_all = QPushButton("Reset all keybindings to defaults")
        reset_all.setStyleSheet(
            "QPushButton { background: #222; color: #666; border: 1px solid #444; "
            "padding: 5px 10px; border-radius: 3px; }"
            "QPushButton:hover { color: #d4d4d4; }")
        reset_all.clicked.connect(self._reset_all_keybinds)
        layout.addWidget(reset_all)

        scroll.setWidget(inner)
        return scroll

    # ------------------------------------------------------------------
    # Button row (shared OK / Cancel / Reset display)
    # ------------------------------------------------------------------

    def _build_button_row(self) -> QWidget:
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(12, 8, 12, 8)

        reset_btn = QPushButton("Reset display to defaults")
        reset_btn.clicked.connect(self._reset_display)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)

        w = QWidget()
        w.setLayout(btn_layout)
        return w

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _pick_color(self):
        btn = self.sender()
        key = btn.property('pref_key')
        current = getattr(prefs, key)
        from PyQt6.QtWidgets import QColorDialog
        new_qc = QColorDialog.getColor(
            _color_to_qcolor(current), self, f"Choose color — {key}")
        if new_qc.isValid():
            new_color = _qcolor_to_tuple(new_qc)
            btn.setStyleSheet(_swatch_style(new_color))
            btn.setProperty('current_color', new_color)

    def _reset_keybind(self):
        action = self.sender().property('action')
        default = KEYBIND_DEFAULTS.get(action, "")
        self._keybind_widgets[action].setKeySequence(QKeySequence(default))

    def _reset_all_keybinds(self):
        for action, kse in self._keybind_widgets.items():
            default = KEYBIND_DEFAULTS.get(action, "")
            kse.setKeySequence(QKeySequence(default))

    def _reset_display(self):
        prefs.reset()
        self.close()
        PrefsDialog(self.parent()).exec()

    def _apply_and_accept(self):
        old_scale = prefs.ui_scale_offset

        # Display prefs
        for key, widget in self._widgets.items():
            if isinstance(widget, QPushButton):
                c = widget.property('current_color')
                if c is not None:
                    setattr(prefs, key, tuple(c))
            elif isinstance(widget, QCheckBox):
                setattr(prefs, key, widget.isChecked())
            elif isinstance(widget, QSpinBox):
                setattr(prefs, key, widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                setattr(prefs, key, widget.value())
            elif isinstance(widget, QComboBox):
                setattr(prefs, key, widget.currentText())

        # Keybinds — save "" for intentionally unbound actions
        for action, kse in self._keybind_widgets.items():
            seq = kse.keySequence()
            prefs.keybinds[action] = seq.toString() if not seq.isEmpty() else ""

        prefs.save()

        if prefs.ui_scale_offset != old_scale:
            _apply_ui_scale(old_offset=old_scale)
            self.close()
            PrefsDialog(self.parent()).exec()
            return

        self.accept()
