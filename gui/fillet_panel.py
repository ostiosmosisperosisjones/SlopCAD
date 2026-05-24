"""
gui/fillet_panel.py

FilletPanel — small floating panel for the 2D sketch fillet tool.
"""

from __future__ import annotations
from cad.prefs import prefs
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt

_STYLE = """
QWidget#FilletPanel {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
}
QLabel { color: #d4d4d4; font-size: 12px; }
QLabel#title { color: #ffffff; font-size: 13px; font-weight: bold; }
QPushButton {
    background: #2a2a2a; color: #d4d4d4;
    border: 1px solid #444; border-radius: 3px;
    padding: 4px 10px; font-size: 12px;
}
QPushButton:hover  { background: #333; }
QPushButton:pressed { background: #222; }
QPushButton#ok {
    background: #1a4a7a; border-color: #4a90d9;
    color: #fff; padding: 5px 18px; font-weight: bold;
}
QPushButton#ok:hover { background: #1e5a8a; }
"""


class FilletPanel(QWidget):
    confirmed = pyqtSignal(float)
    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FilletPanel")
        self.setStyleSheet(prefs.scale_stylesheet(_STYLE))
        self.setMinimumWidth(prefs.scaled_px(220))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._corner = None
        self._tool   = None
        self._build_ui()

    def set_corner(self, corner: tuple, tool=None):
        self._corner = corner
        self._tool   = tool
        self._validate()

    def _build_ui(self):
        from gui.expr_spinbox import ExprSpinBox
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Fillet")
        title.setObjectName("title")
        root.addWidget(title)

        row = QHBoxLayout()
        row.addWidget(QLabel("Radius"))
        self._spinbox = ExprSpinBox()
        self._spinbox.set_mm(1.0)
        self._spinbox.value_changed.connect(self._on_radius_changed)
        row.addWidget(self._spinbox)
        root.addLayout(row)

        self._hint = QLabel("")
        self._hint.setStyleSheet(prefs.scale_stylesheet("color: #c05050; font-size: 10px;"))
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        self._flip_btn = QPushButton("⇄ Flip")
        self._flip_btn.setToolTip("Cycle to the other fillet direction")
        self._flip_btn.clicked.connect(self._on_flip)
        self._flip_btn.setEnabled(False)
        root.addWidget(self._flip_btn)

        btns = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancelled.emit)
        self._ok_btn = QPushButton("OK")
        self._ok_btn.setObjectName("ok")
        self._ok_btn.clicked.connect(self._on_ok)
        self._ok_btn.setEnabled(False)
        btns.addWidget(cancel)
        btns.addWidget(self._ok_btn)
        root.addLayout(btns)

    def _on_radius_changed(self, _):
        self._validate()
        vp = self.parent()
        if vp is not None:
            vp.update()

    def _validate(self):
        from cad.sketch_tools.fillet import compute_fillet
        v = self._spinbox.mm_value()
        if v is None or v <= 0 or self._corner is None:
            self._set_invalid("Enter a positive radius.")
            return
        pt, ent_a, ent_b, end_a, end_b = self._corner
        result = compute_fillet(pt, ent_a, end_a, ent_b, end_b, v)
        if result is None:
            self._set_invalid("Radius too large for this corner.")
        else:
            self._set_valid(v)

    def _on_flip(self):
        if self._tool is not None:
            self._tool.flip()
        vp = self.parent()
        if vp is not None:
            vp.update()

    def _set_valid(self, radius):
        self._spinbox.setStyleSheet(self._spinbox._STYLE_NORMAL)
        self._hint.setText("")
        self._ok_btn.setEnabled(True)
        if self._tool is not None:
            self._tool.preview_radius = radius
            self._tool._refresh_results(radius)
            self._flip_btn.setEnabled(len(self._tool._all_results) > 1)

    def _set_invalid(self, msg: str):
        self._spinbox.setStyleSheet(self._spinbox._STYLE_ERROR)
        self._hint.setText(msg)
        self._ok_btn.setEnabled(False)
        self._flip_btn.setEnabled(False)
        if self._tool is not None:
            self._tool.preview_radius = None

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_ok()
        elif e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
        else:
            super().keyPressEvent(e)

    def _on_ok(self):
        v = self._spinbox.mm_value()
        if v is not None and v > 0 and self._ok_btn.isEnabled():
            self.confirmed.emit(v)
