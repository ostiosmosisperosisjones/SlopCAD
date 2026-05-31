"""
gui/offset_panel.py

OffsetPanel — small floating panel shown when the offset tool has a selection.
"""

from __future__ import annotations
from cad.prefs import prefs
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt


_STYLE = """
QWidget#OffsetPanel {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
}
QLabel {
    color: #d4d4d4;
    font-size: 12px;
}
QLabel#title {
    color: #ffffff;
    font-size: 13px;
    font-weight: bold;
}
QPushButton {
    background: #2a2a2a;
    color: #d4d4d4;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 4px 10px;
    font-size: 12px;
}
QPushButton:hover  { background: #333; }
QPushButton:pressed { background: #222; }
QPushButton#ok {
    background: #1a4a7a;
    border-color: #4a90d9;
    color: #fff;
    padding: 5px 18px;
    font-weight: bold;
}
QPushButton#ok:hover { background: #1e5a8a; }
"""


class OffsetPanel(QWidget):
    confirmed  = pyqtSignal(float)   # distance_mm
    cancelled  = pyqtSignal()
    changed    = pyqtSignal()        # distance edited (live preview)
    flipped    = pyqtSignal()        # flip-direction toggled

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("OffsetPanel")
        self.setStyleSheet(prefs.scale_stylesheet(_STYLE))
        self.setMinimumWidth(prefs.scaled_px(220))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._build_ui()

    def _build_ui(self):
        from gui.expr_spinbox import ExprSpinBox
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Offset")
        title.setObjectName("title")
        root.addWidget(title)

        row = QHBoxLayout()
        row.addWidget(QLabel("Distance"))
        self._spinbox = ExprSpinBox()
        self._spinbox.set_mm(1.0)
        row.addWidget(self._spinbox)
        root.addLayout(row)

        flip = QPushButton("⇄  Flip direction")
        flip.clicked.connect(self.flipped.emit)
        root.addWidget(flip)

        btns = QHBoxLayout()
        ok = QPushButton("OK")
        ok.setObjectName("ok")
        ok.clicked.connect(self._on_ok)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancelled.emit)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        root.addLayout(btns)

        self._spinbox.value_changed.connect(lambda _: self.changed.emit())

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_ok()
        elif e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
        else:
            super().keyPressEvent(e)

    def _on_ok(self):
        v = self._spinbox.mm_value()
        if v is not None and v > 0:
            self.confirmed.emit(v)
