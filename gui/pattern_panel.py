"""
gui/pattern_panel.py

PatternPanel — floating panel for linear/circular pattern parameters.
Emits `changed` live (for preview) and `confirmed` / `cancelled`.
"""

from __future__ import annotations
import math
from cad.prefs import prefs
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QDoubleSpinBox)
from PyQt6.QtCore import pyqtSignal, Qt


_STYLE = """
QWidget#PatternPanel {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
}
QLabel { color: #d4d4d4; font-size: 12px; }
QLabel#title { color: #ffffff; font-size: 13px; font-weight: bold; }
QSpinBox, QDoubleSpinBox {
    background: #2a2a2a; color: #d4d4d4;
    border: 1px solid #444; border-radius: 3px; padding: 3px 6px;
}
QPushButton {
    background: #2a2a2a; color: #d4d4d4; border: 1px solid #444;
    border-radius: 3px; padding: 4px 10px; font-size: 12px;
}
QPushButton:hover  { background: #333; }
QPushButton#ok {
    background: #1a4a7a; border-color: #4a90d9; color: #fff;
    padding: 5px 18px; font-weight: bold;
}
QPushButton#ok:hover { background: #1e5a8a; }
"""


class PatternPanel(QWidget):
    changed    = pyqtSignal()        # any parameter changed (live preview)
    confirmed  = pyqtSignal()
    cancelled  = pyqtSignal()

    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        self.mode = mode             # "linear" | "circular"
        self.setObjectName("PatternPanel")
        self.setStyleSheet(prefs.scale_stylesheet(_STYLE))
        self.setMinimumWidth(prefs.scaled_px(230))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(9)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Linear Pattern" if self.mode == "linear"
                       else "Circular Pattern")
        title.setObjectName("title")
        root.addWidget(title)

        # Count (shared)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Count"))
        self._count = QSpinBox()
        self._count.setRange(2, 200)
        self._count.setValue(3)
        self._count.valueChanged.connect(lambda _: self.changed.emit())
        crow.addWidget(self._count)
        root.addLayout(crow)

        if self.mode == "linear":
            srow = QHBoxLayout()
            srow.addWidget(QLabel("Spacing (mm)"))
            self._spacing = QDoubleSpinBox()
            self._spacing.setRange(0.001, 100000.0)
            self._spacing.setDecimals(3)
            self._spacing.setValue(10.0)
            self._spacing.valueChanged.connect(lambda _: self.changed.emit())
            srow.addWidget(self._spacing)
            root.addLayout(srow)
        else:
            arow = QHBoxLayout()
            arow.addWidget(QLabel("Total angle (°)"))
            self._angle = QDoubleSpinBox()
            self._angle.setRange(-360.0, 360.0)
            self._angle.setDecimals(1)
            self._angle.setValue(360.0)
            self._angle.valueChanged.connect(lambda _: self.changed.emit())
            arow.addWidget(self._angle)
            root.addLayout(arow)

        btns = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancelled.emit)
        ok = QPushButton("OK")
        ok.setObjectName("ok")
        ok.clicked.connect(self.confirmed.emit)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        root.addLayout(btns)

    # -- accessors -------------------------------------------------------
    def count(self) -> int:
        return self._count.value()

    def spacing_mm(self) -> float:
        return self._spacing.value() if self.mode == "linear" else 0.0

    def angle_rad(self) -> float:
        return (math.radians(self._angle.value())
                if self.mode == "circular" else 0.0)

    def set_spacing_mm(self, v: float):
        if self.mode == "linear":
            self._spacing.blockSignals(True)
            self._spacing.setValue(float(v))
            self._spacing.blockSignals(False)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.confirmed.emit()
        elif e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
        else:
            super().keyPressEvent(e)
