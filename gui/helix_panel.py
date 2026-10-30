"""
gui/helix_panel.py

HelixPanel — a helix curve around a cylindrical face or a sketch circle:
height (or the face's full length), turns or pitch, hand, flip.
"""

from __future__ import annotations
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QFrame, QDoubleSpinBox,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer

from gui.loft_panel import _PANEL_STYLE, _SEP_STYLE
from gui.expr_spinbox import ExprSpinBox


class HelixPanel(QWidget):
    helix_requested = pyqtSignal(dict)      # HelixOp fields, minus the source
    cancelled       = pyqtSignal()
    picking_changed = pyqtSignal(bool)
    preview_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent.window() if parent is not None else None)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Helix")
        self.setObjectName("LoftPanel")             # loft panel styling
        self.setStyleSheet(prefs.scale_stylesheet(_PANEL_STYLE))
        self.setMinimumWidth(prefs.scaled_px(280))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._source: str | None = None             # "face" | "circle"
        self._picking = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.preview_changed)
        self._build_ui()
        self._on_spec_changed()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)
        title = QLabel("Helix")
        title.setObjectName("title")
        root.addWidget(title)
        root.addWidget(self._separator())

        row = QHBoxLayout()
        sec = QLabel("AROUND")
        sec.setObjectName("section")
        row.addWidget(sec)
        row.addStretch()
        self._pick_btn = QPushButton("Pick")
        self._pick_btn.setObjectName("pick_profile")
        self._pick_btn.setCheckable(True)
        self._pick_btn.clicked.connect(self.set_picking)
        row.addWidget(self._pick_btn)
        root.addLayout(row)
        self._source_label = QLabel("a cylindrical face or a sketch circle")
        self._source_label.setWordWrap(True)
        self._source_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        root.addWidget(self._source_label)
        root.addWidget(self._separator())

        g = QGridLayout()
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        self._full_chk = QCheckBox("Full face length")
        self._full_chk.setChecked(True)
        self._full_chk.toggled.connect(self._on_full_toggled)
        g.addWidget(self._full_chk, 0, 0, 1, 2)
        self._height = ExprSpinBox(unit=prefs.default_unit)
        self._height.set_mm(20.0)
        self._height.value_changed.connect(lambda _v: self._changed())
        g.addWidget(QLabel("Height"), 1, 0)
        g.addWidget(self._height, 1, 1)
        self._spec = QComboBox()
        self._spec.addItem("Turns", "turns")
        self._spec.addItem("Pitch", "pitch")
        self._spec.currentIndexChanged.connect(self._on_spec_changed)
        g.addWidget(self._spec, 2, 0)
        self._turns = QDoubleSpinBox()
        self._turns.setRange(0.05, 10000.0)
        self._turns.setDecimals(2)
        self._turns.setValue(5.0)
        self._turns.valueChanged.connect(lambda _v: self._changed())
        self._pitch = ExprSpinBox(unit=prefs.default_unit)
        self._pitch.set_mm(1.5)
        self._pitch.value_changed.connect(lambda _v: self._changed())
        g.addWidget(self._turns, 2, 1)
        g.addWidget(self._pitch, 3, 1)
        root.addLayout(g)
        opts = QHBoxLayout()
        self._left_chk = QCheckBox("Left-handed")
        self._flip_chk = QCheckBox("Flip direction")
        for c in (self._left_chk, self._flip_chk):
            c.toggled.connect(lambda _c: self._changed())
            opts.addWidget(c)
        opts.addStretch()
        root.addLayout(opts)
        root.addWidget(self._separator())

        btns = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancelled)
        btns.addWidget(cancel)
        btns.addStretch()
        self._ok_btn = QPushButton("OK")
        self._ok_btn.setObjectName("ok")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._on_ok)
        btns.addWidget(self._ok_btn)
        root.addLayout(btns)
        self._update_rows()

    @staticmethod
    def _separator():
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(_SEP_STYLE)
        sep.setFixedHeight(1)
        return sep

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def set_source(self, kind: str, label: str, length: float | None = None):
        """A face (with its axial *length*) or a circle was picked."""
        self._source = kind
        self._source_label.setText(label)
        self._source_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #7ab3d4; font-size: 11px; font-family: monospace;"))
        if length is not None and self._full_chk.isChecked():
            self._height.set_mm(length)
        self.set_picking(False)
        self._update_rows()
        self._changed()

    def fields(self) -> dict | None:
        """HelixOp fields (no source), or None while incomplete."""
        if self._source is None:
            return None
        full = self._source == "face" and self._full_chk.isChecked()
        h = 0.0 if full else self._height.mm_value()
        pitch = self._pitch.mm_value() if self._spec.currentData() == "pitch" else 0.0
        if h is None or pitch is None or (not full and h <= 0):
            return None
        if self._spec.currentData() == "pitch" and pitch <= 0:
            return None
        return dict(height=h, turns=self._turns.value(), pitch=pitch,
                    left_hand=self._left_chk.isChecked(),
                    flip=self._flip_chk.isChecked())

    def set_fields(self, op, exprs: dict):
        self._full_chk.setChecked(op.face_ref is not None and op.height <= 0)
        if op.height > 0:
            self._height.set_binding(exprs.get("height"), op.height)
        self._spec.setCurrentIndex(1 if op.pitch > 0 else 0)
        if op.pitch > 0:
            self._pitch.set_binding(exprs.get("pitch"), op.pitch)
        self._turns.setValue(op.turns)
        self._left_chk.setChecked(op.left_hand)
        self._flip_chk.setChecked(op.flip)
        self._update_rows()

    def spinboxes(self) -> dict:
        out = {}
        if not (self._source == "face" and self._full_chk.isChecked()):
            out["height"] = self._height
        if self._spec.currentData() == "pitch":
            out["pitch"] = self._pitch
        return out

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _update_rows(self):
        face = self._source == "face"
        self._full_chk.setVisible(face)
        self._height.setEnabled(not (face and self._full_chk.isChecked()))

    def _on_full_toggled(self, _c):
        self._update_rows()
        self._changed()

    def _on_spec_changed(self, *_):
        pitch = self._spec.currentData() == "pitch"
        self._turns.setVisible(not pitch)
        self._pitch.setVisible(pitch)
        self._changed()

    def _changed(self):
        self._ok_btn.setEnabled(self.fields() is not None)
        self._timer.start()

    def set_picking(self, on: bool):
        self._pick_btn.setChecked(on)
        self._pick_btn.setProperty("active", on)
        self._pick_btn.style().unpolish(self._pick_btn)
        self._pick_btn.style().polish(self._pick_btn)
        if self._picking != on:
            self._picking = on
            self.picking_changed.emit(on)

    def _on_ok(self):
        f = self.fields()
        if f is not None:
            self.helix_requested.emit(f)

    def closeEvent(self, e):
        self.cancelled.emit()
        super().closeEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            e.accept()
        elif e.key() == Qt.Key.Key_Escape:
            if self._picking:
                self.set_picking(False)
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
