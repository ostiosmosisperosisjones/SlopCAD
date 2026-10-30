"""
gui/transform_panel.py

TransformPanel — move / rotate a body (or a copy).  Translate takes dx, dy,
dz; rotate an angle about a world axis or a picked edge of the body.
"""

from __future__ import annotations
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer

from gui.loft_panel import _PANEL_STYLE, _SEP_STYLE
from gui.expr_spinbox import ExprSpinBox


class TransformPanel(QWidget):
    transform_requested   = pyqtSignal(dict)    # TransformOp fields (no body / edge ref)
    cancelled             = pyqtSignal()
    picking_body_changed  = pyqtSignal(bool)
    picking_edge_changed  = pyqtSignal(bool)
    preview_changed       = pyqtSignal()

    def __init__(self, workspace, parent=None):
        super().__init__(parent.window() if parent is not None else None)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Transform")
        self.setObjectName("LoftPanel")             # loft panel styling
        self.setStyleSheet(prefs.scale_stylesheet(_PANEL_STYLE))
        self.setMinimumWidth(prefs.scaled_px(280))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._workspace = workspace
        self._body_id: str | None = None
        self._edge_set = False
        self._picking_body = self._picking_edge = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.preview_changed)
        self._build_ui()
        self._on_mode_changed()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)
        title = QLabel("Transform")
        title.setObjectName("title")
        root.addWidget(title)
        root.addWidget(self._separator())

        row = QHBoxLayout()
        row.addWidget(self._section("Body"))
        row.addStretch()
        self._pick_body_btn = QPushButton("Pick Body")
        self._pick_body_btn.setObjectName("pick_body")
        self._pick_body_btn.setCheckable(True)
        self._pick_body_btn.clicked.connect(
            lambda c: self._set_picking("body", c))
        row.addWidget(self._pick_body_btn)
        root.addLayout(row)
        self._body_label = self._value_label("—")
        root.addWidget(self._body_label)
        root.addWidget(self._separator())

        self._mode = QComboBox()
        self._mode.addItem("Translate", "translate")
        self._mode.addItem("Rotate", "rotate")
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        root.addWidget(self._mode)

        # Translate
        self._move_w = QWidget()
        g = QGridLayout(self._move_w)
        g.setContentsMargins(0, 0, 0, 0)
        self._dx, self._dy, self._dz = (ExprSpinBox(unit=prefs.default_unit)
                                        for _ in range(3))
        for r, (name, box) in enumerate((("X", self._dx), ("Y", self._dy),
                                         ("Z", self._dz))):
            box.set_mm(0.0)
            box.value_changed.connect(lambda _v: self._changed())
            g.addWidget(QLabel(name), r, 0)
            g.addWidget(box, r, 1)
        root.addWidget(self._move_w)

        # Rotate
        self._rot_w = QWidget()
        g = QGridLayout(self._rot_w)
        g.setContentsMargins(0, 0, 0, 0)
        self._angle = ExprSpinBox(unit="deg")
        self._angle.set_mm(90.0)
        self._angle.value_changed.connect(lambda _v: self._changed())
        g.addWidget(QLabel("Angle"), 0, 0)
        g.addWidget(self._angle, 0, 1)
        self._axis = QComboBox()
        for a in ("X", "Y", "Z"):
            self._axis.addItem(f"World {a}", a)
        self._axis.addItem("Body edge…", "edge")
        self._axis.setCurrentIndex(2)
        self._axis.currentIndexChanged.connect(self._on_axis_changed)
        g.addWidget(QLabel("Axis"), 1, 0)
        g.addWidget(self._axis, 1, 1)
        edge_row = QHBoxLayout()
        self._pick_edge_btn = QPushButton("Pick Edge")
        self._pick_edge_btn.setObjectName("pick_profile")
        self._pick_edge_btn.setCheckable(True)
        self._pick_edge_btn.clicked.connect(lambda c: self._set_picking("edge", c))
        edge_row.addWidget(self._pick_edge_btn)
        self._edge_label = self._value_label("straight or circular edge")
        edge_row.addWidget(self._edge_label, 1)
        self._edge_w = QWidget()
        self._edge_w.setLayout(edge_row)
        edge_row.setContentsMargins(0, 0, 0, 0)
        g.addWidget(self._edge_w, 2, 0, 1, 2)
        self._flip = QCheckBox("Reverse direction")
        self._flip.toggled.connect(lambda _c: self._changed())
        g.addWidget(self._flip, 3, 0, 1, 2)
        root.addWidget(self._rot_w)

        root.addWidget(self._separator())
        self._copy = QCheckBox("Copy (keep the original)")
        self._copy.toggled.connect(lambda _c: self._changed())
        root.addWidget(self._copy)
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

    def _separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(_SEP_STYLE)
        sep.setFixedHeight(1)
        return sep

    @staticmethod
    def _section(text):
        lbl = QLabel(text.upper())
        lbl.setObjectName("section")
        return lbl

    @staticmethod
    def _value_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        return lbl

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def set_body(self, body_id: str, name: str):
        self._body_id = body_id
        self._body_label.setText(name)
        self._body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #7ab3d4; font-size: 11px; font-family: monospace;"))
        self.set_edge(None)
        self._set_picking("body", False)
        self._changed()

    def set_edge(self, label: str | None):
        self._edge_set = label is not None
        self._edge_label.setText(label or "straight or circular edge")
        if label is not None:
            self._set_picking("edge", False)
            self._changed()

    def fields(self) -> dict | None:
        """TransformOp fields, or None while incomplete."""
        if self._body_id is None:
            return None
        mode = self._mode.currentData()
        f = dict(mode=mode, copy=self._copy.isChecked())
        if mode == "rotate":
            a = self._angle.mm_value()
            axis = self._axis.currentData()
            if a is None or (axis == "edge" and not self._edge_set):
                return None
            f.update(angle_deg=a, axis=axis, flip_axis=self._flip.isChecked())
        else:
            vals = [b.mm_value() for b in (self._dx, self._dy, self._dz)]
            if any(v is None for v in vals):
                return None
            f.update(dx=vals[0], dy=vals[1], dz=vals[2])
        return f

    def set_fields(self, op, exprs: dict):
        self._mode.setCurrentIndex(self._mode.findData(op.mode))
        for k in ("dx", "dy", "dz"):
            getattr(self, "_" + k).set_binding(exprs.get(k), getattr(op, k))
        self._angle.set_binding(exprs.get("angle_deg"), op.angle_deg)
        self._axis.setCurrentIndex(self._axis.findData(op.axis))
        self._flip.setChecked(op.flip_axis)
        self._copy.setChecked(op.copy)

    def spinboxes(self) -> dict:
        if self._mode.currentData() == "rotate":
            return {"angle_deg": self._angle}
        return {"dx": self._dx, "dy": self._dy, "dz": self._dz}

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _changed(self):
        self._ok_btn.setEnabled(self.fields() is not None)
        self._timer.start()

    def _on_mode_changed(self, *_):
        rot = self._mode.currentData() == "rotate"
        self._move_w.setVisible(not rot)
        self._rot_w.setVisible(rot)
        self._on_axis_changed()

    def _on_axis_changed(self, *_):
        is_edge = self._axis.currentData() == "edge"
        self._edge_w.setVisible(is_edge)
        if is_edge and not self._edge_set and self._mode.currentData() == "rotate":
            self._pick_edge_btn.setChecked(True)
            self._set_picking("edge", True)
        elif not is_edge:
            self._set_picking("edge", False)
        self._changed()

    def _set_picking(self, what: str, on: bool):
        btn = self._pick_body_btn if what == "body" else self._pick_edge_btn
        if on:                                   # one pick mode at a time
            self._set_picking("edge" if what == "body" else "body", False)
        btn.setChecked(on)
        btn.setProperty("active", on)
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        if what == "body":
            if self._picking_body != on:
                self._picking_body = on
                self.picking_body_changed.emit(on)
        elif self._picking_edge != on:
            self._picking_edge = on
            self.picking_edge_changed.emit(on)

    def _on_ok(self):
        f = self.fields()
        if f is not None:
            self.transform_requested.emit(f)

    def closeEvent(self, e):
        self.cancelled.emit()
        super().closeEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            e.accept()
        elif e.key() == Qt.Key.Key_Escape:
            if self._picking_body or self._picking_edge:
                self._set_picking("body", False)
                self._set_picking("edge", False)
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
