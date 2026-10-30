"""
gui/coil_panel.py

CoilPanel — LoftPanel's mode / target handling around a coil: pick a sketch
circle (the axis + diameter), then height, turns and section.  The profile
list and loft surface options are hidden.
"""

from __future__ import annotations
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QHBoxLayout, QGridLayout, QLabel, QPushButton, QCheckBox, QComboBox,
    QDoubleSpinBox,
)
from PyQt6.QtCore import pyqtSignal, QTimer

from gui.loft_panel import LoftPanel
from gui.expr_spinbox import ExprSpinBox


class CoilPanel(LoftPanel):
    coil_requested        = pyqtSignal(object, object, dict)
    # (mode_str, target_body_id|None, fields) — fields: circle_sketch_id,
    # circle_uid, height, turns, section, section_size, left_hand, flip
    picking_circle_changed = pyqtSignal(bool)

    def __init__(self, workspace, parent=None):
        self._picking_circle = False
        self._circle: tuple | None = None          # (sketch_id, uid)
        super().__init__(workspace, parent)
        self.setWindowTitle("Coil")
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self.preview_changed)

    def _build_ui(self):
        super()._build_ui()
        root = self.layout()
        root.itemAt(0).widget().setText("Coil")
        self._radio_loft.setText("Coil")

        # Profiles section → CIRCLE.  Header row is just above the list.
        at = root.indexOf(self._profile_list)
        hdr = root.itemAt(at - 1).layout()
        hdr.itemAt(0).widget().setText("CIRCLE")
        self._pick_profile_btn.hide()
        self._profile_list.hide()
        self._pick_circle_btn = QPushButton("Pick Circle")
        self._pick_circle_btn.setObjectName("pick_profile")
        self._pick_circle_btn.setCheckable(True)
        self._pick_circle_btn.clicked.connect(self._on_pick_circle_toggle)
        hdr.addWidget(self._pick_circle_btn)
        self._circle_label = QLabel("Pick a sketch circle — the coil's axis + diameter")
        self._circle_label.setWordWrap(True)
        self._circle_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #555; font-size: 11px; font-family: monospace;"))
        root.insertWidget(at, self._circle_label)

        # Loft SURFACE section → SIZE: [header, continuity row, ruled].
        surf_idx = root.indexOf(self._ruled_chk)
        root.itemAt(surf_idx - 2).widget().setText("SIZE")
        row = root.itemAt(surf_idx - 1).layout()
        for j in range(row.count()):
            row.itemAt(j).widget().hide()
        self._ruled_chk.hide()

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self._height = ExprSpinBox(unit=prefs.default_unit)
        self._height.set_mm(20.0)
        self._turns = QDoubleSpinBox()
        self._turns.setRange(0.1, 1000.0)
        self._turns.setDecimals(2)
        self._turns.setValue(5.0)
        self._section = QComboBox()
        self._section.addItem("Circle", "circle")
        self._section.addItem("Square", "square")
        self._size = ExprSpinBox(unit=prefs.default_unit)
        self._size.set_mm(2.0)
        for r, (name, w) in enumerate((("Height", self._height),
                                       ("Turns", self._turns),
                                       ("Section", self._section),
                                       ("Section size", self._size))):
            grid.addWidget(QLabel(name), r, 0)
            grid.addWidget(w, r, 1)
        root.insertLayout(surf_idx + 1, grid)
        opts = QHBoxLayout()
        self._left_chk = QCheckBox("Left-handed")
        self._flip_chk = QCheckBox("Flip direction")
        opts.addWidget(self._left_chk)
        opts.addWidget(self._flip_chk)
        opts.addStretch()
        root.insertLayout(surf_idx + 2, opts)

        self._height.value_changed.connect(lambda _v: self._schedule())
        self._size.value_changed.connect(lambda _v: self._schedule())
        self._turns.valueChanged.connect(lambda _v: self._schedule())
        self._section.currentIndexChanged.connect(lambda _i: self._schedule())
        self._left_chk.toggled.connect(lambda _c: self._schedule())
        self._flip_chk.toggled.connect(lambda _c: self._schedule())

    def _schedule(self):
        self._preview_timer.start()

    # ------------------------------------------------------------------
    # Circle
    # ------------------------------------------------------------------

    def set_circle(self, sketch_id: str, uid: str, label: str):
        self._circle = (sketch_id, uid)
        self._circle_label.setText(label)
        self._circle_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #7ab3d4; font-size: 11px; font-family: monospace;"))
        if self._picking_circle:
            self._end_pick_circle()
        self._emit_changed()

    def circle(self) -> tuple | None:
        return self._circle

    def _on_pick_circle_toggle(self, checked: bool):
        if checked:
            self._picking_circle = True
            self._set_active(self._pick_circle_btn, True)
            self.picking_circle_changed.emit(True)
        else:
            self._end_pick_circle()

    def _end_pick_circle(self):
        self._picking_circle = False
        self._pick_circle_btn.setChecked(False)
        self._set_active(self._pick_circle_btn, False)
        self.picking_circle_changed.emit(False)

    @staticmethod
    def _set_active(btn, on: bool):
        btn.setProperty("active", on)
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    def fields(self) -> dict | None:
        """The op fields, or None while the panel is incomplete / invalid."""
        h, s = self._height.mm_value(), self._size.mm_value()
        if self._circle is None or not h or not s:
            return None
        return dict(circle_sketch_id=self._circle[0], circle_uid=self._circle[1],
                    height=h, turns=self._turns.value(),
                    section=self._section.currentData(), section_size=s,
                    left_hand=self._left_chk.isChecked(),
                    flip=self._flip_chk.isChecked())

    def set_fields(self, op, exprs: dict):
        self._height.set_binding(exprs.get("height"), op.height)
        self._size.set_binding(exprs.get("section_size"), op.section_size)
        self._turns.setValue(op.turns)
        self._section.setCurrentIndex(max(0, self._section.findData(op.section)))
        self._left_chk.setChecked(op.left_hand)
        self._flip_chk.setChecked(op.flip)

    def spinboxes(self) -> dict:
        return {"height": self._height, "section_size": self._size}

    # ------------------------------------------------------------------
    # OK gating / emit
    # ------------------------------------------------------------------

    def _update_ok_enabled(self):
        if not hasattr(self, "_size"):             # during base __init__
            return super()._update_ok_enabled()
        if self.fields() is None:
            self._ok_btn.setEnabled(False)
            self._ok_btn.setToolTip("Pick a sketch circle")
            return
        if self._mode_str() == "merge" and self._target_body_id is None:
            self._ok_btn.setEnabled(False)
            self._ok_btn.setToolTip("Pick a body to merge with")
            return
        self._ok_btn.setEnabled(True)
        self._ok_btn.setToolTip(
            "Cut through every intersecting body"
            if self._mode_str() == "cut" and self._target_body_id is None else "")

    def _on_ok(self):
        f = self.fields()
        if f is None:
            return
        mode = self._mode_str()
        if mode == "merge" and self._target_body_id is None:
            return
        self.coil_requested.emit(mode, self._target_body_id, f)

    def keyPressEvent(self, e):
        from PyQt6.QtCore import Qt
        if e.key() == Qt.Key.Key_Escape and self._picking_circle:
            self._end_pick_circle()
            return
        super().keyPressEvent(e)
