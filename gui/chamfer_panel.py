"""
gui/chamfer_panel.py

ChamferPanel — floating panel for 3D edge chamfer operations.

Distance + angle (degrees) instead of fillet's radius, plus a
"Flip reference face" checkbox so users can mirror an asymmetric chamfer
across the picked edge.
"""

from __future__ import annotations
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QDoubleSpinBox, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QKeyEvent

from gui.selection_list import SelectionList


_PANEL_STYLE = """
QWidget#ChamferPanel {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
}
QLabel { color: #d4d4d4; font-size: 12px; }
QLabel#title { color: #ffffff; font-size: 13px; font-weight: bold; }
QLabel#section {
    color: #888; font-size: 10px;
    text-transform: uppercase; letter-spacing: 1px;
}
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
QPushButton#pick_face[active=true] {
    background: #2a1e1e; border-color: #d96a4a; color: #ff9977;
}
QPushButton#pick_edge[active=true] {
    background: #2a1e3a; border-color: #aa6aee; color: #cc8dff;
}
QDoubleSpinBox {
    background: #2a2a2a; color: #d4d4d4;
    border: 1px solid #444; border-radius: 3px;
    padding: 2px 6px; font-size: 12px; font-family: monospace;
}
QDoubleSpinBox:focus { border-color: #4a90d9; }
QCheckBox { color: #d4d4d4; font-size: 12px; spacing: 6px; }
QCheckBox::indicator {
    width: 13px; height: 13px; border-radius: 3px;
    border: 1px solid #555; background: #2a2a2a;
}
QCheckBox::indicator:checked {
    background: #4a90d9; border-color: #4a90d9;
}
"""

_SEP_STYLE = "background: #333;"


class ChamferPanel(QWidget):
    confirmed            = pyqtSignal(float, float, bool)
    # (distance_mm, angle_deg, flip_reference_face)
    cancelled            = pyqtSignal()
    preview_changed      = pyqtSignal()
    face_entry_removed   = pyqtSignal(int)
    edge_entry_removed   = pyqtSignal(int)
    picking_face_changed = pyqtSignal(bool)
    picking_edge_changed = pyqtSignal(bool)

    def __init__(self, workspace, parent=None):
        super().__init__(parent.window() if parent is not None else None)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Chamfer")
        self._viewport = parent
        self.setObjectName("ChamferPanel")
        self.setStyleSheet(prefs.scale_stylesheet(_PANEL_STYLE))
        self.setMinimumWidth(prefs.scaled_px(260))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._workspace    = workspace
        self._picking_face = False
        self._picking_edge = False

        # Modest debounce so dragging the spinboxes doesn't queue many builds.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)
        self._preview_timer.timeout.connect(self.preview_changed.emit)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        from gui.expr_spinbox import ExprSpinBox

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Chamfer")
        title.setObjectName("title")
        root.addWidget(title)

        root.addWidget(self._separator())

        # ── Faces ─────────────────────────────────────────────────────
        face_header = QHBoxLayout()
        face_header.setSpacing(6)
        face_header.addWidget(self._section_label("Faces"))
        face_header.addStretch()
        self._pick_face_btn = QPushButton("+ Add")
        self._pick_face_btn.setObjectName("pick_face")
        self._pick_face_btn.setCheckable(True)
        self._pick_face_btn.clicked.connect(self._on_pick_face_toggle)
        face_header.addWidget(self._pick_face_btn)
        root.addLayout(face_header)

        self._face_list = SelectionList(empty_text="No faces selected")
        self._face_list.entry_removed.connect(self.face_entry_removed)
        root.addWidget(self._face_list)

        root.addWidget(self._separator())

        # ── Edges ─────────────────────────────────────────────────────
        edge_header = QHBoxLayout()
        edge_header.setSpacing(6)
        edge_header.addWidget(self._section_label("Edges"))
        edge_header.addStretch()
        self._pick_edge_btn = QPushButton("+ Add")
        self._pick_edge_btn.setObjectName("pick_edge")
        self._pick_edge_btn.setCheckable(True)
        self._pick_edge_btn.clicked.connect(self._on_pick_edge_toggle)
        edge_header.addWidget(self._pick_edge_btn)
        root.addLayout(edge_header)

        self._edge_list = SelectionList(empty_text="No edges selected")
        self._edge_list.entry_removed.connect(self.edge_entry_removed)
        root.addWidget(self._edge_list)

        root.addWidget(self._separator())

        # ── Distance ──────────────────────────────────────────────────
        root.addWidget(self._section_label("Distance"))
        self._distance_spin = ExprSpinBox(unit=prefs.default_unit)
        self._distance_spin.set_mm(1.0)
        self._distance_spin.value_changed.connect(self._on_value_changed)
        root.addWidget(self._distance_spin)

        # ── Angle ─────────────────────────────────────────────────────
        root.addWidget(self._section_label("Angle (°)"))
        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(0.1, 89.9)
        self._angle_spin.setDecimals(2)
        self._angle_spin.setSingleStep(1.0)
        self._angle_spin.setValue(45.0)
        self._angle_spin.setSuffix(" °")
        self._angle_spin.valueChanged.connect(self._on_value_changed_f)
        root.addWidget(self._angle_spin)

        # ── Flip reference face ───────────────────────────────────────
        self._flip_chk = QCheckBox("Flip reference face")
        self._flip_chk.setToolTip(
            "Measure the angle from the other adjacent face.  Useful when\n"
            "the default chamfer biases the wrong way on an asymmetric edge.")
        self._flip_chk.toggled.connect(lambda _c: self._emit_preview())
        root.addWidget(self._flip_chk)

        root.addWidget(self._separator())

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancelled)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        self._ok_btn = QPushButton("OK")
        self._ok_btn.setObjectName("ok")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(self._ok_btn)
        root.addLayout(btn_row)

    def _separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(_SEP_STYLE)
        sep.setFixedHeight(1)
        return sep

    def _section_label(self, text):
        lbl = QLabel(text.upper())
        lbl.setObjectName("section")
        return lbl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_distance(self, mm: float):
        self._distance_spin.set_mm(mm)

    def set_angle(self, deg: float):
        self._angle_spin.setValue(deg)

    def set_flip(self, flip: bool):
        self._flip_chk.setChecked(bool(flip))

    def add_face_entry(self, body_id: str | None, face_idx: int | None,
                       label: str):
        self._face_list.add((body_id, face_idx), label)

    def remove_face_entry(self, index: int):
        self._face_list.remove_at(index)

    def clear_face_entries(self):
        self._face_list.clear()

    def add_edge_entry(self, body_id: str | None, edge_idx: int | None,
                       label: str):
        self._edge_list.add((body_id, edge_idx), label)

    def remove_edge_entry(self, index: int):
        self._edge_list.remove_at(index)

    def clear_edge_entries(self):
        self._edge_list.clear()

    def set_face_entry_error(self, index: int, message: str):
        self._face_list.set_error(index, message)

    def set_edge_entry_error(self, index: int, message: str):
        self._edge_list.set_error(index, message)

    def clear_entry_errors(self):
        self._face_list.clear_errors()
        self._edge_list.clear_errors()

    @property
    def _has_selection(self) -> bool:
        return len(self._face_list) > 0 or len(self._edge_list) > 0

    def end_pick_face(self):
        self._picking_face = False
        self._pick_face_btn.setChecked(False)
        self._pick_face_btn.setProperty("active", False)
        self._pick_face_btn.style().unpolish(self._pick_face_btn)
        self._pick_face_btn.style().polish(self._pick_face_btn)
        self.picking_face_changed.emit(False)

    def end_pick_edge(self):
        self._picking_edge = False
        self._pick_edge_btn.setChecked(False)
        self._pick_edge_btn.setProperty("active", False)
        self._pick_edge_btn.style().unpolish(self._pick_edge_btn)
        self._pick_edge_btn.style().polish(self._pick_edge_btn)
        self.picking_edge_changed.emit(False)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _emit_preview(self):
        self._preview_timer.start()

    def _on_value_changed(self, _):
        self._emit_preview()

    def _on_value_changed_f(self, _f: float):
        self._emit_preview()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_pick_face_toggle(self, checked: bool):
        if checked:
            self.end_pick_edge()
            self._picking_face = True
            self._pick_face_btn.setProperty("active", True)
            self._pick_face_btn.style().unpolish(self._pick_face_btn)
            self._pick_face_btn.style().polish(self._pick_face_btn)
            self.picking_face_changed.emit(True)
        else:
            self.end_pick_face()

    def _on_pick_edge_toggle(self, checked: bool):
        if checked:
            self.end_pick_face()
            self._picking_edge = True
            self._pick_edge_btn.setProperty("active", True)
            self._pick_edge_btn.style().unpolish(self._pick_edge_btn)
            self._pick_edge_btn.style().polish(self._pick_edge_btn)
            self.picking_edge_changed.emit(True)
        else:
            self.end_pick_edge()

    def _on_ok(self):
        if not self._has_selection:
            return
        dist  = self._distance_spin.mm_value()
        angle = float(self._angle_spin.value())
        if dist is None or dist <= 0:
            return
        if not (0.0 < angle < 90.0):
            return
        self.confirmed.emit(dist, angle, self._flip_chk.isChecked())

    # ------------------------------------------------------------------
    # Keyboard / window
    # ------------------------------------------------------------------

    def closeEvent(self, e):
        self.cancelled.emit()
        super().closeEvent(e)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            e.accept()
        elif e.key() == Qt.Key.Key_Escape:
            if self._picking_face:
                self.end_pick_face()
            elif self._picking_edge:
                self.end_pick_edge()
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
