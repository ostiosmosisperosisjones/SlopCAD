"""
gui/revolve_panel.py

RevolvePanel — floating non-modal panel for the revolve operation.
"""

from __future__ import annotations
import numpy as np
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QFrame, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from gui.fillet3d_panel import _adaptive_delay
from PyQt6.QtGui import QKeyEvent

from gui.selection_list import SelectionList


_PANEL_STYLE = """
QWidget#RevolvePanel {
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
QLabel#section {
    color: #888;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
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
QPushButton#pick_axis[active=true] {
    background: #2a3a1e;
    border-color: #6aaa44;
    color: #8dcc5a;
}
QPushButton#pick_face[active=true] {
    background: #2a1e1e;
    border-color: #d96a4a;
    color: #ff9977;
}
QPushButton#pick_body[active=true] {
    background: #1e2a3a;
    border-color: #4a90d9;
    color: #7ab3d4;
}
QRadioButton {
    color: #d4d4d4;
    font-size: 12px;
    spacing: 6px;
}
QRadioButton::indicator {
    width: 13px; height: 13px;
    border-radius: 7px;
    border: 1px solid #555;
    background: #2a2a2a;
}
QRadioButton::indicator:checked {
    background: #4a90d9;
    border-color: #4a90d9;
}
"""

_SEP_STYLE = "background: #333;"


class RevolvePanel(QWidget):
    # (angle_deg, axis_point|None, axis_dir|None, merge_body_id|None|"__new_body__", is_cut)
    revolve_requested    = pyqtSignal(float, object, object, object, bool)
    cancelled            = pyqtSignal()
    picking_axis_changed = pyqtSignal(bool)   # True while user picks an axis edge/line
    picking_face_changed = pyqtSignal(bool)
    picking_body_changed = pyqtSignal(bool)
    preview_changed      = pyqtSignal(float, object, object)  # (angle, pt, dir)
    face_entry_removed   = pyqtSignal(int)

    def __init__(self, workspace, parent=None):
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.Tool |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint)
        self._viewport = parent
        self.setObjectName("RevolvePanel")
        self.setStyleSheet(prefs.scale_stylesheet(_PANEL_STYLE))
        self.setMinimumWidth(prefs.scaled_px(260))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._workspace      = workspace
        self._picking_axis   : bool = False
        self._picking_face   : bool = False
        self._picking_body   : bool = False
        self._flip           : bool = False

        self._axis_point     : np.ndarray | None = None   # world-space point on axis
        self._axis_dir       : np.ndarray | None = None   # stored unit axis (unflipped)
        self._merge_body_id  : str | None = None
        self._self_cut       : bool = True   # cleared once user picks a target body

        self._preview_timer      = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._fire_preview)
        self._preview_last_value = None
        self._preview_last_ms    = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        from gui.expr_spinbox import ExprSpinBox
        from cad.prefs import prefs

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Revolve / Cut")
        title.setObjectName("title")
        root.addWidget(title)

        root.addWidget(self._separator())

        # ── Profile (face / sketch) ───────────────────────────────────
        face_header = QHBoxLayout()
        face_header.setSpacing(6)
        face_header.addWidget(self._section_label("Profile"))
        face_header.addStretch()
        self._pick_face_btn = QPushButton("+ Add Face")
        self._pick_face_btn.setObjectName("pick_face")
        self._pick_face_btn.setCheckable(True)
        self._pick_face_btn.clicked.connect(self._on_pick_face_toggle)
        face_header.addWidget(self._pick_face_btn)
        root.addLayout(face_header)

        self._face_list = SelectionList(empty_text="No profile selected")
        self._face_list.entry_removed.connect(self.face_entry_removed)
        root.addWidget(self._face_list)

        root.addWidget(self._separator())

        # ── Mode (Revolve / Cut) ──────────────────────────────────────
        root.addWidget(self._section_label("Mode"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self._radio_revolve = QRadioButton("Revolve")
        self._radio_cut     = QRadioButton("Cut")
        self._radio_revolve.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._radio_revolve, 0)
        self._mode_group.addButton(self._radio_cut,     1)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        mode_row.addWidget(self._radio_revolve)
        mode_row.addWidget(self._radio_cut)
        root.addLayout(mode_row)

        root.addWidget(self._separator())

        # ── Axis ──────────────────────────────────────────────────────
        axis_header = QHBoxLayout()
        axis_header.setSpacing(6)
        axis_header.addWidget(self._section_label("Axis"))
        axis_header.addStretch()
        self._pick_axis_btn = QPushButton("Pick Line / Edge")
        self._pick_axis_btn.setObjectName("pick_axis")
        self._pick_axis_btn.setCheckable(True)
        self._pick_axis_btn.clicked.connect(self._on_pick_axis_toggle)
        axis_header.addWidget(self._pick_axis_btn)
        root.addLayout(axis_header)

        self._axis_label = QLabel("No axis selected")
        self._axis_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._axis_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        axis_reset_row = QHBoxLayout()
        axis_reset_row.setSpacing(6)
        axis_reset_row.addWidget(self._axis_label)
        self._flip_btn = QPushButton("⇅")
        self._flip_btn.setObjectName("pick_axis")
        self._flip_btn.setToolTip("Flip revolution direction")
        self._flip_btn.setCheckable(True)
        self._flip_btn.setFixedWidth(prefs.scaled_px(28))
        self._flip_btn.clicked.connect(self._on_flip)
        axis_reset_row.addWidget(self._flip_btn)
        reset_axis_btn = QPushButton("✕")
        reset_axis_btn.setFixedWidth(prefs.scaled_px(24))
        reset_axis_btn.clicked.connect(self._reset_axis)
        axis_reset_row.addWidget(reset_axis_btn)
        root.addLayout(axis_reset_row)

        root.addWidget(self._separator())

        # ── Angle ─────────────────────────────────────────────────────
        root.addWidget(self._section_label("Angle"))
        self._angle_spinbox = ExprSpinBox(unit="deg")
        self._angle_spinbox.set_mm(360.0)
        self._angle_spinbox.value_changed.connect(self._emit_preview)
        root.addWidget(self._angle_spinbox)

        root.addWidget(self._separator())

        # ── Operation (Revolve: New Body / Merge with) ────────────────
        self._op_section_label = self._section_label("Operation")
        root.addWidget(self._op_section_label)

        # Revolve operation widget
        self._revolve_op_widget = QWidget()
        revolve_op_layout = QVBoxLayout(self._revolve_op_widget)
        revolve_op_layout.setContentsMargins(0, 0, 0, 0)
        revolve_op_layout.setSpacing(6)

        self._radio_new   = QRadioButton("New Body")
        self._radio_merge = QRadioButton("Merge with")
        self._radio_new.setChecked(True)
        self._op_group = QButtonGroup(self)
        self._op_group.addButton(self._radio_new,   0)
        self._op_group.addButton(self._radio_merge, 1)
        self._op_group.idClicked.connect(self._on_op_changed)
        revolve_op_layout.addWidget(self._radio_new)

        merge_row = QHBoxLayout()
        merge_row.setSpacing(6)
        merge_row.addWidget(self._radio_merge)
        self._pick_body_btn = QPushButton("Pick Body")
        self._pick_body_btn.setObjectName("pick_body")
        self._pick_body_btn.setEnabled(False)
        self._pick_body_btn.setCheckable(True)
        self._pick_body_btn.clicked.connect(self._on_pick_body_toggle)
        merge_row.addWidget(self._pick_body_btn)
        self._body_label = QLabel("—")
        self._body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        merge_row.addWidget(self._body_label)
        revolve_op_layout.addLayout(merge_row)
        root.addWidget(self._revolve_op_widget)

        # Cut operation widget — "Cut from: <pick body>" (blank = no-target)
        self._cut_op_widget = QWidget()
        cut_op_layout = QHBoxLayout(self._cut_op_widget)
        cut_op_layout.setContentsMargins(0, 0, 0, 0)
        cut_op_layout.setSpacing(6)
        self._pick_cut_body_btn = QPushButton("Pick Body")
        self._pick_cut_body_btn.setObjectName("pick_body")
        self._pick_cut_body_btn.setCheckable(True)
        self._pick_cut_body_btn.clicked.connect(self._on_pick_body_toggle)
        cut_op_layout.addWidget(self._pick_cut_body_btn)
        self._cut_body_label = QLabel("—")
        self._cut_body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        cut_op_layout.addWidget(self._cut_body_label)
        root.addWidget(self._cut_op_widget)
        self._cut_op_widget.hide()

        root.addWidget(self._separator())

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancelled)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("ok")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(ok_btn)
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

    @property
    def _has_face(self) -> bool:
        return self._face_list.has_valid_entries

    def add_face_entry(self, body_id, face_idx, label: str, valid: bool = True):
        key = (body_id, face_idx) if body_id is not None else label
        self._face_list.add(key, label, valid=valid)

    def clear_face_entries(self):
        self._face_list.clear()

    def set_axis(self, point: np.ndarray, direction: np.ndarray):
        """Called when the user picks an edge or sketch line to use as the axis."""
        self._axis_point = point.copy()
        self._axis_dir   = direction / np.linalg.norm(direction)
        x, y, z = self._axis_dir
        self._axis_label.setText(f"{x:+.2f}, {y:+.2f}, {z:+.2f}")
        self._axis_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #8dcc5a; font-size: 11px; font-family: monospace;"))
        self._end_pick_axis()
        self._emit_preview()

    def set_merge_body(self, body_id: str, body_name: str):
        """Called by viewport when the user picks a target body for merge/cut."""
        self._merge_body_id = body_id
        self._self_cut = False
        self._end_pick_body()
        self._body_label.setText(body_name)
        self._body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #7ab3d4; font-size: 11px; font-family: monospace;"))
        self._cut_body_label.setText(body_name)
        self._cut_body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #7ab3d4; font-size: 11px; font-family: monospace;"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def _effective_axis_dir(self) -> np.ndarray | None:
        if self._axis_dir is None:
            return None
        return -self._axis_dir if self._flip else self._axis_dir

    def _on_flip(self, checked: bool):
        self._flip = checked
        self._flip_btn.setText("⇵" if checked else "⇅")
        self._emit_preview()

    def _emit_preview(self, _=None):
        self._preview_timer.start(_adaptive_delay(
            self._angle_spinbox.mm_value(), self))

    def _fire_preview(self):
        angle = self._angle_spinbox.mm_value()
        if angle is None:
            return
        self.preview_changed.emit(float(angle), self._axis_point, self._effective_axis_dir)

    def _on_mode_changed(self, btn_id: int):
        is_cut = btn_id == 1
        self._revolve_op_widget.setVisible(not is_cut)
        self._cut_op_widget.setVisible(is_cut)
        # Reset target body so the user must re-pick (or accept self-cut)
        self._merge_body_id = None
        self._self_cut = True
        self._body_label.setText("—")
        self._body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._cut_body_label.setText("—")
        self._cut_body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._emit_preview()

    def _on_op_changed(self, btn_id: int):
        self._pick_body_btn.setEnabled(btn_id == 1)
        if btn_id == 0:
            self._merge_body_id = None
            self._body_label.setText("—")
            self._body_label.setStyleSheet(prefs.scale_stylesheet(
                "color: #888; font-size: 11px; font-family: monospace;"))
            self._end_pick_body()

    def _on_pick_axis_toggle(self, checked: bool):
        if checked:
            self._picking_axis = True
            self._pick_axis_btn.setProperty("active", True)
            self._pick_axis_btn.style().unpolish(self._pick_axis_btn)
            self._pick_axis_btn.style().polish(self._pick_axis_btn)
            self.picking_axis_changed.emit(True)
        else:
            self._end_pick_axis()

    def _end_pick_axis(self):
        self._picking_axis = False
        self._pick_axis_btn.setChecked(False)
        self._pick_axis_btn.setProperty("active", False)
        self._pick_axis_btn.style().unpolish(self._pick_axis_btn)
        self._pick_axis_btn.style().polish(self._pick_axis_btn)
        self.picking_axis_changed.emit(False)

    def _on_pick_face_toggle(self, checked: bool):
        if checked:
            self._picking_face = True
            self._pick_face_btn.setProperty("active", True)
            self._pick_face_btn.style().unpolish(self._pick_face_btn)
            self._pick_face_btn.style().polish(self._pick_face_btn)
            self.picking_face_changed.emit(True)
        else:
            self._end_pick_face()

    def _end_pick_face(self):
        self._picking_face = False
        self._pick_face_btn.setChecked(False)
        self._pick_face_btn.setProperty("active", False)
        self._pick_face_btn.style().unpolish(self._pick_face_btn)
        self._pick_face_btn.style().polish(self._pick_face_btn)
        self.picking_face_changed.emit(False)

    def _on_pick_body_toggle(self, checked: bool):
        if checked:
            self._picking_body = True
            self._pick_body_btn.setProperty("active", True)
            self._pick_body_btn.style().unpolish(self._pick_body_btn)
            self._pick_body_btn.style().polish(self._pick_body_btn)
            self.picking_body_changed.emit(True)
        else:
            self._end_pick_body()

    def _end_pick_body(self):
        self._picking_body = False
        self._pick_body_btn.setChecked(False)
        self._pick_body_btn.setProperty("active", False)
        self._pick_body_btn.style().unpolish(self._pick_body_btn)
        self._pick_body_btn.style().polish(self._pick_body_btn)
        self.picking_body_changed.emit(False)

    def _reset_axis(self):
        self._axis_point = None
        self._axis_dir   = None
        self._axis_label.setText("No axis selected")
        self._axis_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._end_pick_axis()
        self._emit_preview()

    def _on_ok(self):
        if not self._has_face:
            return
        if self._axis_point is None or self._axis_dir is None:
            return
        self._angle_spinbox._on_commit()
        angle = self._angle_spinbox.mm_value()
        if angle is None or angle == 0:
            return

        is_cut = self._mode_group.checkedId() == 1
        if is_cut:
            # Cross-body cut requires a picked target; self-cut/no-target is
            # allowed (merge_body_id=None signals fan-out across intersections).
            merge_body_id = self._merge_body_id  # None for self/no-target
        else:
            op_id = self._op_group.checkedId()
            if op_id == 1:
                if self._merge_body_id is None:
                    return
                merge_body_id = self._merge_body_id
            else:
                merge_body_id = "__new_body__"

        self.revolve_requested.emit(
            float(angle),
            self._axis_point,
            self._effective_axis_dir,
            merge_body_id,
            is_cut,
        )

    # ------------------------------------------------------------------
    # Keyboard / close
    # ------------------------------------------------------------------

    def closeEvent(self, e):
        self.cancelled.emit()
        super().closeEvent(e)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            e.accept()
        elif e.key() == Qt.Key.Key_Escape:
            if self._picking_face:
                self._end_pick_face()
            elif self._picking_axis:
                self._end_pick_axis()
            elif self._picking_body:
                self._end_pick_body()
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
