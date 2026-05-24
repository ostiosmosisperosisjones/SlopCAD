"""
gui/extrude_panel.py

ExtrudePanel — floating non-modal panel for the extrude/cut operation.
"""

from __future__ import annotations
import numpy as np
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QFrame, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QKeyEvent

from gui.selection_list import SelectionList


_PANEL_STYLE = """
QWidget#ExtrudePanel {
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
QPushButton#pick_edge[active=true] {
    background: #2a3a1e;
    border-color: #6aaa44;
    color: #8dcc5a;
}
QPushButton#pick_vertex[active=true] {
    background: #2a1e3a;
    border-color: #aa6aee;
    color: #cc8dff;
}
QPushButton#pick_body[active=true] {
    background: #1e2a3a;
    border-color: #4a90d9;
    color: #7ab3d4;
}
QPushButton#pick_face[active=true] {
    background: #2a1e1e;
    border-color: #d96a4a;
    color: #ff9977;
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


class ExtrudePanel(QWidget):
    extrude_requested      = pyqtSignal(float, object, object)  # (dist_mm, dir|None, body_id|None)
    cancelled              = pyqtSignal()
    picking_edge_changed   = pyqtSignal(bool)
    picking_vertex_changed = pyqtSignal(bool)
    picking_body_changed   = pyqtSignal(bool)
    picking_face_changed   = pyqtSignal(bool)
    preview_changed        = pyqtSignal(float, object)
    face_entry_removed     = pyqtSignal(int)   # index of removed entry (forwarded from SelectionList)

    def __init__(self, workspace, parent=None):
        # Parent to the top-level main window so this Tool window rides its
        # z-order (raises when the main window is activated).
        super().__init__(parent.window() if parent is not None else None)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Extrude / Cut")
        self._viewport = parent  # keep reference for positioning
        self.setObjectName("ExtrudePanel")
        self.setStyleSheet(prefs.scale_stylesheet(_PANEL_STYLE))
        self.setMinimumWidth(prefs.scaled_px(260))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._workspace      = workspace
        self._direction      : np.ndarray | None = None
        self._flip           : bool = False
        self._picking_edge   : bool = False
        self._picking_vertex : bool = False
        self._picking_body   : bool = False
        self._picking_face   : bool = False

        self._target_vertex  : np.ndarray | None = None
        self._face_origin    : np.ndarray | None = None
        self._face_normal    : np.ndarray | None = None
        self._vertex_dist_mm : float | None = None
        self._merge_body_id  : str | None = None   # picked target body for merge/cut
        self._self_cut       : bool = True          # False only when user explicitly picks a cross-body cut target

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

        title = QLabel("Extrude / Cut")
        title.setObjectName("title")
        root.addWidget(title)

        root.addWidget(self._separator())

        # ── Face ──────────────────────────────────────────────────────
        face_header = QHBoxLayout()
        face_header.setSpacing(6)
        face_header.addWidget(self._section_label("Faces"))
        face_header.addStretch()
        self._pick_face_btn = QPushButton("+ Add Face")
        self._pick_face_btn.setObjectName("pick_face")
        self._pick_face_btn.setCheckable(True)
        self._pick_face_btn.clicked.connect(self._on_pick_face_toggle)
        face_header.addWidget(self._pick_face_btn)
        root.addLayout(face_header)

        self._face_list = SelectionList(empty_text="No faces selected")
        self._face_list.entry_removed.connect(self.face_entry_removed)
        root.addWidget(self._face_list)

        root.addWidget(self._separator())

        # ── Mode ──────────────────────────────────────────────────────
        root.addWidget(self._section_label("Mode"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self._radio_extrude = QRadioButton("Extrude")
        self._radio_cut     = QRadioButton("Cut")
        self._radio_extrude.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._radio_extrude, 0)
        self._mode_group.addButton(self._radio_cut,     1)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        mode_row.addWidget(self._radio_extrude)
        mode_row.addWidget(self._radio_cut)
        root.addLayout(mode_row)

        root.addWidget(self._separator())

        # ── Target ────────────────────────────────────────────────────
        root.addWidget(self._section_label("Target"))
        target_row = QHBoxLayout()
        target_row.setSpacing(6)
        self._radio_manual = QRadioButton("Distance")
        self._radio_vertex = QRadioButton("Up to Vertex")
        self._radio_manual.setChecked(True)
        self._target_group = QButtonGroup(self)
        self._target_group.addButton(self._radio_manual, 0)
        self._target_group.addButton(self._radio_vertex, 1)
        self._target_group.idClicked.connect(self._on_target_mode_changed)
        target_row.addWidget(self._radio_manual)
        target_row.addWidget(self._radio_vertex)
        root.addLayout(target_row)

        # Distance input
        self._dist_widget = QWidget()
        dist_layout = QVBoxLayout(self._dist_widget)
        dist_layout.setContentsMargins(0, 0, 0, 0)
        dist_layout.setSpacing(2)
        self._spinbox = ExprSpinBox(unit=prefs.default_unit)
        self._spinbox.set_mm(10.0)
        self._spinbox.value_changed.connect(self._emit_preview)
        dist_layout.addWidget(self._spinbox)
        root.addWidget(self._dist_widget)

        # Vertex target input
        self._vertex_widget = QWidget()
        vtx_layout = QVBoxLayout(self._vertex_widget)
        vtx_layout.setContentsMargins(0, 0, 0, 0)
        vtx_layout.setSpacing(4)
        vtx_row = QHBoxLayout()
        vtx_row.setSpacing(6)
        self._vtx_label = QLabel("No vertex selected")
        self._vtx_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._vtx_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        vtx_row.addWidget(self._vtx_label)
        self._pick_vtx_btn = QPushButton("Pick")
        self._pick_vtx_btn.setObjectName("pick_vertex")
        self._pick_vtx_btn.setCheckable(True)
        self._pick_vtx_btn.clicked.connect(self._on_pick_vertex_toggle)
        vtx_row.addWidget(self._pick_vtx_btn)
        reset_vtx_btn = QPushButton("✕")
        reset_vtx_btn.setFixedWidth(prefs.scaled_px(24))
        reset_vtx_btn.clicked.connect(self._reset_vertex)
        vtx_row.addWidget(reset_vtx_btn)
        vtx_layout.addLayout(vtx_row)
        self._vtx_dist_label = QLabel("—")
        self._vtx_dist_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #666; font-size: 11px; font-family: monospace;"))
        vtx_layout.addWidget(self._vtx_dist_label)
        root.addWidget(self._vertex_widget)
        self._vertex_widget.hide()

        # ── Offsets ───────────────────────────────────────────────────
        root.addWidget(self._separator())
        root.addWidget(self._section_label("Offsets"))
        offset_grid = QHBoxLayout()
        offset_grid.setSpacing(8)
        start_col = QVBoxLayout()
        start_col.setSpacing(2)
        start_col.addWidget(QLabel("Start"))
        self._start_offset = ExprSpinBox(unit=prefs.default_unit)
        self._start_offset.set_mm(0.0)
        self._start_offset.value_changed.connect(self._emit_preview)
        start_col.addWidget(self._start_offset)
        offset_grid.addLayout(start_col)
        end_col = QVBoxLayout()
        end_col.setSpacing(2)
        end_col.addWidget(QLabel("End"))
        self._end_offset = ExprSpinBox(unit=prefs.default_unit)
        self._end_offset.set_mm(0.0)
        self._end_offset.value_changed.connect(self._emit_preview)
        end_col.addWidget(self._end_offset)
        offset_grid.addLayout(end_col)
        root.addLayout(offset_grid)

        # ── Direction ─────────────────────────────────────────────────
        root.addWidget(self._separator())
        root.addWidget(self._section_label("Direction"))
        dir_row = QHBoxLayout()
        dir_row.setSpacing(6)
        self._dir_label = QLabel("Normal")
        self._dir_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._dir_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        dir_row.addWidget(self._dir_label)
        self._flip_btn = QPushButton("⇅")
        self._flip_btn.setObjectName("pick_edge")
        self._flip_btn.setToolTip("Flip direction")
        self._flip_btn.setCheckable(True)
        self._flip_btn.clicked.connect(self._on_flip)
        dir_row.addWidget(self._flip_btn)
        self._pick_btn = QPushButton("Pick Edge")
        self._pick_btn.setObjectName("pick_edge")
        self._pick_btn.setCheckable(True)
        self._pick_btn.clicked.connect(self._on_pick_edge_toggle)
        dir_row.addWidget(self._pick_btn)
        reset_dir_btn = QPushButton("✕")
        reset_dir_btn.setFixedWidth(prefs.scaled_px(24))
        reset_dir_btn.clicked.connect(self._reset_direction)
        dir_row.addWidget(reset_dir_btn)
        root.addLayout(dir_row)

        # ── Operation (extrude only: New Body / Merge with) ───────────
        root.addWidget(self._separator())
        self._op_section_label = self._section_label("Operation")
        root.addWidget(self._op_section_label)

        # Extrude operation widget
        self._extrude_op_widget = QWidget()
        extrude_op_layout = QVBoxLayout(self._extrude_op_widget)
        extrude_op_layout.setContentsMargins(0, 0, 0, 0)
        extrude_op_layout.setSpacing(6)

        self._radio_new   = QRadioButton("New Body")
        self._radio_merge = QRadioButton("Merge with")
        self._radio_new.setChecked(True)
        self._op_group = QButtonGroup(self)
        self._op_group.addButton(self._radio_new,   0)
        self._op_group.addButton(self._radio_merge, 1)
        self._op_group.idClicked.connect(self._on_op_changed)
        extrude_op_layout.addWidget(self._radio_new)

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
        extrude_op_layout.addLayout(merge_row)
        root.addWidget(self._extrude_op_widget)

        # Cut operation widget — just "Cut from: <pick body>"
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

    def set_self_cut(self, is_self: bool):
        """Mark this panel as a same-body cut (no cross-body pick required)."""
        self._self_cut = is_self

    @property
    def _has_face(self) -> bool:
        return self._face_list.has_valid_entries

    def add_face_entry(self, body_id: str | None, face_idx: int | None,
                       label: str, valid: bool = True):
        key = (body_id, face_idx) if body_id is not None else label
        self._face_list.add(key, label, valid=valid)

    def remove_face_entry(self, index: int):
        self._face_list.remove_at(index)   # emits entry_removed → face_entry_removed

    def clear_face_entries(self):
        self._face_list.clear()

    def set_face_entry_error(self, index: int, error_msg: str):
        self._face_list.set_error(index, error_msg)

    def clear_face_errors(self):
        self._face_list.clear_errors()

    def set_face_origin(self, origin: np.ndarray):
        self._face_origin = origin.copy()

    def set_face_normal(self, normal: np.ndarray):
        self._face_normal = normal / np.linalg.norm(normal)
        self._recompute_vertex_dist()
        self._emit_preview()

    def set_direction(self, vec: np.ndarray):
        self._direction = vec / np.linalg.norm(vec)
        x, y, z = self._direction
        self._dir_label.setText(f"{x:+.2f}, {y:+.2f}, {z:+.2f}")
        self._dir_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #8dcc5a; font-size: 11px; font-family: monospace;"))
        self._end_pick_edge()
        self._recompute_vertex_dist()
        self._emit_preview()

    def set_vertex_target(self, vertex: np.ndarray):
        self._target_vertex = vertex.copy()
        x, y, z = vertex
        self._vtx_label.setText(f"{x:.2f}, {y:.2f}, {z:.2f}")
        self._vtx_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #cc8dff; font-size: 11px; font-family: monospace;"))
        self._end_pick_vertex()
        self._recompute_vertex_dist()
        self._emit_preview()

    def set_merge_body(self, body_id: str, body_name: str):
        """Called by viewport when the user picks a body for merge/cut."""
        self._merge_body_id = body_id
        self._self_cut = False  # explicit body pick means cross-body
        self._end_pick_body()
        # Update both the merge and cut body labels
        self._body_label.setText(body_name)
        self._body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #7ab3d4; font-size: 11px; font-family: monospace;"))
        self._cut_body_label.setText(body_name)
        self._cut_body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #7ab3d4; font-size: 11px; font-family: monospace;"))
        self._emit_preview()

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def _recompute_vertex_dist(self):
        """
        Compute the signed distance from the face plane to the target vertex,
        measured along the extrude direction (face normal if no custom direction).

        Always uses the unflipped normal/direction — flip is applied once in
        _dist_and_dir so it never gets double-counted.
        """
        if self._target_vertex is None or self._face_origin is None:
            self._vertex_dist_mm = None
            self._vtx_dist_label.setText("—")
            return
        if self._face_normal is None:
            self._vertex_dist_mm = None
            self._vtx_dist_label.setText("no normal")
            return

        offset = self._target_vertex - self._face_origin
        # depth along the unflipped face normal
        depth = float(np.dot(offset, self._face_normal))

        if self._direction is not None:
            # Project onto custom direction: how far along _direction does the
            # extrusion need to travel so its end plane passes through the vertex?
            cos_angle = float(np.dot(self._direction, self._face_normal))
            if abs(cos_angle) < 1e-6:
                self._vertex_dist_mm = None
                self._vtx_dist_label.setText("⊥ unreachable")
                return
            dist = depth / cos_angle
        else:
            dist = depth

        # Reject near-zero distances — vertex is essentially on the face plane.
        # A sub-0.001mm cut produces a degenerate tool solid that makes OCCT unstable.
        if abs(dist) < 0.001:
            self._vertex_dist_mm = None
            self._vtx_dist_label.setText("⚠ vertex on face plane")
            return

        # Store signed — _dist_and_dir will apply flip and take abs
        self._vertex_dist_mm = dist
        from cad.units import format_value
        from cad.prefs import prefs
        self._vtx_dist_label.setText(f"→ {format_value(abs(dist), prefs.default_unit)}")

    def _dist_and_dir(self):
        """Return (abs_distance_mm, direction_or_None). Distance is always positive."""
        if self._target_group.checkedId() == 1:
            if self._vertex_dist_mm is None:
                return None, None
            # _vertex_dist_mm is signed: negative means vertex is behind the face
            # normal.  Absorb the sign into a direction flip so callers always
            # receive a positive distance.
            vtx_signed = self._vertex_dist_mm
            vtx_flip   = vtx_signed < 0
            dist       = abs(vtx_signed)
        else:
            vtx_flip = False
            dist = self._spinbox.mm_value()
            if dist is None:
                return None, None
            dist = abs(dist)

        flip = self._flip ^ vtx_flip   # XOR: user flip + vertex-direction flip

        if self._direction is not None:
            direction = -self._direction if flip else self._direction
        elif flip and self._face_normal is not None:
            direction = -self._face_normal
        elif self._face_normal is not None:
            direction = self._face_normal
        else:
            direction = None

        return abs(dist), direction

    def _signed_dist_and_dir(self):
        """Return (signed_dist_mm, direction). Negative = cut."""
        dist, direction = self._dist_and_dir()
        if dist is None:
            return None, None
        is_cut = self._mode_group.checkedId() == 1
        return (-abs(dist) if is_cut else abs(dist)), direction

    def _emit_preview(self, _=None):
        dist, direction = self._signed_dist_and_dir()
        if dist is None:
            return
        self.preview_changed.emit(dist, direction)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_mode_changed(self, btn_id: int):
        is_cut = btn_id == 1
        self._extrude_op_widget.setVisible(not is_cut)
        self._cut_op_widget.setVisible(is_cut)
        # Reset body pick when switching modes
        self._merge_body_id = None
        self._body_label.setText("—")
        self._body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._cut_body_label.setText("—")
        self._cut_body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._emit_preview()

    def _on_target_mode_changed(self, btn_id: int):
        if btn_id == 0:
            self._dist_widget.show()
            self._vertex_widget.hide()
            self._end_pick_vertex()
        else:
            self._dist_widget.hide()
            self._vertex_widget.show()
        self._emit_preview()

    def _on_flip(self, checked: bool):
        self._flip = checked
        self._flip_btn.setText("⇅" if not checked else "⇵")
        self._recompute_vertex_dist()
        self._emit_preview()

    def _on_op_changed(self, btn_id: int):
        # Enable pick body only when "Merge with" is selected
        self._pick_body_btn.setEnabled(btn_id == 1)
        if btn_id == 0:
            self._merge_body_id = None
            self._body_label.setText("—")
            self._body_label.setStyleSheet(prefs.scale_stylesheet(
                "color: #888; font-size: 11px; font-family: monospace;"))
            self._end_pick_body()

    def _on_pick_edge_toggle(self, checked: bool):
        if checked:
            self._picking_edge = True
            self._pick_btn.setProperty("active", True)
            self._pick_btn.style().unpolish(self._pick_btn)
            self._pick_btn.style().polish(self._pick_btn)
            self.picking_edge_changed.emit(True)
        else:
            self._end_pick_edge()

    def _end_pick_edge(self):
        self._picking_edge = False
        self._pick_btn.setChecked(False)
        self._pick_btn.setProperty("active", False)
        self._pick_btn.style().unpolish(self._pick_btn)
        self._pick_btn.style().polish(self._pick_btn)
        self.picking_edge_changed.emit(False)

    def _on_pick_vertex_toggle(self, checked: bool):
        if checked:
            self._picking_vertex = True
            self._pick_vtx_btn.setProperty("active", True)
            self._pick_vtx_btn.style().unpolish(self._pick_vtx_btn)
            self._pick_vtx_btn.style().polish(self._pick_vtx_btn)
            self.picking_vertex_changed.emit(True)
        else:
            self._end_pick_vertex()

    def _end_pick_vertex(self):
        self._picking_vertex = False
        self._pick_vtx_btn.setChecked(False)
        self._pick_vtx_btn.setProperty("active", False)
        self._pick_vtx_btn.style().unpolish(self._pick_vtx_btn)
        self._pick_vtx_btn.style().polish(self._pick_vtx_btn)
        self.picking_vertex_changed.emit(False)

    def _on_pick_body_toggle(self, checked: bool):
        if checked:
            self._picking_body = True
            btn = (self._pick_cut_body_btn
                   if self._mode_group.checkedId() == 1
                   else self._pick_body_btn)
            btn.setProperty("active", True)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            self.picking_body_changed.emit(True)
        else:
            self._end_pick_body()

    def _end_pick_body(self):
        self._picking_body = False
        for btn in (self._pick_body_btn, self._pick_cut_body_btn):
            btn.setChecked(False)
            btn.setProperty("active", False)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.picking_body_changed.emit(False)

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

    def _reset_vertex(self):
        self._target_vertex  = None
        self._vertex_dist_mm = None
        self._vtx_label.setText("No vertex selected")
        self._vtx_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._vtx_dist_label.setText("—")
        self._end_pick_vertex()
        self._emit_preview()

    def _reset_direction(self):
        self._direction = None
        self._dir_label.setText("Normal")
        self._dir_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._end_pick_edge()
        self._recompute_vertex_dist()
        self._emit_preview()

    def _on_ok(self):
        if not len(self._face_list):
            return
        self._spinbox._on_commit()
        dist, direction = self._signed_dist_and_dir()
        if dist is None:
            return

        is_cut = self._mode_group.checkedId() == 1
        if is_cut:
            # Cross-body cut requires a picked target; self-cuts don't.
            if self._merge_body_id is None and not self._self_cut:
                return  # no target body picked yet
            merge_body_id = self._merge_body_id  # None for self-cuts
        else:
            op_id = self._op_group.checkedId()
            if op_id == 1:
                # Merge with picked body
                if self._merge_body_id is None:
                    return
                merge_body_id = self._merge_body_id
            else:
                merge_body_id = "__new_body__"

        self.extrude_requested.emit(dist, direction, merge_body_id)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def closeEvent(self, e):
        self.cancelled.emit()
        super().closeEvent(e)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            e.accept()   # consume — default button handles it, don't double-fire
        elif e.key() == Qt.Key.Key_Escape:
            if self._picking_face:
                self._end_pick_face()
            elif self._picking_vertex:
                self._end_pick_vertex()
            elif self._picking_edge:
                self._end_pick_edge()
            elif self._picking_body:
                self._end_pick_body()
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
