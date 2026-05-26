"""
gui/offset_plane_panel.py

OffsetPlanePanel — floating non-modal panel for creating a datum plane that
is offset from a world plane or a body face.
"""

from __future__ import annotations
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QFrame, QSizePolicy, QComboBox, QLineEdit,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QKeyEvent


_PANEL_STYLE = """
QWidget#OffsetPlanePanel {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
}
QLabel { color: #d4d4d4; font-size: 12px; }
QLabel#title    { color: #ffffff; font-size: 13px; font-weight: bold; }
QLabel#section  { color: #888; font-size: 10px;
                  text-transform: uppercase; letter-spacing: 1px; }
QPushButton {
    background: #2a2a2a; color: #d4d4d4;
    border: 1px solid #444; border-radius: 3px;
    padding: 4px 10px; font-size: 12px;
}
QPushButton:hover  { background: #333; }
QPushButton:pressed { background: #222; }
QPushButton#ok {
    background: #1a4a7a; border-color: #4a90d9; color: #fff;
    padding: 5px 18px; font-weight: bold;
}
QPushButton#ok:hover { background: #1e5a8a; }
QPushButton#pick_face[active=true] {
    background: #2a1e1e; border-color: #d96a4a; color: #ff9977;
}
QRadioButton {
    color: #d4d4d4; font-size: 12px; spacing: 6px;
}
QRadioButton::indicator {
    width: 13px; height: 13px; border-radius: 7px;
    border: 1px solid #555; background: #2a2a2a;
}
QRadioButton::indicator:checked {
    background: #4a90d9; border-color: #4a90d9;
}
QComboBox {
    background: #2a2a2a; color: #d4d4d4;
    border: 1px solid #444; border-radius: 3px;
    padding: 2px 6px; font-size: 12px;
}
QLineEdit {
    background: #2a2a2a; color: #d4d4d4;
    border: 1px solid #444; border-radius: 3px;
    padding: 2px 6px; font-size: 12px;
}
QLineEdit:focus { border-color: #4a90d9; }
"""

_SEP_STYLE = "background: #333;"


class OffsetPlanePanel(QWidget):
    # (plane_source_dict, name)  — emitting a dict keeps this UI layer free
    # of cad.* imports.  Viewport mixin reconstructs the SketchPlaneSource.
    confirmed              = pyqtSignal(object, str)
    cancelled              = pyqtSignal()
    picking_face_changed   = pyqtSignal(bool)
    preview_changed        = pyqtSignal()

    def __init__(self, workspace, parent=None):
        super().__init__(parent.window() if parent is not None else None)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Offset Plane")
        self._viewport  = parent
        self._workspace = workspace
        self.setObjectName("OffsetPlanePanel")
        self.setStyleSheet(prefs.scale_stylesheet(_PANEL_STYLE))
        self.setMinimumWidth(prefs.scaled_px(260))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._picking_face: bool = False
        # When parent_mode == "face", these hold the picked target.
        self._parent_body_id:  str | None = None
        self._parent_face_idx: int | None = None
        # Set when the user picked a sketch face: a fully-formed
        # SketchPlaneSource taken straight from the sketch entry.
        self._parent_sketch_source = None
        self._parent_sketch_label:  str = ""
        # When reopened from an existing op without the user re-picking a
        # parent, this holds the loaded source so we don't lose it.
        self._loaded_parent_source = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        from gui.expr_spinbox import ExprSpinBox

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Offset Plane")
        title.setObjectName("title")
        root.addWidget(title)

        root.addWidget(self._separator())

        # ── Parent ────────────────────────────────────────────────────
        root.addWidget(self._section_label("Parent"))
        parent_row = QHBoxLayout()
        parent_row.setSpacing(6)
        self._radio_world = QRadioButton("World")
        self._radio_face  = QRadioButton("Face")
        self._radio_world.setChecked(True)
        self._parent_group = QButtonGroup(self)
        self._parent_group.addButton(self._radio_world, 0)
        self._parent_group.addButton(self._radio_face,  1)
        self._parent_group.idClicked.connect(self._on_parent_mode_changed)
        parent_row.addWidget(self._radio_world)
        parent_row.addWidget(self._radio_face)
        parent_row.addStretch()
        root.addLayout(parent_row)

        # World mode: axis dropdown
        self._world_widget = QWidget()
        world_layout = QHBoxLayout(self._world_widget)
        world_layout.setContentsMargins(0, 0, 0, 0)
        world_layout.setSpacing(6)
        world_layout.addWidget(QLabel("Plane"))
        self._world_axis = QComboBox()
        self._world_axis.addItems(["XY", "XZ", "YZ"])
        self._world_axis.currentIndexChanged.connect(self._emit_preview)
        world_layout.addWidget(self._world_axis, 1)
        root.addWidget(self._world_widget)

        # Face mode: pick button + status label
        self._face_widget = QWidget()
        face_layout = QHBoxLayout(self._face_widget)
        face_layout.setContentsMargins(0, 0, 0, 0)
        face_layout.setSpacing(6)
        self._face_label = QLabel("No face or sketch picked")
        self._face_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._face_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        face_layout.addWidget(self._face_label)
        self._pick_face_btn = QPushButton("Pick Face")
        self._pick_face_btn.setObjectName("pick_face")
        self._pick_face_btn.setCheckable(True)
        self._pick_face_btn.clicked.connect(self._on_pick_face_toggle)
        face_layout.addWidget(self._pick_face_btn)
        root.addWidget(self._face_widget)
        self._face_widget.hide()

        root.addWidget(self._separator())

        # ── Distance ──────────────────────────────────────────────────
        root.addWidget(self._section_label("Offset"))
        dist_row = QHBoxLayout()
        dist_row.setSpacing(6)
        self._spinbox = ExprSpinBox(unit=prefs.default_unit)
        self._spinbox.set_mm(10.0)
        self._spinbox.value_changed.connect(self._emit_preview)
        dist_row.addWidget(self._spinbox, 1)
        self._flip_btn = QPushButton("⇅")
        self._flip_btn.setToolTip("Flip offset direction")
        self._flip_btn.setCheckable(True)
        self._flip_btn.clicked.connect(self._on_flip)
        dist_row.addWidget(self._flip_btn)
        root.addLayout(dist_row)

        root.addWidget(self._separator())

        # ── Name ──────────────────────────────────────────────────────
        root.addWidget(self._section_label("Name"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Plane (auto)")
        root.addWidget(self._name_edit)

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
    # Public API used by the viewport pick router
    # ------------------------------------------------------------------

    def set_picked_face(self, body_id: str, face_idx: int, body_name: str):
        self._parent_body_id       = body_id
        self._parent_face_idx      = face_idx
        self._parent_sketch_source = None
        self._parent_sketch_label  = ""
        self._loaded_parent_source = None  # user picked fresh; drop the snapshot
        self._face_label.setText(f"{body_name}  ·  face {face_idx}")
        self._face_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #ff9977; font-size: 11px; font-family: monospace;"))
        self._end_pick_face()
        self._emit_preview()

    def set_picked_sketch_plane(self, plane_source, label: str):
        """Pick a sketch's plane as the parent (skips face_ref machinery)."""
        self._parent_body_id       = None
        self._parent_face_idx      = None
        self._parent_sketch_source = plane_source
        self._parent_sketch_label  = label
        self._loaded_parent_source = None  # user picked fresh; drop the snapshot
        self._face_label.setText(label)
        self._face_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #4a90d9; font-size: 11px; font-family: monospace;"))
        self._end_pick_face()
        self._emit_preview()

    # ------------------------------------------------------------------
    # State accessors used by the viewport when building the op
    # ------------------------------------------------------------------

    def build_plane_source_dict(self) -> dict | None:
        """Return a serialized SketchPlaneSource dict, or None if incomplete."""
        sign = -1.0 if self._flip_btn.isChecked() else 1.0
        dist = self._spinbox.mm_value()
        if dist is None:
            return None
        dist = float(dist) * sign
        if self._parent_group.checkedId() == 0:
            axis = self._world_axis.currentText()
            return {
                "type": "offset", "distance": dist,
                "parent": {"type": "world", "axis": axis},
            }
        if self._parent_sketch_source is not None:
            # Sketch-parent path: carry the live source through so the mixin
            # can wrap it in an OffsetPlaneSource directly.
            return {
                "type": "offset", "distance": dist,
                "parent": {"type": "sketch_live"},
            }
        if self._parent_body_id is not None and self._parent_face_idx is not None:
            # face_ref is built later by the viewport (needs the live face object).
            return {
                "type": "offset", "distance": dist,
                "parent": {"type": "face_pending",
                            "body_id":  self._parent_body_id,
                            "face_idx": self._parent_face_idx},
            }
        if self._loaded_parent_source is not None:
            # Reopened edit, user didn't re-pick the parent — reuse the
            # original source object.
            return {
                "type": "offset", "distance": dist,
                "parent": {"type": "loaded_source"},
            }
        return None

    def loaded_parent_source(self):
        """The original parent SketchPlaneSource carried over from a reopen."""
        return self._loaded_parent_source

    def parent_sketch_source(self):
        """The live SketchPlaneSource picked from a sketch (or None)."""
        return self._parent_sketch_source

    def load_from_op(self, op, parent_label: str | None = None):
        """Restore panel widgets to match an existing OffsetPlaneOp.

        parent_label: optional human label to show in Face mode.  For face-parent
        we don't have body_id/face_idx readily (the FaceRef abstracts those
        away), so callers pass an appropriate label like 'Top of Body 1'.  When
        omitted we infer something reasonable from the source type.
        """
        from cad.plane_ref import (WorldPlaneSource, FacePlaneSource,
                                    OffsetPlaneSource)
        # The op stores an OffsetPlaneSource wrapping a parent source.
        ps = op.plane_source
        if not isinstance(ps, OffsetPlaneSource):
            return  # unexpected shape; bail rather than misconfigure UI

        # Block preview emissions while we mutate widgets in bulk.
        self.blockSignals(True)
        try:
            self._name_edit.setText(op.name or "")

            # Distance + flip
            dist = float(ps.distance)
            self._flip_btn.setChecked(dist < 0)
            self._flip_btn.setText("⇵" if dist < 0 else "⇅")
            self._spinbox.set_mm(abs(dist))

            # Reset all parent-source state and stash the loaded source so
            # build_plane_source_dict can fall back to it if the user doesn't
            # re-pick.
            self._parent_body_id       = None
            self._parent_face_idx      = None
            self._parent_sketch_source = None
            self._parent_sketch_label  = ""
            self._loaded_parent_source = ps.parent

            parent = ps.parent
            if isinstance(parent, WorldPlaneSource):
                self._radio_world.setChecked(True)
                self._on_parent_mode_changed(0)
                idx = self._world_axis.findText(parent.axis)
                if idx >= 0:
                    self._world_axis.setCurrentIndex(idx)
                # World mode is fully reconstructible from the radio + combo,
                # so we don't need the loaded-source fallback for it.
                self._loaded_parent_source = None
            elif isinstance(parent, FacePlaneSource):
                self._radio_face.setChecked(True)
                self._on_parent_mode_changed(1)
                label = parent_label or f"Face on body {parent.body_id[:8]}"
                self._face_label.setText(label)
                self._face_label.setStyleSheet(prefs.scale_stylesheet(
                    "color: #ff9977; font-size: 11px; font-family: monospace;"))
            else:
                # Any other plane-source type (incl. an OffsetPlaneSource parent
                # for "offset of offset") — treat as a live sketch-style source.
                self._radio_face.setChecked(True)
                self._on_parent_mode_changed(1)
                self._face_label.setText(parent_label or "Parent plane")
                self._face_label.setStyleSheet(prefs.scale_stylesheet(
                    "color: #4a90d9; font-size: 11px; font-family: monospace;"))
        finally:
            self.blockSignals(False)
        self._emit_preview()

    def chosen_name(self) -> str:
        return self._name_edit.text().strip()

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_parent_mode_changed(self, btn_id: int):
        if btn_id == 0:
            self._world_widget.show()
            self._face_widget.hide()
            self._end_pick_face()
        else:
            self._world_widget.hide()
            self._face_widget.show()
        self._emit_preview()

    def _on_flip(self, _checked: bool):
        self._flip_btn.setText("⇵" if self._flip_btn.isChecked() else "⇅")
        self._emit_preview()

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

    def _emit_preview(self, *_):
        self.preview_changed.emit()

    def _on_ok(self):
        self._spinbox._on_commit()
        ps_dict = self.build_plane_source_dict()
        if ps_dict is None:
            return
        self.confirmed.emit(ps_dict, self.chosen_name())

    # ------------------------------------------------------------------
    # Keyboard
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
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
