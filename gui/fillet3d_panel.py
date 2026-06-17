"""
gui/fillet3d_panel.py

Fillet3DPanel — floating panel for 3D edge fillet operations.
Modelled after ExtrudePanel / ThickenPanel.
"""

from __future__ import annotations
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer, QElapsedTimer
from PyQt6.QtGui import QKeyEvent

from gui.selection_list import SelectionList


_PANEL_STYLE = """
QWidget#Fillet3DPanel {
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
"""

_SEP_STYLE = "background: #333;"

_FAST_MS = 16
_SLOW_MS = 150
_CLOCK   = QElapsedTimer()
_CLOCK.start()


def _adaptive_delay(value, panel) -> int:
    """Return a debounce delay in ms that shrinks with faster value changes.

    Maps velocity (units/ms) to delay via a smooth hyperbolic curve:
      stopped          → 150ms
      0.05 units/ms    →  ~54ms
      0.2  units/ms    →  ~28ms  (typical fast drag)
      1.0+ units/ms    →  ~18ms  (floor)
    """
    now = _CLOCK.elapsed()
    dt  = now - panel._preview_last_ms
    if dt > 0 and panel._preview_last_value is not None and value is not None:
        velocity = abs(value - panel._preview_last_value) / dt  # units per ms
        delay = int(_FAST_MS + (_SLOW_MS - _FAST_MS) / (1.0 + 0.5 * velocity * 100))
        delay = max(_FAST_MS, min(_SLOW_MS, delay))
    else:
        delay = _SLOW_MS
    panel._preview_last_value = value
    panel._preview_last_ms    = now
    return delay


class Fillet3DPanel(QWidget):
    # confirmed/preview emit (default_radius, edge_radii) where edge_radii is a
    # list aligned with the edge rows — each entry a float or None (use default).
    confirmed            = pyqtSignal(float, list)
    cancelled            = pyqtSignal()
    preview_changed      = pyqtSignal(float, list)
    fit_requested        = pyqtSignal()        # user clicked "Fit largest"
    face_entry_removed   = pyqtSignal(int)
    edge_entry_removed   = pyqtSignal(int)
    picking_face_changed = pyqtSignal(bool)
    picking_edge_changed = pyqtSignal(bool)

    def __init__(self, workspace, parent=None):
        # Parent to the top-level main window so this Tool window rides its
        # z-order (raises when the main window is activated).
        super().__init__(parent.window() if parent is not None else None)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Fillet")
        self._viewport = parent
        self.setObjectName("Fillet3DPanel")
        self.setStyleSheet(prefs.scale_stylesheet(_PANEL_STYLE))
        self.setMinimumWidth(prefs.scaled_px(240))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._workspace     = workspace
        self._picking_face  = False
        self._picking_edge  = False

        self._preview_timer      = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._fire_preview)
        self._preview_last_value = None
        self._preview_last_ms    = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        from gui.expr_spinbox import ExprSpinBox
        from cad.prefs import prefs

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Fillet")
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

        # Per-edge radius column: each edge row carries its own spinbox.
        # A blank row falls back to the default radius below.
        self._edge_list = SelectionList(empty_text="No edges selected",
                                        per_row_value=True)
        self._edge_list.entry_removed.connect(self.edge_entry_removed)
        self._edge_list.value_changed.connect(self._on_row_radius_changed)
        root.addWidget(self._edge_list)

        root.addWidget(self._separator())

        # ── Default radius ─────────────────────────────────────────────
        # Applies to faces and to any edge row left blank.
        root.addWidget(self._section_label("Default Radius"))
        self._spinbox = ExprSpinBox(unit=prefs.default_unit)
        self._spinbox.set_mm(1.0)
        self._spinbox.value_changed.connect(self._on_radius_changed)
        root.addWidget(self._spinbox)

        # "Fit largest" is always available: it binary-searches for the biggest
        # buildable radius on demand. The note beside it only appears when the
        # current radius can't be built. Auto-fit is too heavy for live preview,
        # so it lives behind this explicit button.
        note_row = QHBoxLayout()
        note_row.setSpacing(6)
        self._applied_note = QLabel("")
        self._applied_note.setStyleSheet("color: #cc9a4a; font-size: 11px;")
        self._applied_note.setWordWrap(True)
        note_row.addWidget(self._applied_note, 1)
        self._fit_btn = QPushButton("Fit largest")
        self._fit_btn.clicked.connect(lambda: self.fit_requested.emit())
        note_row.addWidget(self._fit_btn)
        root.addLayout(note_row)
        self._applied_note.setVisible(False)

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

    def set_radius(self, radius: float):
        self._spinbox.set_mm(radius)

    def add_face_entry(self, body_id: str | None, face_idx: int | None, label: str):
        self._face_list.add((body_id, face_idx), label)

    def remove_face_entry(self, index: int):
        self._face_list.remove_at(index)

    def clear_face_entries(self):
        self._face_list.clear()

    def add_edge_entry(self, body_id: str | None, edge_idx: int | None, label: str,
                       radius: float | None = None):
        self._edge_list.add((body_id, edge_idx), label, value=radius)

    def remove_edge_entry(self, index: int):
        self._edge_list.remove_at(index)

    def clear_edge_entries(self):
        self._edge_list.clear()

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

    def _edge_radii(self) -> list:
        """Per-edge radii aligned with the edge rows: each a float override or
        None (use the default radius)."""
        return list(self._edge_list.values)

    def _fire_preview(self):
        v = self._spinbox.mm_value()
        if v is not None and v > 0:
            self.preview_changed.emit(v, self._edge_radii())

    def _emit_preview(self):
        self._preview_timer.start(_adaptive_delay(
            self._spinbox.mm_value(),
            self))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_radius_changed(self, _):
        # Clear any stale "not buildable" hint the moment the user changes the
        # value — it described the *previous* radius. The next preview result
        # re-shows it only if the new radius also isn't buildable. ("Fit largest"
        # stays available regardless.)
        self._applied_note.setVisible(False)
        self._emit_preview()

    def _on_row_radius_changed(self, _index, _value):
        self._applied_note.setVisible(False)
        self._emit_preview()

    def _on_pick_face_toggle(self, checked: bool):
        if checked:
            self.end_pick_edge()   # mutually exclusive
            self._picking_face = True
            self._pick_face_btn.setProperty("active", True)
            self._pick_face_btn.style().unpolish(self._pick_face_btn)
            self._pick_face_btn.style().polish(self._pick_face_btn)
            self.picking_face_changed.emit(True)
        else:
            self.end_pick_face()

    def _on_pick_edge_toggle(self, checked: bool):
        if checked:
            self.end_pick_face()   # mutually exclusive
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
        v = self._spinbox.mm_value()
        if v is not None and v > 0:
            self.confirmed.emit(v, self._edge_radii())

    def set_buildable(self, buildable: bool):
        """Called after each preview: show the 'not buildable' hint when the
        current radius can't be built. ('Fit largest' is always available.)"""
        self._applied_note.setText(
            "" if buildable else "⚠ not buildable at this radius")
        self._applied_note.setVisible(not buildable)

    def set_max_hint(self, radius_mm: float | None):
        """Apply the largest buildable radius: set the radius field to it (as if
        the user typed it) and confirm in the active unit. radius_mm is mm, or
        None when nothing in range builds reliably.

        Writing it into the field (rather than only hinting) means the value is
        always recoverable — clicking Fit largest again after changing your mind
        re-applies it — and it re-runs the normal preview at that radius."""
        from cad.units import format_value
        unit = self._spinbox._unit
        if radius_mm is None or radius_mm <= 0:
            self._applied_note.setText("⚠ no buildable radius found for this edge")
            self._applied_note.setVisible(True)
            return
        # set_mm displays in the active unit but does NOT emit value_changed
        # (that's user-edit only), so fire the preview explicitly at the fitted
        # radius. Set the note after, so it isn't cleared by _on_radius_changed.
        self._spinbox.set_mm(radius_mm)
        self._fire_preview()
        self._applied_note.setText(
            f"fit to largest buildable ≈ {format_value(radius_mm, unit)}")
        self._applied_note.setVisible(True)

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
