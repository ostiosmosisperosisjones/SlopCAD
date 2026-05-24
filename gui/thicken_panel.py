"""
gui/thicken_panel.py

ThickenPanel — floating non-modal panel for the thicken operation.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QFrame,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from gui.fillet3d_panel import _adaptive_delay
from PyQt6.QtGui import QKeyEvent

from gui.expr_spinbox import ExprSpinBox
from gui.selection_list import SelectionList
from cad.prefs import prefs


_PANEL_STYLE = """
QWidget#ThickenPanel {
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


class ThickenPanel(QWidget):
    thicken_requested  = pyqtSignal(float)  # thickness_mm (signed)
    cancelled          = pyqtSignal()
    preview_changed    = pyqtSignal(float)  # thickness_mm for live preview
    face_entry_removed = pyqtSignal(int)    # index of removed face entry (forwarded from SelectionList)

    def __init__(self, parent=None):
        # Parent to the top-level main window so this Tool window rides its
        # z-order (raises when the main window is activated).
        super().__init__(parent.window() if parent is not None else None)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Thicken")
        self._viewport = parent
        self.setObjectName("ThickenPanel")
        self.setStyleSheet(prefs.scale_stylesheet(_PANEL_STYLE))
        self.setMinimumWidth(prefs.scaled_px(240))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._picking_face : bool = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 10, 12, 10)

        title = QLabel("Thicken / Cut")
        title.setObjectName("title")
        layout.addWidget(title)

        layout.addWidget(self._separator())

        # ── Faces ──────────────────────────────────────────────────────
        face_header = QHBoxLayout()
        face_header.setSpacing(6)
        face_header.addWidget(self._section_label("Faces"))
        face_header.addStretch()
        self._pick_face_btn = QPushButton("+ Add Face")
        self._pick_face_btn.setObjectName("pick_face")
        self._pick_face_btn.setCheckable(True)
        self._pick_face_btn.clicked.connect(self._on_pick_face_toggle)
        face_header.addWidget(self._pick_face_btn)
        layout.addLayout(face_header)

        self._face_list = SelectionList(empty_text="No faces selected")
        self._face_list.entry_removed.connect(self.face_entry_removed)
        layout.addWidget(self._face_list)

        layout.addWidget(self._separator())

        # ── Mode ───────────────────────────────────────────────────────
        mode_lbl = self._section_label("Mode")
        layout.addWidget(mode_lbl)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self._radio_thicken = QRadioButton("Thicken")
        self._radio_cut     = QRadioButton("Cut")
        self._radio_thicken.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._radio_thicken, 0)
        self._mode_group.addButton(self._radio_cut,     1)
        self._mode_group.idClicked.connect(self._schedule_preview)
        mode_row.addWidget(self._radio_thicken)
        mode_row.addWidget(self._radio_cut)
        layout.addLayout(mode_row)

        layout.addWidget(self._separator())

        # ── Thickness ──────────────────────────────────────────────────
        thick_lbl = self._section_label("Thickness")
        layout.addWidget(thick_lbl)

        self._spinbox = ExprSpinBox(unit=prefs.default_unit)
        self._spinbox.set_mm(1.0)
        layout.addWidget(self._spinbox)

        layout.addWidget(self._separator())

        # ── Buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancelled)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("ok")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self._preview_timer      = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._fire_preview)
        self._preview_last_value = None
        self._preview_last_ms    = 0

        self._spinbox.value_changed.connect(self._schedule_preview)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    # Face list API (delegated to SelectionList)
    # ------------------------------------------------------------------

    @property
    def _has_face(self) -> bool:
        return self._face_list.has_valid_entries

    def add_face_entry(self, body_id: str, face_idx: int, label: str, valid: bool = True):
        self._face_list.add((body_id, face_idx), label, valid=valid)

    def remove_face_entry(self, index: int):
        self._face_list.remove_at(index)   # emits entry_removed → face_entry_removed

    def clear_face_entries(self):
        self._face_list.clear()

    def set_face_entry_error(self, index: int, error_msg: str):
        self._face_list.set_error(index, error_msg)

    def clear_face_errors(self):
        self._face_list.clear_errors()

    # ------------------------------------------------------------------
    # Pick face toggle
    # ------------------------------------------------------------------

    def _on_pick_face_toggle(self, checked: bool):
        if checked:
            self._picking_face = True
            self._pick_face_btn.setProperty("active", True)
            self._pick_face_btn.style().unpolish(self._pick_face_btn)
            self._pick_face_btn.style().polish(self._pick_face_btn)
        else:
            self._end_pick_face()

    def end_pick_face(self):
        self._end_pick_face()

    def _end_pick_face(self):
        self._picking_face = False
        self._pick_face_btn.setChecked(False)
        self._pick_face_btn.setProperty("active", False)
        self._pick_face_btn.style().unpolish(self._pick_face_btn)
        self._pick_face_btn.style().polish(self._pick_face_btn)

    # ------------------------------------------------------------------
    # Value helpers
    # ------------------------------------------------------------------

    def _signed_value(self) -> float | None:
        val = self._spinbox.mm_value()
        if val is None:
            return None
        val = abs(val)
        if self._mode_group.checkedId() == 1:  # Cut
            val = -val
        return val

    def _on_ok(self):
        if not len(self._face_list):
            return
        val = self._signed_value()
        if val is not None and val != 0:
            self.thicken_requested.emit(val)

    def _schedule_preview(self, _val=None):
        self._preview_timer.start(_adaptive_delay(self._signed_value(), self))

    def _fire_preview(self):
        val = self._signed_value()
        if val is not None:
            self.preview_changed.emit(val)

    def _emit_preview(self, _val=None):
        self._preview_timer.start(_adaptive_delay(self._signed_value(), self))

    def set_thickness(self, thickness_mm: float):
        """Restore panel state from a saved thickness (signed)."""
        if thickness_mm < 0:
            self._radio_cut.setChecked(True)
        else:
            self._radio_thicken.setChecked(True)
        self._spinbox.set_mm(abs(thickness_mm))

    def closeEvent(self, e):
        self.cancelled.emit()
        super().closeEvent(e)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._on_ok()
        elif e.key() == Qt.Key.Key_Escape:
            if self._picking_face:
                self._end_pick_face()
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
