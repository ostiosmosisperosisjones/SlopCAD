"""
gui/loft_panel.py

LoftPanel — floating non-modal panel for the loft / loft-cut operation.

Profile list is ordered; the user adds sketches via toggle-pick, reorders with
▲/▼, removes with ✕.  OK is gated on having ≥2 profiles and (in cut/merge
modes) a picked target body.
"""

from __future__ import annotations
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QButtonGroup, QRadioButton, QFrame, QSizePolicy,
    QCheckBox, QComboBox,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QKeyEvent

from gui.selection_list import SelectionList


_PANEL_STYLE = """
QWidget#LoftPanel {
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
QPushButton:disabled { color: #555; background: #1a1a1a; border-color: #2a2a2a; }
QPushButton#ok {
    background: #1a4a7a; border-color: #4a90d9; color: #fff;
    padding: 5px 18px; font-weight: bold;
}
QPushButton#ok:hover    { background: #1e5a8a; }
QPushButton#ok:disabled { background: #1a1a1a; color: #555; border-color: #2a2a2a; }
QPushButton#pick_profile[active=true] {
    background: #2a3a1e; border-color: #6aaa44; color: #8dcc5a;
}
QPushButton#pick_body[active=true] {
    background: #1e2a3a; border-color: #4a90d9; color: #7ab3d4;
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
QRadioButton:disabled { color: #555; }
QCheckBox {
    color: #d4d4d4; font-size: 12px; spacing: 6px;
}
QCheckBox::indicator {
    width: 13px; height: 13px; border-radius: 3px;
    border: 1px solid #555; background: #2a2a2a;
}
QCheckBox::indicator:checked {
    background: #4a90d9; border-color: #4a90d9;
}
QComboBox {
    background: #2a2a2a; color: #d4d4d4;
    border: 1px solid #444; border-radius: 3px;
    padding: 2px 6px; font-size: 12px;
}
"""

_SEP_STYLE = "background: #333;"


class LoftPanel(QWidget):
    loft_requested              = pyqtSignal(list, object, object, bool, str)
    # (profiles, mode_str, target_body_id|None, ruled, continuity)
    #   profiles is [(sketch_entry_id, face_idx_or_None), ...]
    cancelled                   = pyqtSignal()
    picking_profile_changed     = pyqtSignal(bool)
    picking_body_changed        = pyqtSignal(bool)
    profile_removed             = pyqtSignal(int)
    profile_order_changed       = pyqtSignal()
    preview_changed             = pyqtSignal()

    def __init__(self, workspace, parent=None):
        super().__init__(parent.window() if parent is not None else None)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Loft")
        self._viewport  = parent
        self._workspace = workspace
        self.setObjectName("LoftPanel")
        self.setStyleSheet(prefs.scale_stylesheet(_PANEL_STYLE))
        self.setMinimumWidth(prefs.scaled_px(280))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._picking_profile: bool = False
        self._picking_body:    bool = False
        self._target_body_id:  str | None = None

        self._build_ui()
        self._update_ok_enabled()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Loft")
        title.setObjectName("title")
        root.addWidget(title)

        root.addWidget(self._separator())

        # ── Profiles ──────────────────────────────────────────────────
        prof_header = QHBoxLayout()
        prof_header.setSpacing(6)
        prof_header.addWidget(self._section_label("Profiles"))
        prof_header.addStretch()
        self._pick_profile_btn = QPushButton("+ Add Profile")
        self._pick_profile_btn.setObjectName("pick_profile")
        self._pick_profile_btn.setCheckable(True)
        self._pick_profile_btn.clicked.connect(self._on_pick_profile_toggle)
        prof_header.addWidget(self._pick_profile_btn)
        root.addLayout(prof_header)

        self._profile_list = SelectionList(
            empty_text="Pick 2+ sketch profiles in order",
            reorderable=True)
        self._profile_list.entry_removed.connect(self._on_profile_removed)
        self._profile_list.order_changed.connect(self._on_order_changed)
        root.addWidget(self._profile_list)

        root.addWidget(self._separator())

        # ── Mode ──────────────────────────────────────────────────────
        root.addWidget(self._section_label("Mode"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self._radio_loft = QRadioButton("Loft")
        self._radio_cut  = QRadioButton("Cut")
        self._radio_loft.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._radio_loft, 0)
        self._mode_group.addButton(self._radio_cut,  1)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        mode_row.addWidget(self._radio_loft)
        mode_row.addWidget(self._radio_cut)
        root.addLayout(mode_row)

        # ── Operation ─────────────────────────────────────────────────
        root.addWidget(self._separator())
        root.addWidget(self._section_label("Operation"))

        # Loft mode: New Body / Merge with
        self._loft_op_widget = QWidget()
        loft_op_layout = QVBoxLayout(self._loft_op_widget)
        loft_op_layout.setContentsMargins(0, 0, 0, 0)
        loft_op_layout.setSpacing(6)
        self._radio_new   = QRadioButton("New Body")
        self._radio_merge = QRadioButton("Merge with")
        self._radio_new.setChecked(True)
        self._op_group = QButtonGroup(self)
        self._op_group.addButton(self._radio_new,   0)
        self._op_group.addButton(self._radio_merge, 1)
        self._op_group.idClicked.connect(self._on_op_changed)
        loft_op_layout.addWidget(self._radio_new)

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
        loft_op_layout.addLayout(merge_row)
        root.addWidget(self._loft_op_widget)

        # Cut mode: pick body to cut from (optional — empty = cut thru all)
        self._cut_op_widget = QWidget()
        cut_op_layout = QHBoxLayout(self._cut_op_widget)
        cut_op_layout.setContentsMargins(0, 0, 0, 0)
        cut_op_layout.setSpacing(6)
        self._pick_cut_body_btn = QPushButton("Pick Body")
        self._pick_cut_body_btn.setObjectName("pick_body")
        self._pick_cut_body_btn.setCheckable(True)
        self._pick_cut_body_btn.clicked.connect(self._on_pick_body_toggle)
        cut_op_layout.addWidget(self._pick_cut_body_btn)
        self._cut_body_label = QLabel("(cut thru all)")
        self._cut_body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        cut_op_layout.addWidget(self._cut_body_label)
        root.addWidget(self._cut_op_widget)
        self._cut_op_widget.hide()

        # ── Surface ───────────────────────────────────────────────────
        root.addWidget(self._separator())
        root.addWidget(self._section_label("Surface"))
        surf_row = QHBoxLayout()
        surf_row.setSpacing(6)
        surf_row.addWidget(QLabel("Continuity"))
        self._continuity_combo = QComboBox()
        # Display labels chosen for plain English; data carries OCCT enum names.
        self._continuity_combo.addItem("Smooth (C1)",        "C1")
        self._continuity_combo.addItem("Extra smooth (C2)",  "C2")
        self._continuity_combo.addItem("Allow creases (C0)", "C0")
        self._continuity_combo.setToolTip(
            "How smoothly the surface flows between profiles.\n"
            "C1 (default) matches tangents; C2 also matches curvature;\n"
            "C0 allows visible creases at each profile.")
        self._continuity_combo.currentIndexChanged.connect(
            lambda _i: self.preview_changed.emit())
        surf_row.addWidget(self._continuity_combo, 1)
        root.addLayout(surf_row)

        self._ruled_chk = QCheckBox("Ruled (straight walls between profiles)")
        self._ruled_chk.setToolTip(
            "Connect adjacent profiles with straight line segments rather\n"
            "than a blended spline.  Visible difference with 3+ profiles.")
        self._ruled_chk.toggled.connect(lambda _c: self.preview_changed.emit())
        root.addWidget(self._ruled_chk)

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

    def add_profile(self, sketch_entry_id: str, label: str,
                    face_idx: int | None = None) -> bool:
        key = (sketch_entry_id, face_idx)
        added = self._profile_list.add(key, label)
        if added:
            self._emit_changed()
        return added

    def remove_profile(self, sketch_entry_id: str,
                       face_idx: int | None = None):
        key = (sketch_entry_id, face_idx)
        ks = self._profile_list.keys
        if key in ks:
            # remove_at triggers _on_profile_removed → _emit_changed
            self._profile_list.remove_at(ks.index(key))

    def clear_profiles(self):
        self._profile_list.clear()
        self._emit_changed()

    def profiles(self) -> list:
        """Return [(sketch_id, face_idx_or_None)] in display order."""
        return list(self._profile_list.keys)

    def set_picked_body(self, body_id: str, body_name: str):
        self._target_body_id = body_id
        self._end_pick_body()
        for lbl in (self._body_label, self._cut_body_label):
            lbl.setText(body_name)
            lbl.setStyleSheet(prefs.scale_stylesheet(
                "color: #7ab3d4; font-size: 11px; font-family: monospace;"))
        self._emit_changed()

    # ------------------------------------------------------------------
    # OK gating
    # ------------------------------------------------------------------

    def _mode_str(self) -> str:
        """'cut' | 'merge' | 'new'"""
        if self._mode_group.checkedId() == 1:
            return "cut"
        return "merge" if self._op_group.checkedId() == 1 else "new"

    def _update_ok_enabled(self):
        if len(self._profile_list) < 2:
            self._ok_btn.setEnabled(False)
            self._ok_btn.setToolTip("Pick at least 2 sketch profiles")
            return
        mode = self._mode_str()
        if mode == "merge" and self._target_body_id is None:
            self._ok_btn.setEnabled(False)
            self._ok_btn.setToolTip("Pick a body to merge with")
            return
        self._ok_btn.setEnabled(True)
        if mode == "cut" and self._target_body_id is None:
            self._ok_btn.setToolTip("Cut through every intersecting body")
        else:
            self._ok_btn.setToolTip("")

    def _emit_changed(self):
        """Refresh OK gating + fire a preview signal.  Use from state slots."""
        self._update_ok_enabled()
        self.preview_changed.emit()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_mode_changed(self, btn_id: int):
        is_cut = btn_id == 1
        self._loft_op_widget.setVisible(not is_cut)
        self._cut_op_widget.setVisible(is_cut)
        # Reset target on mode change
        self._target_body_id = None
        self._body_label.setText("—")
        self._body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._cut_body_label.setText("(cut thru all)")
        self._cut_body_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #888; font-size: 11px; font-family: monospace;"))
        self._emit_changed()

    def _on_op_changed(self, btn_id: int):
        self._pick_body_btn.setEnabled(btn_id == 1)
        if btn_id == 0:
            self._target_body_id = None
            self._body_label.setText("—")
            self._body_label.setStyleSheet(prefs.scale_stylesheet(
                "color: #888; font-size: 11px; font-family: monospace;"))
            self._end_pick_body()
        self._emit_changed()

    def _on_pick_profile_toggle(self, checked: bool):
        if checked:
            self._picking_profile = True
            self._pick_profile_btn.setProperty("active", True)
            self._pick_profile_btn.style().unpolish(self._pick_profile_btn)
            self._pick_profile_btn.style().polish(self._pick_profile_btn)
            self.picking_profile_changed.emit(True)
        else:
            self._end_pick_profile()

    def _end_pick_profile(self):
        self._picking_profile = False
        self._pick_profile_btn.setChecked(False)
        self._pick_profile_btn.setProperty("active", False)
        self._pick_profile_btn.style().unpolish(self._pick_profile_btn)
        self._pick_profile_btn.style().polish(self._pick_profile_btn)
        self.picking_profile_changed.emit(False)

    def _on_pick_body_toggle(self, checked: bool):
        if checked:
            self._picking_body = True
            btn = (self._pick_cut_body_btn
                   if self._mode_group.checkedId() == 1
                   else self._pick_body_btn)
            btn.setProperty("active", True)
            btn.style().unpolish(btn); btn.style().polish(btn)
            self.picking_body_changed.emit(True)
        else:
            self._end_pick_body()

    def _end_pick_body(self):
        self._picking_body = False
        for btn in (self._pick_body_btn, self._pick_cut_body_btn):
            btn.setChecked(False)
            btn.setProperty("active", False)
            btn.style().unpolish(btn); btn.style().polish(btn)
        self.picking_body_changed.emit(False)

    def _on_profile_removed(self, index: int):
        self.profile_removed.emit(index)
        self._emit_changed()

    def _on_order_changed(self):
        self.profile_order_changed.emit()
        self.preview_changed.emit()

    def _on_ok(self):
        profiles = self.profiles()
        if len(profiles) < 2:
            return
        mode = self._mode_str()
        # Cut mode allows target=None (cut-thru-all).  Merge requires a target.
        target = self._target_body_id
        if mode == "merge" and target is None:
            return
        self.loft_requested.emit(
            profiles, mode, target, self.ruled(), self.continuity())

    def ruled(self) -> bool:
        return self._ruled_chk.isChecked()

    def continuity(self) -> str:
        return self._continuity_combo.currentData() or "C1"

    def set_ruled(self, value: bool):
        self._ruled_chk.setChecked(bool(value))

    def set_continuity(self, value: str):
        idx = self._continuity_combo.findData(value)
        if idx >= 0:
            self._continuity_combo.setCurrentIndex(idx)

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
            if self._picking_profile:
                self._end_pick_profile()
            elif self._picking_body:
                self._end_pick_body()
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
