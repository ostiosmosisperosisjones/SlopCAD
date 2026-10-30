"""
gui/sweep_panel.py

SweepPanel — LoftPanel with one profile plus a path: a list of sketch edges
(one sketch) the profile travels along.  Mode / target handling is
LoftPanel's; the loft surface options are swapped for the profile
orientation toggle.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QCheckBox
from PyQt6.QtCore import pyqtSignal

from gui.loft_panel import LoftPanel
from gui.selection_list import SelectionList


class SweepPanel(LoftPanel):
    sweep_requested       = pyqtSignal(list, object, object, str, list, bool)
    # (profiles, mode_str, target_body_id|None, path_sketch_id, path_uids, frenet)
    picking_path_changed  = pyqtSignal(bool)

    def __init__(self, workspace, parent=None):
        self._picking_path = False
        self._path_sketch_id: str | None = None
        super().__init__(workspace, parent)
        self.setWindowTitle("Sweep")
        self.setObjectName("LoftPanel")          # reuse the loft styling

    def _build_ui(self):
        super()._build_ui()
        root = self.layout()
        root.itemAt(0).widget().setText("Sweep")
        self._radio_loft.setText("Sweep")
        self._profile_list._empty_label.setText("Pick one closed sketch region")
        self._pick_profile_btn.setText("+ Pick Profile")

        # Loft's SURFACE section is [header, continuity row, ruled checkbox];
        # sweep's ORIENTATION section reuses the header, hides the rest.
        surf_idx = root.indexOf(self._ruled_chk)
        root.itemAt(surf_idx - 2).widget().setText("ORIENTATION")
        row = root.itemAt(surf_idx - 1).layout()
        for j in range(row.count()):
            row.itemAt(j).widget().hide()
        self._ruled_chk.hide()
        self._frenet_chk = QCheckBox("Profile turns with the path")
        self._frenet_chk.setChecked(True)
        self._frenet_chk.setToolTip(
            "On: the profile stays square to the path as it bends.\n"
            "Off: the profile keeps its sketched orientation.")
        self._frenet_chk.toggled.connect(lambda _c: self.preview_changed.emit())
        root.insertWidget(surf_idx + 1, self._frenet_chk)

        # ── Path, right after the profile list ────────────────────────
        at = root.indexOf(self._profile_list) + 1
        root.insertWidget(at, self._separator())
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        hdr.addWidget(self._section_label("Path"))
        hdr.addStretch()
        self._pick_path_btn = QPushButton("+ Add Path Edge")
        self._pick_path_btn.setObjectName("pick_profile")
        self._pick_path_btn.setCheckable(True)
        self._pick_path_btn.clicked.connect(self._on_pick_path_toggle)
        hdr.addWidget(self._pick_path_btn)
        root.insertLayout(at + 1, hdr)
        self._path_list = SelectionList(
            empty_text="Pick connected sketch edges")
        self._path_list.entry_removed.connect(lambda _i: self._on_path_removed())
        root.insertWidget(at + 2, self._path_list)

    # ------------------------------------------------------------------
    # Profile: exactly one — a new pick replaces the old
    # ------------------------------------------------------------------

    def add_profile(self, sketch_entry_id, label, face_idx=None) -> bool:
        if self._profile_list.keys and (sketch_entry_id, face_idx) not in self._profile_list.keys:
            self._profile_list.clear()
        added = super().add_profile(sketch_entry_id, label, face_idx)
        if added and self._picking_profile:
            self._end_pick_profile()
            if not len(self._path_list):
                self._pick_path_btn.setChecked(True)
                self._on_pick_path_toggle(True)
        return added

    # ------------------------------------------------------------------
    # Path
    # ------------------------------------------------------------------

    def toggle_path_edge(self, sketch_id: str, uid: str, label: str) -> str | None:
        """Add the edge, or remove it if already in.  Edges from another
        sketch than the path's start a new path.  Returns None or a reason
        the pick was refused."""
        if self._path_sketch_id not in (None, sketch_id) and len(self._path_list):
            self._path_list.clear()
        self._path_sketch_id = sketch_id
        keys = self._path_list.keys
        if uid in keys:
            self._path_list.remove_at(keys.index(uid))
        else:
            self._path_list.add(uid, label)
            self._emit_changed()
        return None

    def path(self) -> tuple[str | None, list]:
        uids = list(self._path_list.keys)
        return (self._path_sketch_id if uids else None), uids

    def frenet(self) -> bool:
        return self._frenet_chk.isChecked()

    def set_frenet(self, value: bool):
        self._frenet_chk.setChecked(bool(value))

    def _on_path_removed(self):
        if not len(self._path_list):
            self._path_sketch_id = None
        self._emit_changed()

    def _on_pick_path_toggle(self, checked: bool):
        if checked:
            if self._picking_profile:
                self._end_pick_profile()
            self._picking_path = True
            self._set_active(self._pick_path_btn, True)
            self.picking_path_changed.emit(True)
        else:
            self._end_pick_path()

    def _on_pick_profile_toggle(self, checked: bool):
        if checked and self._picking_path:
            self._end_pick_path()
        super()._on_pick_profile_toggle(checked)

    def _end_pick_path(self):
        self._picking_path = False
        self._pick_path_btn.setChecked(False)
        self._set_active(self._pick_path_btn, False)
        self.picking_path_changed.emit(False)

    @staticmethod
    def _set_active(btn, on: bool):
        btn.setProperty("active", on)
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    # ------------------------------------------------------------------
    # OK gating / emit
    # ------------------------------------------------------------------

    def _update_ok_enabled(self):
        if not hasattr(self, "_path_list"):        # during base __init__
            return super()._update_ok_enabled()
        if len(self._profile_list) != 1:
            self._ok_btn.setEnabled(False)
            self._ok_btn.setToolTip("Pick a sketch profile")
            return
        if not len(self._path_list):
            self._ok_btn.setEnabled(False)
            self._ok_btn.setToolTip("Pick the path's sketch edges, or a helix")
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
        profiles = self.profiles()
        sid, uids = self.path()
        if len(profiles) != 1 or not uids:
            return
        mode = self._mode_str()
        if mode == "merge" and self._target_body_id is None:
            return
        self.sweep_requested.emit(profiles, mode, self._target_body_id,
                                  sid, uids, self.frenet())

    def keyPressEvent(self, e):
        from PyQt6.QtCore import Qt
        if e.key() == Qt.Key.Key_Escape and self._picking_path:
            self._end_pick_path()
            return
        super().keyPressEvent(e)
