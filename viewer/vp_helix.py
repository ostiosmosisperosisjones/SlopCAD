"""
viewer/vp_helix.py

HelixMixin — helix panel lifecycle, cylindrical-face / sketch-circle pick,
the curve preview, commit and edit.  The curves are drawn and picked by
viewer/curve_overlay.py.
"""

from __future__ import annotations


class HelixMixin:

    def _try_helix(self):
        if getattr(self, '_helix_panel', None) is not None:
            return
        self._show_helix_panel()
        sel = self.selection
        done = any(self._set_helix_face(f.body_id, f.face_idx) for f in sel.faces[:1])
        if not done:
            done = any(self._set_helix_circle(se.history_idx, se.entity_idx)
                       for se in sel.sketch_edges if se.history_idx != -1)
        if not done:
            self._helix_panel.set_picking(True)

    def _show_helix_panel(self):
        from gui.helix_panel import HelixPanel
        panel = HelixPanel(parent=self)
        panel.helix_requested.connect(self._on_helix_ok)
        panel.cancelled.connect(self._close_helix_panel)
        panel.picking_changed.connect(
            lambda a: setattr(self, '_helix_pick_active', a))
        panel.preview_changed.connect(self._update_helix_preview)
        self._helix_panel        = panel
        self._helix_source       = None     # dict of HelixOp source fields
        self._helix_pick_active  = False
        self._helix_preview_pts  = None
        origin = self.mapToGlobal(self.rect().topLeft())
        panel.move(origin.x() + 16, origin.y() + 16)
        panel.show()
        panel.setFocus()

    def _close_helix_panel(self):
        p = getattr(self, '_helix_panel', None)
        self._helix_panel = None
        if p is not None:
            p.close()
        self._helix_pick_active = False
        self._helix_preview_pts = None
        idx = getattr(self, '_editing_helix_idx', None)
        if idx is not None:                 # closed without OK while editing
            self._editing_helix_idx = None
            entries = self.history.entries
            if 0 <= idx < len(entries):
                entries[idx].editing = False
            self.history_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Source picks
    # ------------------------------------------------------------------

    def _set_helix_face(self, body_id: str, face_idx: int) -> bool:
        from cad.face_ref import AnyFaceRef
        from cad.op_helix import cylinder_axis
        from cad.units import format_value
        from cad.prefs import prefs
        panel = getattr(self, '_helix_panel', None)
        shape = self.workspace.current_shape(body_id)
        if panel is None or shape is None:
            return False
        faces = list(shape.faces())
        if not 0 <= face_idx < len(faces):
            return False
        ax = cylinder_axis(faces[face_idx].wrapped)
        if ax is None:
            print("[Helix] Pick a cylindrical face (or a sketch circle).")
            return False
        self._helix_source = dict(
            body_id=body_id, face_ref=AnyFaceRef.from_occ_face(faces[face_idx].wrapped))
        body = self.workspace.bodies.get(body_id)
        d = format_value(2 * ax[3], prefs.default_unit, prefs.display_decimals)
        panel.set_source("face", f"{body.name if body else body_id} · face "
                                 f"{face_idx}  ⌀ {d}", length=ax[4])
        return True

    def _set_helix_circle(self, history_idx: int, entity_idx: int) -> bool:
        import math
        from cad.sketch import ArcEntity
        from cad.units import format_value
        from cad.prefs import prefs
        panel = getattr(self, '_helix_panel', None)
        entries = self.history.entries
        if panel is None or not (0 <= history_idx < len(entries)):
            return False
        se = entries[history_idx].params.get("sketch_entry")
        if se is None or not (0 <= entity_idx < len(se.entities)):
            return False
        ent = se.entities[entity_idx]
        if not (isinstance(ent, ArcEntity)
                and abs(ent.end_angle - ent.start_angle - 2 * math.pi) < 1e-6):
            print("[Helix] Pick a full circle (or a cylindrical face).")
            return False
        self._helix_source = dict(circle_sketch_id=entries[history_idx].entry_id,
                                  circle_uid=ent.uid)
        d = format_value(2 * ent.radius, prefs.default_unit, prefs.display_decimals)
        panel.set_source("circle", f"Sketch {history_idx} · circle {entity_idx}  ⌀ {d}")
        return True

    def route_face_pick_for_helix(self, body_id: str, face_idx: int) -> bool:
        """Takes cylindrical faces only: a click on a flat face falls through
        to the sketch-edge pick (a circle drawn on it)."""
        if not getattr(self, '_helix_pick_active', False):
            return False
        ok = self._set_helix_face(body_id, face_idx)
        self.update()
        return ok

    def route_sketch_edge_pick_for_helix(self, history_idx: int, entity_idx: int) -> bool:
        if not getattr(self, '_helix_pick_active', False):
            return False
        self._set_helix_circle(history_idx, entity_idx)
        self.update()
        return True

    # ------------------------------------------------------------------
    # Preview / OK / edit
    # ------------------------------------------------------------------

    def _helix_op(self, fields):
        from cad.op_helix import HelixOp
        return HelixOp(**self._helix_source, **fields)

    def _update_helix_preview(self):
        from viewer.curve_overlay import helix_points
        panel = getattr(self, '_helix_panel', None)
        f = panel.fields() if panel is not None else None
        self._helix_preview_pts = None
        if f is not None and self._helix_source is not None:
            idx = getattr(self, '_editing_helix_idx', None)
            at = idx if idx is not None else self.history.cursor + 1
            try:
                edge, _axis, turns = self._helix_op(f).edge(self.history, at)
                self._helix_preview_pts = helix_points(edge, turns)
            except Exception as ex:
                print(f"[Helix preview] {ex}")
        self.update()

    def _on_helix_ok(self, fields: dict):
        from gui.expr_spinbox import exprs_extra
        from cad.units import format_op_label as _lbl
        panel = self._helix_panel
        op = self._helix_op(fields)
        extra = exprs_extra(**panel.spinboxes()) or {}
        idx = getattr(self, '_editing_helix_idx', None)
        self._editing_helix_idx = None
        self._close_helix_panel()
        if idx is None:
            op.commit(self, extra or None)
            return
        entry = self.history.entries[idx]
        entry.params.clear()
        entry.params.update(op.to_params(), **extra)
        self.history.update_entry_op(idx, {})
        entry.label = _lbl("helix", entry.params)
        entry.editing = False
        ok, err, _ = self.history.replay_from(idx)
        if not ok:
            print(f"[Helix edit] Downstream replay failed: {err}")
        self.history.seek(len(self.history.entries) - 1)
        self._rebuild_all_meshes()
        self.history_changed.emit()
        self.update()

    def reopen_helix(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation != "helix" or entry.op is None:
            return
        if getattr(self, '_helix_panel', None) is not None:
            self._close_helix_panel()
        op = entry.op
        entry.editing = True
        self._editing_helix_idx = history_idx
        self._show_helix_panel()
        panel = self._helix_panel
        if op.face_ref is not None:
            self._helix_source = dict(body_id=op.body_id, face_ref=op.face_ref)
            try:
                length = op.axis(self.history, history_idx)[4] \
                    if op.height <= 0 else None
            except Exception:
                length = None
            body = self.workspace.bodies.get(op.body_id)
            panel.set_source("face", f"{body.name if body else op.body_id} · "
                                     f"face (saved)", length=length)
        else:
            self._helix_source = dict(circle_sketch_id=op.circle_sketch_id,
                                      circle_uid=op.circle_uid)
            panel.set_source("circle", "circle (saved)")
        panel.set_fields(op, entry.params.get("exprs", {}))
        self.history_changed.emit()
