"""
viewer/vp_coil.py

CoilMixin — panel lifecycle, circle / body pick routing, preview and
dispatch for coil / coil-cut.  Same shape as SweepMixin.
"""

from __future__ import annotations
from cad.solid_ref import as_compound


class CoilMixin:

    def _try_coil(self):
        if getattr(self, '_coil_panel', None) is not None:
            return
        self._show_coil_panel()
        for se in self.selection.sketch_edges:     # a preselected circle
            if se.history_idx != -1 and self._set_coil_circle(se.history_idx,
                                                               se.entity_idx):
                break
        p = self._coil_panel
        if p.circle() is None:
            self._arm_pick(p._pick_circle_btn, p._on_pick_circle_toggle)

    def _show_coil_panel(self):
        from gui.coil_panel import CoilPanel
        if getattr(self, '_coil_panel', None) is not None:
            return
        panel = CoilPanel(self.workspace, parent=self)
        panel.coil_requested.connect(self._on_coil_panel_ok)
        panel.cancelled.connect(self._close_coil_panel)
        panel.picking_circle_changed.connect(
            lambda a: setattr(self, '_coil_circle_pick_active', a))
        panel.picking_body_changed.connect(
            lambda a: setattr(self, '_coil_body_pick_active', a))
        panel.preview_changed.connect(self._on_coil_preview_live)
        self._coil_panel               = panel
        self._coil_circle_pick_active  = False
        self._coil_body_pick_active    = False
        self._coil_preview_mesh        = None
        self._coil_preview_is_cut      = False
        origin = self.mapToGlobal(self.rect().topLeft())
        panel.move(origin.x() + 16, origin.y() + 16)
        panel.show()
        panel.setFocus()

    def _close_coil_panel(self):
        p = getattr(self, '_coil_panel', None)
        self._coil_panel = None
        if p is not None:
            p.close()
        self._coil_circle_pick_active = False
        self._coil_body_pick_active   = False
        self._coil_job().cancel()
        self._coil_preview_mesh       = None
        idx = getattr(self, '_editing_coil_idx', None)
        if idx is not None:                 # closed without OK while editing
            self._editing_coil_idx = None
            entries = self.history.entries
            if 0 <= idx < len(entries):
                entries[idx].editing = False
                self.history.seek(idx)
                self._rebuild_all_meshes()
            self.history_changed.emit()
        self.update()

    @staticmethod
    def _coil_op(mode, target_body_id, fields):
        from cad.op_coil import CoilOp
        return CoilOp(merge_body_id  = target_body_id if mode == "merge" else None,
                      cut_body_id    = target_body_id if mode == "cut" else None,
                      force_new_body = (mode == "new"),
                      **fields)

    # ------------------------------------------------------------------
    # Reopen / edit
    # ------------------------------------------------------------------

    def reopen_coil(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation not in ("coil", "coil_cut") or entry.op is None:
            return
        entry.editing = True
        self._editing_coil_idx = history_idx
        op = entry.op
        if history_idx > 0:
            self.history.seek(history_idx - 1)
            self._rebuild_all_meshes()
        self.history_changed.emit()

        self._show_coil_panel()
        panel = self._coil_panel
        cidx = self.history.id_to_index(op.circle_sketch_id)
        if cidx is not None:
            se = entries[cidx].params.get("sketch_entry")
            for i, e in enumerate(se.entities if se else []):
                if e.uid == op.circle_uid:
                    self._set_coil_circle(cidx, i)
        panel.set_fields(op, entry.params.get("exprs", {}))
        if op.cut_body_id is not None and "fanout_id" not in entry.params:
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)
            if op.cut_body_id in self.workspace.bodies:
                panel.set_picked_body(op.cut_body_id,
                                      self.workspace.bodies[op.cut_body_id].name)
        elif entry.operation == "coil_cut":
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)
        elif op.merge_body_id is not None and not op.force_new_body:
            panel._radio_merge.setChecked(True)
            panel._on_op_changed(1)
            if op.merge_body_id in self.workspace.bodies:
                panel.set_picked_body(op.merge_body_id,
                                      self.workspace.bodies[op.merge_body_id].name)

    # ------------------------------------------------------------------
    # Pick routing
    # ------------------------------------------------------------------

    def _set_coil_circle(self, history_idx: int, entity_idx: int) -> bool:
        import math
        from cad.sketch import ArcEntity
        from cad.units import format_value
        from cad.prefs import prefs
        panel = getattr(self, '_coil_panel', None)
        entries = self.history.entries
        if panel is None or not (0 <= history_idx < len(entries)):
            return False
        se = entries[history_idx].params.get("sketch_entry")
        if se is None or not (0 <= entity_idx < len(se.entities)):
            return False
        ent = se.entities[entity_idx]
        if not (isinstance(ent, ArcEntity)
                and abs(ent.end_angle - ent.start_angle - 2 * math.pi) < 1e-6):
            print("[Coil] Pick a full circle.")
            return False
        d = format_value(2 * ent.radius, prefs.default_unit, prefs.display_decimals)
        panel.set_circle(entries[history_idx].entry_id, ent.uid,
                         f"Sketch {history_idx}  ·  circle {entity_idx}  ⌀ {d}")
        return True

    def route_sketch_edge_pick_for_coil(self, history_idx: int, entity_idx: int) -> bool:
        if not getattr(self, '_coil_circle_pick_active', False):
            return False
        self._set_coil_circle(history_idx, entity_idx)
        self.update()
        return True                          # swallow non-circles while picking

    def route_body_pick_for_coil(self, body_id: str) -> bool:
        if not getattr(self, '_coil_body_pick_active', False):
            return False
        body = self.workspace.bodies.get(body_id)
        panel = getattr(self, '_coil_panel', None)
        if body is None or panel is None:
            return False
        panel.set_picked_body(body_id, body.name)
        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _draw_coil_preview(self):
        if getattr(self, '_coil_preview_mesh', None) is None:
            return
        saved = (getattr(self, '_loft_preview_mesh', None),
                 getattr(self, '_loft_preview_is_cut', False))
        self._loft_preview_mesh   = self._coil_preview_mesh
        self._loft_preview_is_cut = self._coil_preview_is_cut
        try:
            self._draw_loft_preview()
        finally:
            self._loft_preview_mesh, self._loft_preview_is_cut = saved

    def _on_coil_preview_live(self):
        panel = getattr(self, '_coil_panel', None)
        if panel is None:
            return
        f = panel.fields()
        if f is None:
            self._coil_job().cancel()
            self._coil_preview_mesh = None
            self.update()
            return
        probe = self._coil_op("new", None, f)
        try:
            probe._resolve_profile_faces(self.history, self.history.cursor + 1)
        except Exception as ex:
            print(f"[Coil preview] {ex}")
            self._coil_preview_mesh = None
            self.update()
            return
        self._coil_preview_is_cut = (panel._mode_str() == "cut")

        def _compute():
            from viewer.preview_job import preview_mesh_data, pack_preview
            tris, edges = preview_mesh_data(probe._task([]).wrapped, 0.15, 24)
            return pack_preview([tris], [edges])
        self._coil_job().submit(_compute)

    def _coil_job(self):
        from viewer.preview_job import job_for
        return job_for(self, '_coil_preview_job', self._on_coil_mesh_done,
                       "Coil preview")

    def _on_coil_mesh_done(self, result, error):
        if getattr(self, '_coil_panel', None) is None:
            return
        if error:
            print(f"[Coil preview] {error}")
        self._coil_preview_mesh = result
        self.update()

    # ------------------------------------------------------------------
    # OK
    # ------------------------------------------------------------------

    def _on_coil_panel_ok(self, mode, target_body_id, fields):
        from gui.expr_spinbox import exprs_extra
        from cad.cut_all import fan_out_cut
        panel = self._coil_panel
        extra = exprs_extra(**panel.spinboxes()) or {}
        editing_idx = getattr(self, '_editing_coil_idx', None)
        self._editing_coil_idx = None          # so close doesn't restore
        self._close_coil_panel()
        preserved = (self._tear_down_loft_entry_group(editing_idx)
                     if editing_idx is not None else None)
        if preserved:
            extra["_preserved_body_ids"] = preserved

        if mode == "cut" and target_body_id is None:
            probe = self._coil_op("new", None, fields)
            try:
                probe._resolve_profile_faces(self.history, self.history.cursor + 1)
                tool = as_compound(probe._task([]))
            except Exception as ex:
                print(f"[Coil cut] Tool construction failed: {ex}")
                return
            exprs = extra.get("exprs")
            fan_out_cut(self, tool, lambda bid: self._coil_op("cut", bid, fields),
                        {"exprs": exprs} if exprs else None, op_label="Coil cut")
        else:
            op = self._coil_op(mode, target_body_id, fields)
            if editing_idx is None:
                op.commit_async(self, extra or None)
                return
            op.commit(self, extra or None)
        if editing_idx is not None:
            ok, err, _ = self.history.replay_from(self.history.cursor + 1)
            if not ok:
                print(f"[Edit] Downstream replay failed: {err}")
            self.history.seek(len(self.history.entries) - 1)
            self._rebuild_all_meshes()
            self.history_changed.emit()
