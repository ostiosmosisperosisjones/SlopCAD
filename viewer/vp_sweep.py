"""
viewer/vp_sweep.py

SweepMixin — panel lifecycle, profile / path / body pick routing, preview
and dispatch for sweep / sweep-cut.  Entry teardown on edit is the loft
mixin's (_tear_down_loft_entry_group is op-agnostic).
"""

from __future__ import annotations
from cad.solid_ref import as_compound


class SweepMixin:

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _try_sweep(self):
        if getattr(self, '_sweep_panel', None) is not None:
            return
        self._show_sweep_panel()
        p = self._sweep_panel
        # Preselection: a sketch face → the profile; committed sketch edges → path.
        hidx = getattr(self, '_selected_sketch_entry', None)
        if hidx is not None and getattr(self, '_selected_sketch_face', None):
            self._sweep_profile_pick_active = True
            self.route_sketch_face_pick_for_sweep(hidx, self._selected_sketch_face[0])
            self._sweep_profile_pick_active = False
        for se in self.selection.sketch_edges:
            if se.history_idx != -1:
                self._add_sweep_path_edge(se.history_idx, se.entity_idx)
        if not p.profiles():
            self._arm_pick(p._pick_profile_btn, p._on_pick_profile_toggle)
        elif not p.path()[1]:
            self._arm_pick(p._pick_path_btn, p._on_pick_path_toggle)

    def _show_sweep_panel(self):
        from gui.sweep_panel import SweepPanel
        if getattr(self, '_sweep_panel', None) is not None:
            return
        panel = SweepPanel(self.workspace, parent=self)
        panel.sweep_requested.connect(self._on_sweep_panel_ok)
        panel.cancelled.connect(self._close_sweep_panel)
        panel.picking_profile_changed.connect(
            lambda a: setattr(self, '_sweep_profile_pick_active', a))
        panel.picking_path_changed.connect(
            lambda a: setattr(self, '_sweep_path_pick_active', a))
        panel.picking_body_changed.connect(
            lambda a: setattr(self, '_sweep_body_pick_active', a))
        panel.preview_changed.connect(self._on_sweep_preview_live)
        self._sweep_panel               = panel
        self._sweep_profile_pick_active = False
        self._sweep_path_pick_active    = False
        self._sweep_body_pick_active    = False
        self._sweep_preview_mesh        = None
        self._sweep_preview_is_cut      = False
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        panel.move(origin.x() + margin, origin.y() + margin)
        panel.show()
        panel.setFocus()

    def _close_sweep_panel(self):
        p = getattr(self, '_sweep_panel', None)
        self._sweep_panel = None
        if p is not None:
            p.close()
        self._sweep_profile_pick_active = False
        self._sweep_path_pick_active    = False
        self._sweep_body_pick_active    = False
        self._sweep_job().cancel()
        self._sweep_preview_mesh        = None
        self._helix_highlight_idx       = None
        idx = getattr(self, '_editing_sweep_idx', None)
        if idx is not None:                 # closed without OK while editing
            self._editing_sweep_idx = None
            entries = self.history.entries
            if 0 <= idx < len(entries):
                entries[idx].editing = False
                self.history.seek(idx)
                self._rebuild_all_meshes()
            self.history_changed.emit()
        self.update()

    def _sweep_op(self, profiles, mode, target_body_id, path_sid, path_uids, frenet):
        from cad.op_sweep import SketchSweepOp
        return SketchSweepOp(
            from_profiles  = list(profiles),
            merge_body_id  = target_body_id if mode == "merge" else None,
            cut_body_id    = target_body_id if mode == "cut" else None,
            force_new_body = (mode == "new"),
            path_sketch_id = path_sid or "",
            path_uids      = list(path_uids),
            frenet         = frenet,
        )

    # ------------------------------------------------------------------
    # Reopen / edit
    # ------------------------------------------------------------------

    def reopen_sweep(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation not in ("sweep", "sweep_cut") or entry.op is None:
            return
        entry.editing = True
        self._editing_sweep_idx = history_idx
        op = entry.op
        if history_idx > 0:
            self.history.seek(history_idx - 1)
            self._rebuild_all_meshes()
        self.history_changed.emit()

        self._show_sweep_panel()
        panel = self._sweep_panel
        for sid, face_idx in op.from_profiles:
            sidx = self.history.id_to_index(sid)
            if sidx is not None:
                panel.add_profile(sid, self._sweep_profile_label(sidx, face_idx),
                                  face_idx=face_idx)
        pidx = self.history.id_to_index(op.path_sketch_id)
        if pidx is not None and entries[pidx].operation == "helix":
            self.route_helix_pick_for_sweep(pidx, force=True)
        elif pidx is not None:
            se = entries[pidx].params.get("sketch_entry")
            by_uid = {e.uid: i for i, e in enumerate(se.entities)} if se else {}
            for uid in op.path_uids:
                if uid in by_uid:
                    self._add_sweep_path_edge(pidx, by_uid[uid])
        if op.cut_body_id is not None and "fanout_id" not in entry.params:
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)
            if op.cut_body_id in self.workspace.bodies:
                panel.set_picked_body(op.cut_body_id,
                                      self.workspace.bodies[op.cut_body_id].name)
        elif entry.operation == "sweep_cut":
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)
        elif op.merge_body_id is not None and not op.force_new_body:
            panel._radio_merge.setChecked(True)
            panel._on_op_changed(1)
            if op.merge_body_id in self.workspace.bodies:
                panel.set_picked_body(op.merge_body_id,
                                      self.workspace.bodies[op.merge_body_id].name)
        panel.set_frenet(op.frenet)

    # ------------------------------------------------------------------
    # Pick routing
    # ------------------------------------------------------------------

    @staticmethod
    def _sweep_profile_label(sidx, face_idx):
        return (f"Sketch {sidx}" if face_idx is None
                else f"Sketch {sidx}  ·  face {face_idx}")

    def route_sketch_face_pick_for_sweep(self, sketch_idx: int, face_idx: int) -> bool:
        if not getattr(self, '_sweep_profile_pick_active', False):
            return False
        panel = getattr(self, '_sweep_panel', None)
        entries = self.history.entries
        if panel is None or not (0 <= sketch_idx < len(entries)):
            return False
        entry = entries[sketch_idx]
        if entry.operation != "sketch":
            return False
        face_count = len(self._sketch_faces.get(sketch_idx, []))
        key_face_idx = None if face_count <= 1 else face_idx
        panel.add_profile(entry.entry_id,
                          self._sweep_profile_label(sketch_idx, key_face_idx),
                          face_idx=key_face_idx)
        return True

    def _add_sweep_path_edge(self, history_idx: int, entity_idx: int) -> bool:
        panel = getattr(self, '_sweep_panel', None)
        entries = self.history.entries
        if panel is None or not (0 <= history_idx < len(entries)):
            return False
        se = entries[history_idx].params.get("sketch_entry")
        if se is None or not (0 <= entity_idx < len(se.entities)):
            return False
        ent = se.entities[entity_idx]
        if se.entity_edge(ent) is None:
            print("[Sweep] Path edges must be lines, arcs or splines.")
            return True
        self._helix_highlight_idx = None
        panel.toggle_path_edge(entries[history_idx].entry_id, ent.uid,
                               f"Sketch {history_idx}  ·  "
                               f"{type(ent).__name__.replace('Entity', '').lower()} "
                               f"{entity_idx}")
        return True

    def route_helix_pick_for_sweep(self, history_idx: int, force: bool = False) -> bool:
        """A helix curve picked as the path: it replaces any path edges."""
        from cad.op_sweep import HELIX_PATH
        panel = getattr(self, '_sweep_panel', None)
        if panel is None or not (force or getattr(self, '_sweep_path_pick_active', False)):
            return False
        entry = self.history.entries[history_idx]
        if panel.path() != (entry.entry_id, [HELIX_PATH]):
            panel.toggle_path_edge(entry.entry_id, HELIX_PATH,
                                   f"Helix {history_idx}  ·  {entry.label}")
        self._helix_highlight_idx = history_idx
        return True

    def route_sketch_edge_pick_for_sweep(self, history_idx: int, entity_idx: int) -> bool:
        if not getattr(self, '_sweep_path_pick_active', False):
            return False
        ok = self._add_sweep_path_edge(history_idx, entity_idx)
        self.update()
        return ok

    def route_body_pick_for_sweep(self, body_id: str) -> bool:
        if not getattr(self, '_sweep_body_pick_active', False):
            return False
        body = self.workspace.bodies.get(body_id)
        panel = getattr(self, '_sweep_panel', None)
        if body is None or panel is None:
            return False
        panel.set_picked_body(body_id, body.name)
        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _draw_sweep_preview(self):
        if getattr(self, '_sweep_preview_mesh', None) is None:
            return
        # Same look as the loft preview: swap the loft fields in for one draw.
        saved = (getattr(self, '_loft_preview_mesh', None),
                 getattr(self, '_loft_preview_is_cut', False))
        self._loft_preview_mesh   = self._sweep_preview_mesh
        self._loft_preview_is_cut = self._sweep_preview_is_cut
        try:
            self._draw_loft_preview()
        finally:
            self._loft_preview_mesh, self._loft_preview_is_cut = saved

    def _on_sweep_preview_live(self):
        panel = getattr(self, '_sweep_panel', None)
        if panel is None:
            return
        sid, uids = panel.path()
        if len(panel.profiles()) != 1 or not uids:
            self._sweep_job().cancel()
            self._sweep_preview_mesh = None
            self.update()
            return
        probe = self._sweep_op(panel.profiles(), "new", None, sid, uids, panel.frenet())
        try:
            faces = probe._resolve_profile_faces(self.history, self.history.cursor + 1)
        except Exception as ex:
            print(f"[Sweep preview] {ex}")
            self._sweep_preview_mesh = None
            self.update()
            return
        self._sweep_preview_is_cut = (panel._mode_str() == "cut")
        path_edges, frenet, binormal = probe._path_edges, probe.frenet, probe._binormal

        def _compute():
            from viewer.preview_job import sweep_preview_meshes
            return sweep_preview_meshes(faces, path_edges, frenet, binormal)
        self._sweep_job().submit(_compute)

    def _sweep_job(self):
        from viewer.preview_job import job_for
        return job_for(self, '_sweep_preview_job', self._on_sweep_mesh_done,
                       "Sweep preview")

    def _on_sweep_mesh_done(self, result, error):
        if getattr(self, '_sweep_panel', None) is None:
            return
        if error:
            print(f"[Sweep preview] {error}")
        self._sweep_preview_mesh = result
        self.update()

    # ------------------------------------------------------------------
    # OK
    # ------------------------------------------------------------------

    def _do_sweep_cut_thru_all(self, profiles, path_sid, path_uids, frenet):
        from cad.cut_all import fan_out_cut
        probe = self._sweep_op(profiles, "new", None, path_sid, path_uids, frenet)
        try:
            faces = probe._resolve_profile_faces(self.history, self.history.cursor + 1)
            tool = as_compound(probe._task(faces))
        except Exception as ex:
            print(f"[Sweep cut] Tool construction failed: {ex}")
            return
        fan_out_cut(self, tool,
                    lambda bid: self._sweep_op(profiles, "cut", bid,
                                               path_sid, path_uids, frenet),
                    None, op_label="Sweep cut")

    def _on_sweep_panel_ok(self, profiles, mode, target_body_id,
                           path_sid, path_uids, frenet):
        editing_idx = getattr(self, '_editing_sweep_idx', None)
        self._editing_sweep_idx = None         # so close doesn't restore
        self._close_sweep_panel()
        preserved = (self._tear_down_loft_entry_group(editing_idx)
                     if editing_idx is not None else None)

        if mode == "cut" and target_body_id is None:
            self._do_sweep_cut_thru_all(profiles, path_sid, path_uids, frenet)
        else:
            op = self._sweep_op(profiles, mode, target_body_id,
                                path_sid, path_uids, frenet)
            if editing_idx is None:
                op.commit_async(self, None)
                return
            op.commit(self, {"_preserved_body_ids": preserved} if preserved else None)
        if editing_idx is not None:
            ok, err, _ = self.history.replay_from(self.history.cursor + 1)
            if not ok:
                print(f"[Edit] Downstream replay failed: {err}")
            self.history.seek(len(self.history.entries) - 1)
            self._rebuild_all_meshes()
            self.history_changed.emit()
