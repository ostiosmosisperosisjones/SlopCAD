"""
viewer/vp_shell.py

ShellMixin — shell panel, open-face picking, preview, commit and edit.
The preview swaps the hollowed body in for the source body (see
Viewport._visible_meshes), like chamfer.
"""

from __future__ import annotations


class ShellMixin:

    def _try_shell(self):
        faces = self.selection.faces
        body_id, face_indices = None, []
        if faces:
            body_id = faces[0].body_id
            if any(f.body_id != body_id for f in faces):
                print("[Shell] All selected faces must be on the same body."); return
            if self.workspace.current_shape(body_id) is None:
                print("[Shell] Active body has no shape."); return
            face_indices = [f.face_idx for f in faces]
        else:
            # One body in the scene: it's the target (closed hollow until
            # an open face is picked).  Otherwise the first pick decides.
            solid = [b for b in self.workspace.bodies
                     if self.workspace.current_shape(b) is not None]
            if len(solid) == 1:
                body_id = solid[0]
        self._show_shell_panel(body_id, face_indices)
        if not face_indices:
            p = self._shell_panel
            self._arm_pick(p._pick_face_btn, p._on_pick_face_toggle)

    def _show_shell_panel(self, body_id, face_indices: list, editing_entry=None):
        from gui.shell_panel import ShellPanel
        if getattr(self, '_shell_panel', None) is not None:
            self._shell_panel._preview_timer.stop()
            self._shell_panel.close()
            self._shell_panel = None
        self._shell_body_id      = body_id
        self._shell_face_indices = list(face_indices)
        self._shell_preview_mesh = None
        panel = ShellPanel(parent=self)
        panel.thicken_requested.connect(self._on_shell_ok)
        panel.cancelled.connect(self._close_shell_panel)
        panel.preview_changed.connect(self._update_shell_preview)
        panel._mode_group.idClicked.connect(lambda _i: panel._emit_preview())
        panel.face_entry_removed.connect(self._on_shell_face_removed)
        for fi in face_indices:
            panel.add_face_entry(body_id, fi, self._shell_face_label(body_id, fi))
        self._shell_panel = panel
        origin = self.mapToGlobal(self.rect().topLeft())
        panel.move(origin.x() + 16, origin.y() + 16)
        panel.show()
        panel.setFocus()
        panel._emit_preview()

    def _shell_face_label(self, body_id, face_idx: int) -> str:
        body = self.workspace.bodies.get(body_id)
        return f"{body.name if body else body_id} · face {face_idx}"

    def _close_shell_panel(self):
        panel = getattr(self, '_shell_panel', None)
        if panel is not None:
            self._shell_panel = None
            panel._preview_timer.stop()
            panel.end_pick_face()
            for sig, slot in ((panel.preview_changed, self._update_shell_preview),
                              (panel.face_entry_removed, self._on_shell_face_removed),
                              (panel.cancelled, self._close_shell_panel)):
                try:
                    sig.disconnect(slot)
                except Exception:
                    pass
            panel.close()
        if getattr(self, '_editing_shell_idx', None) is not None:
            self._cancel_shell_edit()
        self._shell_preview_mesh = None
        self._shell_job().cancel()
        self.update()

    # ------------------------------------------------------------------
    # Face picking
    # ------------------------------------------------------------------

    def _on_shell_face_removed(self, index: int):
        indices = getattr(self, '_shell_face_indices', [])
        if 0 <= index < len(indices):
            indices.pop(index)
        panel = getattr(self, '_shell_panel', None)
        if panel is not None:
            panel._emit_preview()
        self.update()

    def route_face_pick_for_shell(self, body_id: str, face_idx: int) -> bool:
        """Face click while the shell panel's pick mode is on: toggle that
        face open / closed.  Returns True if consumed."""
        panel = getattr(self, '_shell_panel', None)
        if panel is None or not panel._picking_face:
            return False
        if getattr(self, '_shell_body_id', None) is None:
            self._shell_body_id = body_id
        elif body_id != self._shell_body_id:
            print("[Shell] All faces must be on the same body.")
            return True
        indices = self._shell_face_indices
        if face_idx in indices:
            panel.remove_face_entry(indices.index(face_idx))
        else:
            indices.append(face_idx)
            panel.add_face_entry(body_id, face_idx,
                                 self._shell_face_label(body_id, face_idx))
            panel._emit_preview()
        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _shell_job(self):
        from viewer.preview_job import job_for
        return job_for(self, '_shell_preview_job', self._on_shell_preview_done,
                       "Shell preview")

    def _update_shell_preview(self, thickness: float):
        panel = getattr(self, '_shell_panel', None)
        body_id = getattr(self, '_shell_body_id', None)
        shape = self.workspace.current_shape(body_id) if body_id else None
        if panel is None or shape is None or not thickness or thickness <= 0:
            self._shell_preview_mesh = None
            self._shell_job().cancel()
            self.update()
            return
        all_faces = list(shape.faces())
        face_occs = [all_faces[i].wrapped for i in self._shell_face_indices
                     if i < len(all_faces)]
        outside = panel.is_outside()
        panel.clear_face_errors()

        def _compute():
            from cad.operations.shell import shell_body
            from viewer.mesh import Mesh
            try:
                return Mesh(shell_body(shape.wrapped, face_occs, thickness,
                                       outside)), None
            except Exception as ex:
                return None, str(ex)
        self._shell_job().submit(_compute)

    def _on_shell_preview_done(self, result, error):
        panel = getattr(self, '_shell_panel', None)
        if panel is None or result is None:
            return
        mesh, err = result
        if mesh is not None:
            self.makeCurrent()
            try:
                mesh.upload()
            except Exception:
                mesh = None
        self._shell_preview_mesh = mesh
        if err:
            print(f"[Shell] preview: {err}")
            for i in range(len(self._shell_face_indices)):
                panel.set_face_entry_error(i, err)
        self.update()

    # ------------------------------------------------------------------
    # Commit / edit
    # ------------------------------------------------------------------

    def _on_shell_ok(self, thickness: float):
        panel = getattr(self, '_shell_panel', None)
        body_id = getattr(self, '_shell_body_id', None)
        face_indices = list(getattr(self, '_shell_face_indices', []))
        outside = panel.is_outside() if panel else False
        from gui.expr_spinbox import exprs_extra
        extra = exprs_extra(thickness=panel._spinbox if panel else None)
        editing_idx = getattr(self, '_editing_shell_idx', None)
        self._editing_shell_idx = None          # don't cancel the edit on close
        self._close_shell_panel()
        if body_id is None:
            print("[Shell] Pick a face of the body to shell.")
            return
        from cad.op_shell import ShellOp
        op = ShellOp(source_body_id=body_id, face_indices=face_indices,
                     thickness=thickness, outside=outside)
        if editing_idx is not None:
            self._replace_entry_group(editing_idx, body_id)
        op.commit_async(self, extra)

    def _replace_entry_group(self, idx: int, body_id: str):
        """Drop history entry *idx* (and split children it made) so its
        edited version can be committed in its place."""
        entries = self.history.entries
        if idx >= len(entries):
            return
        entry = entries[idx]
        entry.editing = False
        group = [idx] + [j for j in range(idx + 1, len(entries))
                         if entries[j].params.get("source_entry_id") == entry.entry_id]
        ids = {entries[j].entry_id for j in group}
        made = {entries[j].body_id for j in group
                if (b := self.workspace.bodies.get(entries[j].body_id)) is not None
                and b.created_at_entry_id in ids}
        self.history.seek(max(idx - 1, 0))
        for j in reversed(group):
            self.history.delete(j)
        for bid in made:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

    def reopen_shell(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation != "shell" or entry.op is None:
            return
        entry.editing = True
        entry.op.reopen(self, history_idx)
        self._editing_shell_idx = history_idx

    def _cancel_shell_edit(self):
        idx = getattr(self, '_editing_shell_idx', None)
        self._editing_shell_idx = None
        if idx is None:
            return
        entries = self.history.entries
        if idx < len(entries):
            entries[idx].editing = False
        self.history.seek(idx)
        body_id = getattr(self, '_shell_body_id', None)
        if body_id is not None:
            self._rebuild_body_mesh(body_id)
        self.history_changed.emit()
