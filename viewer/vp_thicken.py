"""
viewer/vp_thicken.py

ThickenMixin — uniform body offset panel, preview, and commit.
"""

from __future__ import annotations


class ThickenMixin:

    def _try_thicken(self):
        faces = self.selection.faces
        if faces:
            body_id = faces[0].body_id
            if any(f.body_id != body_id for f in faces):
                print("[Thicken] All selected faces must be on the same body."); return
            if self.workspace.current_shape(body_id) is None:
                print("[Thicken] Active body has no shape."); return
            face_indices = [f.face_idx for f in faces]
        else:
            body_id      = None
            face_indices = []
        self._show_thicken_panel(body_id, face_indices)

    def _show_thicken_panel(self, body_id: str | None, face_indices: list, editing_entry=None):
        from gui.thicken_panel import ThickenPanel
        if getattr(self, '_thicken_panel', None) is not None:
            old = self._thicken_panel
            old._preview_timer.stop()
            try:
                old.preview_changed.disconnect(self._update_thicken_preview)
            except Exception:
                pass
            old.close()
            self._thicken_panel = None

        self._thicken_body_id        = body_id
        self._thicken_face_indices   = list(face_indices)
        self._thicken_preview_mesh   = None
        panel = ThickenPanel(parent=self)
        panel.thicken_requested.connect(self._on_thicken_ok)
        panel.cancelled.connect(self._close_thicken_panel)
        panel.preview_changed.connect(self._update_thicken_preview)
        panel.face_entry_removed.connect(self._on_thicken_face_removed)

        # Populate initial face entries
        shape = self.workspace.current_shape(body_id)
        all_faces = list(shape.faces()) if shape is not None else []
        for fi in face_indices:
            label = self._thicken_face_label(body_id, fi, all_faces)
            panel.add_face_entry(body_id, fi, label)

        self._thicken_panel = panel
        self._position_thicken_panel()
        panel.show()
        panel.setFocus()
        panel._emit_preview()  # triggers _update_thicken_preview → wire build

    def _thicken_face_label(self, body_id: str, face_idx: int, all_faces: list) -> str:
        body = self.workspace.bodies.get(body_id)
        body_name = body.name if body else body_id
        return f"{body_name} · face {face_idx}"

    def _position_thicken_panel(self):
        p = getattr(self, '_thicken_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_thicken_panel(self):
        panel = getattr(self, '_thicken_panel', None)
        if panel is not None:
            panel._preview_timer.stop()
            panel.end_pick_face()
            try:
                panel.preview_changed.disconnect(self._update_thicken_preview)
            except Exception:
                pass
            try:
                panel.face_entry_removed.disconnect(self._on_thicken_face_removed)
            except Exception:
                pass
            panel.close()
            self._thicken_panel = None
        if getattr(self, '_editing_thicken_idx', None) is not None:
            self._cancel_thicken_edit()
        self._thicken_preview_mesh = None
        self._thicken_arrow_origin = None
        self._thicken_arrow_dir    = None
        self.update()

    # ------------------------------------------------------------------
    # Face picking for thicken panel
    # ------------------------------------------------------------------

    def _on_thicken_face_removed(self, index: int):
        indices = getattr(self, '_thicken_face_indices', [])
        if 0 <= index < len(indices):
            indices.pop(index)
            self._thicken_face_indices = indices
        self._thicken_preview_mesh = None
        panel = getattr(self, '_thicken_panel', None)
        if panel is not None:
            panel._emit_preview()
        self.update()

    def route_face_pick_for_thicken(self, body_id: str, face_idx: int) -> bool:
        """
        Called by the viewport face-pick path when a thicken panel is open
        and the pick_face button is active.  Returns True if consumed.
        """
        panel = getattr(self, '_thicken_panel', None)
        if panel is None or not panel._picking_face:
            return False

        # Thicken only supports a single body.
        # If none locked yet (panel opened with no selection), lock to first picked body.
        thicken_body = getattr(self, '_thicken_body_id', None)
        if thicken_body is None:
            self._thicken_body_id = body_id
            thicken_body = body_id
        elif body_id != thicken_body:
            print("[Thicken] All faces must be on the same body.")
            return True

        indices = getattr(self, '_thicken_face_indices', [])

        if face_idx in indices:
            # Toggle off — remove
            idx = indices.index(face_idx)
            panel.remove_face_entry(idx)   # emits face_entry_removed → _on_thicken_face_removed
        else:
            indices.append(face_idx)
            self._thicken_face_indices = indices
            shape = self.workspace.current_shape(body_id)
            all_faces = list(shape.faces()) if shape is not None else []
            label = self._thicken_face_label(body_id, face_idx, all_faces)
            panel.add_face_entry(body_id, face_idx, label)
            panel._emit_preview()

        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_thicken_preview(self, thickness: float):
        body_id      = getattr(self, '_thicken_body_id', None)
        face_indices = getattr(self, '_thicken_face_indices', None)
        if body_id is None or not face_indices:
            self._thicken_preview_mesh = None
            self._thicken_arrow_origin = None
            self._thicken_arrow_dir    = None
            self.update(); return
        shape = self.workspace.current_shape(body_id)
        if shape is None:
            self._thicken_preview_mesh = None
            self._thicken_arrow_origin = None
            self._thicken_arrow_dir    = None
            self.update(); return

        all_faces = list(shape.faces())
        face_occs = [all_faces[idx].wrapped for idx in face_indices
                     if idx < len(all_faces)]
        if not face_occs:
            self._thicken_preview_mesh = None
            self._thicken_arrow_origin = None
            self._thicken_arrow_dir    = None
            self.update(); return

        self._update_thicken_arrow(all_faces, face_indices, thickness)
        self._thicken_preview_dist = thickness

        panel = getattr(self, '_thicken_panel', None)
        if panel is not None:
            panel.clear_face_errors()

        if thickness == 0.0:
            self._thicken_preview_mesh = None
            self.update(); return

        from cad.operations.thicken import thicken_face_preview
        slabs = []
        for i, fo in enumerate(face_occs):
            try:
                slabs.append(thicken_face_preview(fo, thickness))
            except Exception as ex:
                if panel is not None:
                    panel.set_face_entry_error(i, str(ex))
        self._thicken_preview_mesh = slabs if slabs else None
        self.update()

    def _draw_thicken_preview(self):
        solids = getattr(self, '_thicken_preview_mesh', None)
        if solids:
            from OCP.BRep import BRep_Tool
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
            from OCP.TopoDS import TopoDS
            from OCP.TopLoc import TopLoc_Location
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            from OCP.GCPnts import GCPnts_UniformAbscissa
            from OpenGL.GL import (glDisable, glEnable, glColor4f, glBegin, glEnd,
                                   glVertex3f, glLineWidth, glBlendFunc,
                                   GL_LIGHTING, GL_DEPTH_TEST, GL_CULL_FACE,
                                   GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                                   GL_TRIANGLES, GL_LINE_STRIP)
            from cad.prefs import prefs as _prefs
            r, g, b = _prefs.op_preview_color
            op = _prefs.op_preview_opacity
            fill_color = (r, g, b, op)
            edge_color = (min(r+0.23, 1.0), min(g+0.30, 1.0), min(b+0.15, 1.0), min(op+0.35, 1.0))

            glDisable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)
            glDisable(GL_CULL_FACE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            for solid in solids:
                try:
                    wrapped = solid.wrapped
                    BRepMesh_IncrementalMesh(wrapped, 0.15)
                    _identity = TopLoc_Location()
                    glColor4f(*fill_color)
                    exp = TopExp_Explorer(wrapped, TopAbs_FACE)
                    while exp.More():
                        face = TopoDS.Face_s(exp.Current())
                        tri  = BRep_Tool.Triangulation_s(face, _identity)
                        if tri is not None:
                            glBegin(GL_TRIANGLES)
                            for i in range(1, tri.NbTriangles() + 1):
                                n1, n2, n3 = tri.Triangle(i).Get()
                                for ni in (n1, n2, n3):
                                    p = tri.Node(ni)
                                    glVertex3f(p.X(), p.Y(), p.Z())
                            glEnd()
                        exp.Next()
                    glColor4f(*edge_color)
                    glLineWidth(1.4)
                    exp2 = TopExp_Explorer(wrapped, TopAbs_EDGE)
                    while exp2.More():
                        edge = exp2.Current()
                        try:
                            adp  = BRepAdaptor_Curve(edge)
                            disc = GCPnts_UniformAbscissa()
                            disc.Initialize(adp, 24)
                            if disc.IsDone() and disc.NbPoints() >= 2:
                                glBegin(GL_LINE_STRIP)
                                for pi in range(1, disc.NbPoints() + 1):
                                    p = adp.Value(disc.Parameter(pi))
                                    glVertex3f(p.X(), p.Y(), p.Z())
                                glEnd()
                        except Exception:
                            pass
                        exp2.Next()
                    glLineWidth(1.0)
                except Exception as ex:
                    print(f"[Thicken preview draw] {ex}")

        self._draw_thicken_arrow()

    def _update_thicken_arrow(self, all_faces, face_indices, thickness: float):
        """Compute centroid + average normal of selected faces for arrow placement."""
        import numpy as np
        from OCP.BRep import BRep_Tool
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepGProp import BRepGProp_Face
        from OCP.GProp import GProp_GProps
        from OCP.gp import gp_Pnt2d

        centroids = []
        normals   = []
        for idx in face_indices:
            if idx >= len(all_faces):
                continue
            try:
                face_occ = all_faces[idx].wrapped
                # Area-weighted centroid — works for any surface type
                props = GProp_GProps()
                BRepGProp.SurfaceProperties_s(face_occ, props)
                cog = props.CentreOfMass()
                centroids.append(np.array([cog.X(), cog.Y(), cog.Z()], dtype=float))
                # Normal at the surface centroid via UV evaluation
                surf_props = BRepGProp_Face(face_occ)
                umin, umax, vmin, vmax = surf_props.Bounds()
                u_mid = (umin + umax) * 0.5
                v_mid = (vmin + vmax) * 0.5
                pt    = gp_Pnt2d(u_mid, v_mid)
                from OCP.gp import gp_Pnt, gp_Vec
                sampled = []
                for uf in (0.25, 0.5, 0.75):
                    for vf in (0.25, 0.5, 0.75):
                        try:
                            pt2 = gp_Pnt()
                            nv2 = gp_Vec()
                            surf_props.Normal(
                                umin + uf*(umax-umin),
                                vmin + vf*(vmax-vmin),
                                pt2, nv2,
                            )
                            nv = np.array([nv2.X(), nv2.Y(), nv2.Z()], dtype=float)
                            if np.linalg.norm(nv) > 1e-10:
                                sampled.append(nv)
                        except Exception:
                            pass
                if sampled:
                    normals.append(np.mean(sampled, axis=0))
            except Exception:
                pass

        if not centroids or not normals:
            self._thicken_arrow_origin = None
            self._thicken_arrow_dir    = None
            return

        centroid = np.mean(centroids, axis=0)
        avg_normal = np.mean(normals, axis=0)
        n = np.linalg.norm(avg_normal)
        if n < 1e-10:
            self._thicken_arrow_origin = None
            self._thicken_arrow_dir    = None
            return
        avg_normal /= n

        # Arrow direction follows thickness sign; base sits at the offset surface.
        sign = 1.0 if thickness >= 0 else -1.0
        arrow_dir = avg_normal * sign
        self._thicken_arrow_origin = centroid + arrow_dir * abs(thickness)
        self._thicken_arrow_dir    = arrow_dir

    def _draw_thicken_arrow(self):
        import numpy as np
        from viewer.drag_arrow import DragArrow

        origin    = getattr(self, '_thicken_arrow_origin', None)
        direction = getattr(self, '_thicken_arrow_dir',    None)
        if origin is None or direction is None:
            return

        thickness = getattr(self, '_thicken_preview_dist', 0.0)
        is_cut    = thickness < 0
        color     = (0.95, 0.25, 0.25) if is_cut else (0.95, 0.85, 0.15)
        scale     = self.camera.distance * 0.10
        scale     = max(scale, abs(thickness) * 0.18) if thickness != 0.0 else scale

        DragArrow().draw(origin, direction, scale, color=color)

    def _on_thicken_ok(self, thickness: float):
        body_id      = getattr(self, '_thicken_body_id', None)
        face_indices = getattr(self, '_thicken_face_indices', None)
        editing_idx  = getattr(self, '_editing_thicken_idx', None)
        if editing_idx is not None:
            self._editing_thicken_idx = None  # prevent cancel on close
        self._close_thicken_panel()
        if body_id is None or not face_indices:
            return
        from cad.op_types import ThickenOp
        if editing_idx is not None:
            self._editing_thicken_idx = editing_idx  # restore for _commit
            self._commit_thicken_edit(body_id, face_indices, thickness)
            return
        ThickenOp(source_body_id=body_id, face_indices=face_indices, thickness=thickness).commit_async(self)

    def _commit_thicken_edit(self, body_id: str, face_indices: list, thickness: float):
        from cad.op_types import ThickenOp

        idx = getattr(self, '_editing_thicken_idx', None)
        if idx is None:
            return
        self._editing_thicken_idx = None

        entries = self.history.entries
        if idx >= len(entries):
            return

        entry = entries[idx]
        entry.editing = False

        entry_id      = entry.entry_id
        group_indices = [idx]
        group_body_ids = {entry.body_id}
        for j in range(idx + 1, len(entries)):
            if entries[j].params.get("source_entry_id") == entry_id:
                group_indices.append(j)
                group_body_ids.add(entries[j].body_id)

        self.history.seek(max(idx - 1, 0))

        if self.workspace.current_shape(body_id) is None:
            print(f"[Edit] Cannot commit thicken: source body '{body_id}' has no "
                  f"valid shape at this point. Fix upstream errors first.")
            self.history.seek(idx)
            entry.editing = False
            self._rebuild_body_mesh(body_id)
            self.history_changed.emit()
            return

        # Only remove bodies *created* by this op group (split children).
        # The source body and any unrelated bodies in group_body_ids must
        # stay — they existed before this op.
        group_entry_ids = {entries[j].entry_id for j in group_indices}
        removable_bodies = set()
        for bid in group_body_ids:
            body = self.workspace.bodies.get(bid)
            if body is not None and body.created_at_entry_id in group_entry_ids:
                removable_bodies.add(bid)
        for j in reversed(group_indices):
            self.history.delete(j)
        for bid in removable_bodies:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

        ThickenOp(source_body_id=body_id, face_indices=face_indices,
                  thickness=thickness).commit_async(self)

    def reopen_thicken(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation != "thicken" or entry.op is None:
            return
        entry.editing = True
        self._editing_thicken_idx     = history_idx
        self._thicken_face_indices    = list(entry.op.face_indices)
        entry.op.reopen(self, history_idx)

    def _cancel_thicken_edit(self):
        idx = getattr(self, '_editing_thicken_idx', None)
        self._editing_thicken_idx = None
        if idx is None:
            return
        entries = self.history.entries
        if idx < len(entries):
            entries[idx].editing = False
        self.history.seek(idx)
        body_id = getattr(self, '_thicken_body_id', None)
        if body_id is not None:
            self._rebuild_body_mesh(body_id)
        self.history_changed.emit()
