"""
viewer/vp_revolve.py

RevolveMixin — panel management, pick routing, and geometry dispatch for
the revolve operation.

Expects self to have:
  history, workspace, _meshes, _sketch_faces, _selected_sketch_entry,
  _selected_sketch_face, selection,
  _rebuild_body_mesh(), _rebuild_bodies(), _post_push_cascade(),
  history_changed signal.
"""

from __future__ import annotations

class RevolveMixin:

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _try_revolve(self):
        if self._selected_sketch_entry is not None:
            self._show_revolve_panel(sketch_idx=self._selected_sketch_entry)
        elif self.selection.face_count > 0:
            sf = self.selection.single_face or self.selection.faces[0]
            self._show_revolve_panel(body_id=sf.body_id, face_idx=sf.face_idx)
        else:
            self._show_revolve_panel()
        # Arm the next empty field: the axis when a profile came pre-filled.
        p = self._revolve_panel
        if p._face_list.has_valid_entries:
            self._arm_pick(p._pick_axis_btn, p._on_pick_axis_toggle)
        else:
            self._arm_pick(p._pick_face_btn, p._on_pick_face_toggle)

    def _show_revolve_panel(self, sketch_idx: int | None = None,
                             body_id: str | None = None,
                             face_idx: int | None = None,
                             editing_entry=None):
        from gui.revolve_panel import RevolvePanel
        if hasattr(self, '_revolve_panel') and self._revolve_panel is not None:
            self._revolve_panel.close()

        self._revolve_sketch_idx = sketch_idx
        self._revolve_face_pairs = ([(body_id, face_idx)]
                                    if body_id is not None and face_idx is not None
                                    else [])

        panel = RevolvePanel(self.workspace, parent=self)

        panel.revolve_requested.connect(self._on_revolve_panel_ok)
        panel.cancelled.connect(self._close_revolve_panel)
        panel.face_entry_removed.connect(self._on_revolve_face_removed)
        panel.picking_axis_changed.connect(self._on_revolve_pick_axis)
        panel.picking_face_changed.connect(self._on_revolve_pick_face)
        panel.picking_body_changed.connect(self._on_revolve_pick_body)
        panel.preview_changed.connect(self._on_revolve_preview)

        self._revolve_panel          = panel
        self._revolve_axis_active    = False
        self._revolve_axis_ref       = None   # (sketch entry_id, line uid) of a sketch-line axis
        self._revolve_axis_edge      = None   # (edge source dict, pt, dir) of a body-edge axis
        self._revolve_face_active    = False
        self._revolve_body_active    = False
        self._revolve_preview_mesh   = None
        self._refresh_revolve_rows()

        self._position_revolve_panel()
        panel.show()
        panel.setFocus()

    def _position_revolve_panel(self):
        p = getattr(self, '_revolve_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_revolve_panel(self):
        if hasattr(self, '_revolve_panel') and self._revolve_panel is not None:
            panel = self._revolve_panel
            # Disconnect preview signal BEFORE close — closing the widget
            # can fire one last spinbox value_changed → preview_changed,
            # which would otherwise repopulate the arrow/mesh after we
            # cleared them below.
            try:
                panel.preview_changed.disconnect(self._on_revolve_preview)
            except (TypeError, RuntimeError):
                pass
            try:
                panel._preview_timer.stop()
            except Exception:
                pass
            panel.close()
            panel.deleteLater()
            self._revolve_panel = None
        self._revolve_axis_active   = False
        self._revolve_face_active   = False
        self._revolve_body_active   = False
        self._revolve_preview_mesh  = None
        self._revolve_cage_wires    = None
        self._revolve_cage_key      = None
        self._revolve_arrow_origin  = None
        self._revolve_arrow_dir     = None
        self._revolve_face_centroid = None
        self._revolve_axis_point    = None
        self._revolve_axis_dir      = None
        self._revolve_preview_angle = 0.0
        self._revolve_last_preview_params = None
        # Discard any in-flight mesh compute — a worker that started during
        # the last spin must not revive the preview after the panel closed.
        self._revolve_job().cancel()
        if getattr(self, '_editing_history_idx', None) is not None:
            self._cancel_revolve_edit()
        self.update()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_revolve_preview(self, angle: float, axis_point, axis_dir):
        import numpy as np

        if axis_point is None or axis_dir is None or angle == 0:
            self._revolve_cage_wires   = None
            self._revolve_arrow_origin = None
            self._revolve_arrow_dir    = None
            self.update()
            return

        axis_pt = np.array(axis_point, dtype=float)
        axis_d  = np.array(axis_dir,   dtype=float)
        axis_d /= np.linalg.norm(axis_d)

        sketch_idx = getattr(self, '_revolve_sketch_idx', None)
        face_pairs = getattr(self, '_revolve_face_pairs', [])

        faces = self._selection_faces(sketch_idx, face_pairs)

        if not faces:
            self._revolve_cage_wires   = None
            self._revolve_arrow_origin = None
            self._revolve_arrow_dir    = None
            self.update()
            return

        panel = getattr(self, '_revolve_panel', None)
        side = panel.axis_side() if panel is not None else 1
        self._revolve_side = side

        # Re-extract cage wires only when axis, faces or side change, not
        # every angle tick.  An axis through a profile revolves one side of
        # it only (cad/operations/revolve.py profile_side): the cage, arrow
        # and side checkbox follow that half.
        cache_key = (tuple(axis_pt), tuple(axis_d), tuple(id(f) for f in faces), side)
        if cache_key != getattr(self, '_revolve_cage_key', None):
            from cad.operations.revolve import profile_side, split_face_at_axis
            crossing = False
            rev_faces = []
            for f in faces:
                try:
                    crossing |= split_face_at_axis(f, axis_pt, axis_d) is not None
                    rev_faces += profile_side(f, axis_pt, axis_d, side)
                except Exception as ex:
                    print(f"[Revolve preview] {ex}")
                    rev_faces.append(f)
            if panel is not None:
                panel.set_side_visible(crossing)
            self._revolve_rev_faces  = rev_faces
            self._revolve_cage_wires = _extract_face_wires(rev_faces)
            self._revolve_cage_key   = cache_key
            self._revolve_preview_mesh = None  # axis/face changed — old mesh is wrong

        # Store for use by mesh compute on release.
        self._revolve_last_preview_params = (faces, axis_pt, axis_d, angle, side)

        self._update_revolve_arrow(self._revolve_rev_faces or faces, axis_pt, axis_d, angle)

        # During arrow drag show cage only. When not dragging (spinbox, axis pick)
        # kick off a mesh compute so the user sees a proper solid preview.
        if not (getattr(self, '_drag_arrow_active', False)
                and getattr(self, '_drag_arrow_op', None) == 'revolve'):
            self._revolve_compute_mesh(faces, axis_pt, axis_d, angle, side)

        self.update()

    def _revolve_job(self):
        from viewer.preview_job import job_for
        return job_for(self, '_revolve_preview_job', self._on_revolve_mesh_done,
                       "Revolve preview")

    def _revolve_compute_mesh(self, faces, axis_pt, axis_d, angle, side=1):
        """Build the preview mesh in the background (one worker at a time,
        latest params win — see viewer/preview_job.py)."""
        def _compute():
            from viewer.preview_job import revolve_preview_meshes
            return revolve_preview_meshes(faces, axis_pt, axis_d, angle, side)
        self._revolve_job().submit(_compute)

    def _on_revolve_mesh_done(self, result, error):
        if getattr(self, '_revolve_panel', None) is None:
            return
        self._revolve_preview_mesh = result
        self.update()

    def _update_revolve_arrow(self, faces, axis_pt, axis_d, angle_deg: float):
        """Compute arrow position at the swept tip of the revolve."""
        import numpy as np
        import math

        # Centroid of the first face (world space)
        try:
            from build123d import Plane
            pl = Plane(faces[0])
            o  = pl.origin
            centroid = np.array([o.X, o.Y, o.Z], dtype=float)
        except Exception:
            self._revolve_arrow_origin = None
            self._revolve_arrow_dir    = None
            return

        # Store for incremental drag updates
        self._revolve_face_centroid = centroid
        self._revolve_axis_point    = axis_pt.copy()
        self._revolve_axis_dir      = axis_d.copy()
        self._revolve_preview_angle = angle_deg

        # Radial vector: from axis to centroid, perpendicular to axis
        rel     = centroid - axis_pt
        rel_ax  = np.dot(rel, axis_d) * axis_d   # component along axis
        r_perp  = rel - rel_ax                    # component perp to axis
        r       = np.linalg.norm(r_perp)
        if r < 1e-6:
            # Centroid on the axis (a profile the axis runs through, revolved
            # half by half): place the arrow off one half instead.
            try:
                from cad.operations.revolve import split_face_at_axis
                halves = split_face_at_axis(faces[0], axis_pt, axis_d)
                if halves is not None:
                    side = getattr(self, '_revolve_side', 1)
                    c = (halves[0] if side >= 0 else halves[1])[0].center()
                    centroid = np.array([c.X, c.Y, c.Z], dtype=float)
                    self._revolve_face_centroid = centroid
                    rel    = centroid - axis_pt
                    rel_ax = np.dot(rel, axis_d) * axis_d
                    r_perp = rel - rel_ax
                    r      = np.linalg.norm(r_perp)
            except Exception as ex:
                print(f"[Revolve] arrow: {ex}")
        if r < 1e-6:
            self._revolve_arrow_origin = None
            self._revolve_arrow_dir    = None
            return
        r_unit = r_perp / r

        # Tangential direction at angle=0 (cross of axis with radial)
        tan0 = np.cross(axis_d, r_unit)
        tn   = np.linalg.norm(tan0)
        if tn < 1e-10:
            self._revolve_arrow_origin = None
            self._revolve_arrow_dir    = None
            return
        tan0 /= tn

        # Rotate r_unit and tan0 by angle_deg around the axis
        a   = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        # Rotated radial = c*r_unit + s*tan0
        r_rot   = c * r_unit + s * tan0
        # Rotated tangent = -s*r_unit + c*tan0
        tan_rot = -s * r_unit + c * tan0

        # Arrow sits at the swept face centroid
        tip = axis_pt + rel_ax + r * r_rot
        self._revolve_arrow_origin = tip
        # Arrow points in the tangential direction (along the sweep)
        self._revolve_arrow_dir    = tan_rot

    def _draw_revolve_preview(self):
        from OpenGL.GL import (glDisable, glEnable, glBlendFunc,
                               GL_LIGHTING, GL_DEPTH_TEST, GL_CULL_FACE,
                               GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        from cad.prefs import prefs as _prefs
        r, g, b = _prefs.op_preview_color
        op = _prefs.op_preview_opacity
        edge_color = (min(r+0.23, 1.0), min(g+0.30, 1.0), min(b+0.15, 1.0), min(op+0.35, 1.0))

        glDisable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        mesh_data = getattr(self, '_revolve_preview_mesh', None)
        if mesh_data:
            from viewer.preview_job import draw_preview
            draw_preview(mesh_data, (r, g, b, op), edge_color)
        else:
            self._draw_revolve_cage(edge_color)

        self._draw_revolve_arrow()

    def _draw_revolve_cage(self, edge_color):
        """Instant wireframe cage: profile at 0°, profile at angle°, arc lines
        per vertex. Pure python math — no OCCT, no tessellation."""
        import math
        import numpy as np
        from OpenGL.GL import (glColor4f, glBegin, glEnd, glVertex3f,
                               glLineWidth, GL_LINE_STRIP)

        wires    = getattr(self, '_revolve_cage_wires', None)
        angle    = getattr(self, '_revolve_preview_angle', 0.0)
        axis_pt  = getattr(self, '_revolve_axis_point',   None)
        axis_d   = getattr(self, '_revolve_axis_dir',     None)
        if not wires or axis_pt is None or axis_d is None or angle == 0:
            return

        ap = np.asarray(axis_pt, dtype=float)
        ad = np.asarray(axis_d,  dtype=float)
        a  = math.radians(angle)

        def rotate_pt(p):
            """Rodrigues rotation of point p around axis by angle a."""
            v   = p - ap
            par = np.dot(v, ad) * ad
            perp = v - par
            if np.linalg.norm(perp) < 1e-12:
                return p
            perp_rot = (math.cos(a) * perp
                        + math.sin(a) * np.cross(ad, perp))
            return ap + par + perp_rot

        ARC_STEPS = 12

        glColor4f(*edge_color)
        glLineWidth(1.2)

        for wire in wires:
            # Profile at angle=0
            if len(wire) >= 2:
                glBegin(GL_LINE_STRIP)
                for p in wire:
                    glVertex3f(*p)
                glEnd()

            # Profile at current angle (rotate each point)
            rotated = [rotate_pt(np.asarray(p)) for p in wire]
            if len(rotated) >= 2:
                glBegin(GL_LINE_STRIP)
                for p in rotated:
                    glVertex3f(float(p[0]), float(p[1]), float(p[2]))
                glEnd()

            # Arc lines from each wire vertex, sweeping 0→angle
            for p in wire:
                pv   = np.asarray(p, dtype=float)
                v    = pv - ap
                par  = np.dot(v, ad) * ad
                perp = v - par
                r    = np.linalg.norm(perp)
                if r < 1e-10:
                    continue
                glBegin(GL_LINE_STRIP)
                for i in range(ARC_STEPS + 1):
                    t = a * i / ARC_STEPS
                    perp_rot = (math.cos(t) * perp
                                + math.sin(t) * np.cross(ad, perp))
                    q = ap + par + perp_rot
                    glVertex3f(float(q[0]), float(q[1]), float(q[2]))
                glEnd()

        glLineWidth(1.0)

    def _draw_revolve_arrow(self):
        from viewer.drag_arrow import DragArrow
        origin    = getattr(self, '_revolve_arrow_origin', None)
        direction = getattr(self, '_revolve_arrow_dir',    None)
        if origin is None or direction is None:
            return
        angle = getattr(self, '_revolve_preview_angle', 0.0)
        scale = self.camera.distance * 0.10
        scale = max(scale, angle * 0.005) if angle != 0.0 else scale
        DragArrow().draw(origin, direction, scale, color=(0.95, 0.85, 0.15))

    # ------------------------------------------------------------------
    # Pick routing signals
    # ------------------------------------------------------------------

    def _on_revolve_pick_axis(self, active: bool):
        self._revolve_axis_active = active

    def _on_revolve_pick_face(self, active: bool):
        self._revolve_face_active = active

    def _on_revolve_pick_body(self, active: bool):
        self._revolve_body_active = active

    # ------------------------------------------------------------------
    # Incoming picks from mousePressEvent
    # ------------------------------------------------------------------

    def route_sketch_edge_pick_for_revolve(self, history_idx: int,
                                            entity_idx: int) -> bool:
        """
        Called when the user clicks a committed sketch edge while axis-pick
        mode is active.  Extracts the line's world-space endpoints and passes
        the axis to the panel.
        """
        if not getattr(self, '_revolve_axis_active', False):
            return False
        panel = getattr(self, '_revolve_panel', None)
        if panel is None:
            return False

        import numpy as np
        try:
            entries = self.history.entries
            if history_idx >= len(entries):
                return False
            se = entries[history_idx].params.get("sketch_entry")
            if se is None:
                return False
            # Any straight line: drawn, or included (a straight reference
            # polyline) — the same test replay uses to follow the axis.
            from cad.op_revolve_thicken import _axis_line_ends
            ent = se.entities[entity_idx]
            ends = _axis_line_ends(se, getattr(ent, "uid", None))
            if ends is None:
                print("[Revolve] Axis pick requires a straight line.")
                return False
            a, b = ends

            origin = np.array(se.plane_origin,  dtype=float)
            x_axis = np.array(se.plane_x_axis,  dtype=float)
            y_axis = np.array(se.plane_y_axis,  dtype=float)

            p0 = origin + float(a[0]) * x_axis + float(a[1]) * y_axis
            p1 = origin + float(b[0]) * x_axis + float(b[1]) * y_axis
            direction = p1 - p0
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                print("[Revolve] Degenerate sketch line — cannot use as axis.")
                return False
            direction /= norm

            # Remember the line so the op can follow it through sketch edits.
            self._revolve_axis_ref = (entries[history_idx].entry_id, ent.uid)
            panel.set_axis(p0, direction)
            return True
        except Exception as ex:
            print(f"[Revolve] Axis pick failed: {ex}")
            return False

    def route_edge_pick_for_revolve(self, edge_idx: int, body_id: str) -> bool:
        """
        Called when the user clicks a body edge while axis-pick mode is active.
        Extracts the edge tangent as the axis direction, midpoint as axis point.
        """
        if not getattr(self, '_revolve_axis_active', False):
            return False
        panel = getattr(self, '_revolve_panel', None)
        if panel is None:
            return False

        import numpy as np
        mesh = self._meshes.get(body_id)
        if mesh is None or edge_idx >= len(mesh.topo_edges_occ):
            return False
        from cad.op_revolve_thicken import edge_axis
        occ_edge = mesh.topo_edges_occ[edge_idx]
        try:
            ax = edge_axis(occ_edge)
            if ax is None:
                return False
            axis_pt, axis_dir = ax
        except Exception as ex:
            print(f"[Revolve] Edge axis failed: {ex}")
            return False

        # The picked edge, so replay can follow it (kept by the commit only
        # while the panel's axis is still this one).
        from viewer.vp_extrude import _picked_edge_source
        src = _picked_edge_source(self.history, body_id,
                                  self.workspace.current_shape(body_id), occ_edge)
        self._revolve_axis_edge = (None if src is None
                                   else (src, axis_pt.copy(), axis_dir.copy()))
        self._revolve_axis_ref = None
        panel.set_axis(axis_pt, axis_dir)
        return True

    def _refresh_revolve_rows(self):
        panel = getattr(self, '_revolve_panel', None)
        if panel is None:
            return
        panel.clear_face_entries()
        for bid, fi, label, valid in self._selection_rows(
                self._revolve_sketch_idx, self._revolve_face_pairs):
            panel.add_face_entry(bid, fi, label, valid=valid)

    def route_face_pick_for_revolve(self, body_id: str, face_idx: int) -> bool:
        """Toggle a body face in the selection (alongside any sketch loops)."""
        if not getattr(self, '_revolve_face_active', False):
            return False
        panel = getattr(self, '_revolve_panel', None)
        if panel is None or self.workspace.bodies.get(body_id) is None:
            return False
        pairs = list(getattr(self, '_revolve_face_pairs', []))
        pair = (body_id, face_idx)
        self._revolve_face_pairs = ([p for p in pairs if p != pair]
                                    if pair in pairs else pairs + [pair])
        self._refresh_revolve_rows()
        panel._emit_preview()
        return True

    def _on_revolve_face_removed(self, index: int):
        """Remove a profile row from the revolve panel's list."""
        self._revolve_sketch_idx, self._revolve_face_pairs = \
            self._remove_selection_row(index, getattr(self, '_revolve_sketch_idx', None),
                                       getattr(self, '_revolve_face_pairs', []))
        panel = getattr(self, '_revolve_panel', None)
        if panel is not None:
            panel._emit_preview()

    def route_sketch_face_pick_for_revolve(self, sketch_idx: int,
                                           face_idx: int) -> bool:
        """Toggle a sketch loop in the selection (alongside any body faces)
        while '+ Add Face' is active.  Loops of a different sketch replace
        the previous sketch's loops."""
        if not getattr(self, '_revolve_face_active', False):
            return False
        panel = getattr(self, '_revolve_panel', None)
        if panel is None:
            return False

        current_indices = list(self._selected_sketch_face or []) \
            if getattr(self, '_revolve_sketch_idx', None) == sketch_idx else []
        if face_idx in current_indices:
            current_indices.remove(face_idx)
        else:
            current_indices.append(face_idx)

        self._revolve_sketch_idx    = sketch_idx
        self._selected_sketch_entry = sketch_idx
        self._selected_sketch_face  = current_indices if current_indices else None
        self._refresh_revolve_rows()
        panel._emit_preview()
        return True

    def route_body_pick_for_revolve(self, body_id: str) -> bool:
        if not getattr(self, '_revolve_body_active', False):
            return False
        panel = getattr(self, '_revolve_panel', None)
        if panel is None:
            return False
        body = self.workspace.bodies.get(body_id)
        if body is None:
            return False
        panel.set_merge_body(body_id, body.name)
        return True

    # ------------------------------------------------------------------
    # Reopen (double-click history entry)
    # ------------------------------------------------------------------

    def reopen_revolve(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation not in ("revolve", "revolve_cut") or entry.op is None:
            return
        entry.editing = True
        self._editing_history_idx = history_idx
        self._editing_body_id     = entry.body_id
        entry.op.reopen(self, history_idx)

    def _cancel_revolve_edit(self):
        idx = getattr(self, '_editing_history_idx', None)
        if idx is None:
            return
        self._editing_history_idx = None
        entries = self.history.entries
        if idx < len(entries):
            entries[idx].editing = False
        self.history_changed.emit()

    # ------------------------------------------------------------------
    # OK handler
    # ------------------------------------------------------------------

    def _on_revolve_panel_ok(self, angle_deg: float, axis_point, axis_dir,
                              merge_body_id, is_cut: bool):
        sketch_idx  = getattr(self, '_revolve_sketch_idx', None)
        face_pairs  = getattr(self, '_revolve_face_pairs', [])
        editing_idx = getattr(self, '_editing_history_idx', None)

        axis_pt  = list(map(float, axis_point))
        axis_d   = list(map(float, axis_dir))
        angle    = float(angle_deg)

        # Parameter binding on the angle field — every commit path below
        # forwards it as extra_params (read before the panel closes).
        from gui.expr_spinbox import exprs_extra
        panel = getattr(self, '_revolve_panel', None)
        self._revolve_commit_extra = exprs_extra(
            angle_deg=panel._angle_spinbox if panel else None)
        side = panel.axis_side() if panel is not None else 1

        # Built before the panel closes (closing clears the sketch selection).
        profiles = self._selection_profiles(sketch_idx, face_pairs)

        # Prevent _close_revolve_panel from cancelling the edit
        if editing_idx is not None:
            self._editing_history_idx = None

        self._close_revolve_panel()

        if not profiles:
            print("[Revolve] No profile selected.")
            if editing_idx is not None:
                self._editing_history_idx = editing_idx
                self._cancel_revolve_edit()
            return

        # Editing path: defer all branching to _commit_revolve_edit so the
        # seek/delete/replay bookkeeping stays in one place.
        if editing_idx is not None:
            self._editing_history_idx = editing_idx  # restore for _commit
            self._commit_revolve_edit_from_panel(
                editing_idx, angle, axis_pt, axis_d, merge_body_id, is_cut,
                profiles, side)
            return

        # No-target cut: one CrossBodyRevolveCutOp per intersecting body.
        if is_cut and merge_body_id in (None, "__new_body__"):
            self._do_revolve_cut_all_intersecting(angle, axis_pt, axis_d, profiles,
                                                  side)
            return

        op = self._build_revolve_op(profiles, angle, axis_pt, axis_d,
                                    merge_body_id, is_cut, side)
        extra = getattr(self, '_revolve_commit_extra', None)
        if is_cut:
            op.commit(self, extra)
        else:
            op.commit_async(self, extra)

    @staticmethod
    def _build_revolve_op(profiles, angle, axis_pt, axis_d, merge_body_id, is_cut,
                          side=1):
        """CrossBodyRevolveCutOp into a chosen body, else a RevolveOp
        (merge_body_id None / "__new_body__" → new body)."""
        from cad.op_types import RevolveOp, CrossBodyRevolveCutOp
        common = dict(profiles=profiles, angle_deg=angle,
                      axis_point=list(axis_pt), axis_dir=list(axis_d),
                      axis_side=side)
        if is_cut:
            return CrossBodyRevolveCutOp(cut_body_id=merge_body_id, **common)
        new_body = merge_body_id in (None, "__new_body__")
        return RevolveOp(merge_body_id=None if new_body else merge_body_id,
                         **common)

    def _do_revolve_cut_all_intersecting(self, angle_deg, axis_pt, axis_d,
                                          profiles, side=1):
        """
        No-target revolve cut: build the revolved tool solid once, fan out
        one CrossBodyRevolveCutOp per workspace body it intersects.  A
        face-driven tool's source body goes last — cutting it first would
        change the face the later ops revolve.
        """
        import numpy as np
        from cad.op_types import CrossBodyRevolveCutOp
        from cad.op_revolve_thicken import _revolve_faces_to_tool
        from cad.profile_source import BodyFaceProfile
        from cad.cut_all import fan_out_cut

        try:
            tool_faces = []
            for p in profiles:
                tool_faces += p.resolve_live(self, "[Revolve cut]")
        except Exception as ex:
            print(f"[Revolve cut] {ex}"); return
        if not tool_faces:
            print("[Revolve cut] No tool faces."); return
        try:
            tool_solid = _revolve_faces_to_tool(
                tool_faces, np.asarray(axis_pt, dtype=float),
                np.asarray(axis_d, dtype=float), angle_deg, side)
        except Exception as ex:
            print(f"[Revolve cut] Tool construction failed: {ex}"); return
        if tool_solid is None:
            print("[Revolve cut] Tool construction failed."); return

        def build_op(bid: str):
            return CrossBodyRevolveCutOp(
                cut_body_id = bid,
                profiles    = [p.copy() for p in profiles],
                angle_deg   = angle_deg,
                axis_point  = list(axis_pt),
                axis_dir    = list(axis_d),
                axis_side   = side,
            )

        faces = [p for p in profiles if isinstance(p, BodyFaceProfile)]
        fan_out_cut(self, tool_solid, build_op,
                    getattr(self, '_revolve_commit_extra', None),
                    op_label="Revolve cut",
                    last=faces[0].body_id if faces else None)

    def _commit_revolve_edit_from_panel(self, editing_idx, angle, axis_pt,
                                         axis_d, merge_body_id, is_cut,
                                         profiles, side=1):
        """
        Bridge from panel OK to the edit-commit routine: builds the new op
        (or dispatches to fan-out for no-target cuts) so _commit_revolve_edit
        deletes the old entry group correctly in every mode.
        """
        # No-target / self cut: fan-out pushes per-body entries instead of
        # producing a single new_op.
        if is_cut and merge_body_id in (None, "__new_body__"):
            self._editing_history_idx = None
            self._delete_revolve_edit_group(editing_idx)
            self._do_revolve_cut_all_intersecting(angle, axis_pt, axis_d, profiles,
                                                  side)
            new_idx = self.history.cursor
            ok, err, _ = self.history.replay_all_from(new_idx + 1)
            if not ok:
                print(f"[Revolve Edit] Downstream replay failed: {err}")
            # Advance cursor to tip so bodies created downstream are visible.
            self.history.seek(len(self.history.entries) - 1)
            self._rebuild_all_meshes()
            self.history_changed.emit()
            return

        self._commit_revolve_edit(editing_idx, self._build_revolve_op(
            profiles, angle, axis_pt, axis_d, merge_body_id, is_cut, side))

    def _delete_revolve_edit_group(self, idx: int) -> list:
        """Seek to idx-1, delete the group, remove created bodies (shared
        bookkeeping used by both _commit_revolve_edit and the no-target cut
        edit path).  Returns the removed child body ids in child order, for
        the re-commit to reuse (see _add_child_bodies)."""
        entries = self.history.entries
        if idx >= len(entries):
            return []
        entry = entries[idx]
        entry.editing = False

        # Split children + fan-out siblings (a no-target cut is one group).
        group_indices  = self.history.op_group(idx)
        group_body_ids = {entries[j].body_id for j in group_indices}

        self.history.seek(max(idx - 1, 0))

        child_body_ids = entry.params.get("child_body_ids", [])
        group_entry_ids = {entries[j].entry_id for j in group_indices}
        removable = set()
        for bid in group_body_ids | set(child_body_ids):
            body = self.workspace.bodies.get(bid)
            if body is not None and body.created_at_entry_id in group_entry_ids:
                removable.add(bid)
        keep_id = (self.history.entries[min(group_indices)].entry_id, min(group_indices))
        for j in reversed(group_indices):
            self.history.delete(j)
        # The re-commit keeps the entry id: downstream refs name it (revolve /
        # extrude cap and chamfer face origins of planes and edge picks).
        self.history.reuse_entry_id(*keep_id)
        for bid in removable:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)
        return [bid for bid in child_body_ids if bid in removable]

    def _commit_revolve_edit(self, idx: int, new_op):
        self._editing_history_idx = None
        self._editing_body_id = None
        preserved = self._delete_revolve_edit_group(idx)

        # Re-commit into the same child body ids: downstream entries (a
        # subtract using this revolve as its tool) reference them by id.
        extra = dict(getattr(self, '_revolve_commit_extra', None) or {})
        if preserved:
            extra["_preserved_body_ids"] = preserved
        new_op.commit(self, extra or None)

        new_idx = self.history.cursor
        ok, err, _ = self.history.replay_all_from(new_idx + 1)
        if not ok:
            print(f"[Revolve Edit] Downstream replay failed: {err}")

        # Advance cursor to tip so bodies created downstream are visible.
        self.history.seek(len(self.history.entries) - 1)
        self._rebuild_all_meshes()
        self.history_changed.emit()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _extract_face_wires(faces) -> list:
    """Discretize the boundary edges of each face into point lists.
    Returns a list of polylines: [ [(x,y,z), ...], ... ]
    Called on the main thread — no revolve, just edge sampling."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_UniformAbscissa

    wires = []
    for face in faces:
        exp = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
        while exp.More():
            edge = TopoDS.Edge(exp.Current())
            try:
                adp  = BRepAdaptor_Curve(edge)
                disc = GCPnts_UniformAbscissa()
                disc.Initialize(adp, 24)
                if disc.IsDone() and disc.NbPoints() >= 2:
                    pts = []
                    for i in range(1, disc.NbPoints() + 1):
                        p = adp.Value(disc.Parameter(i))
                        pts.append((p.X(), p.Y(), p.Z()))
                    wires.append(pts)
            except Exception:
                pass
            exp.Next()
    return wires
