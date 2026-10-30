"""
viewer/vp_transform.py

TransformMixin — panel lifecycle, body / axis-edge pick routing, preview
and commit for the move / rotate (/ copy) op, and its gizmo: X/Y/Z drag
arrows at the moved body's centre (translate) or a ring around the
rotation axis (rotate) that write the panel's values as they're dragged.
"""

from __future__ import annotations
import math
import numpy as np

_AXES   = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_COLORS = ((0.92, 0.32, 0.32), (0.38, 0.82, 0.38), (0.38, 0.58, 0.97))
_RING   = (0.95, 0.85, 0.15)


class TransformMixin:

    def _try_transform(self):
        if getattr(self, '_transform_panel', None) is not None:
            return
        self._show_transform_panel()
        sel = self.selection
        picked = [f.body_id for f in sel.faces] + [e.body_id for e in sel.edges]
        solid = [b for b in self.workspace.bodies
                 if self.workspace.current_shape(b) is not None]
        target = picked[0] if picked else (solid[0] if len(solid) == 1 else None)
        if target is not None:
            self._set_transform_body(target)
        else:
            p = self._transform_panel
            self._arm_pick(p._pick_body_btn, lambda c: p._set_picking("body", c))

    def _show_transform_panel(self):
        from gui.transform_panel import TransformPanel
        if getattr(self, '_transform_panel', None) is not None:
            return
        panel = TransformPanel(self.workspace, parent=self)
        panel.transform_requested.connect(self._on_transform_ok)
        panel.cancelled.connect(self._close_transform_panel)
        panel.picking_body_changed.connect(
            lambda a: setattr(self, '_transform_body_pick_active', a))
        panel.picking_edge_changed.connect(
            lambda a: setattr(self, '_transform_edge_pick_active', a))
        panel.preview_changed.connect(self._on_transform_preview_live)
        self._transform_panel             = panel
        self._transform_body_id           = None
        self._transform_edge_ref          = None
        self._transform_body_pick_active  = False
        self._transform_edge_pick_active  = False
        self._transform_preview_mesh      = None
        self._transform_rot_axis          = None
        self._transform_drag              = None
        origin = self.mapToGlobal(self.rect().topLeft())
        panel.move(origin.x() + 16, origin.y() + 16)
        panel.show()
        panel.setFocus()

    def _close_transform_panel(self):
        p = getattr(self, '_transform_panel', None)
        self._transform_panel = None
        if p is not None:
            p.close()
        self._transform_body_pick_active = False
        self._transform_edge_pick_active = False
        self._transform_job().cancel()
        self._transform_preview_mesh     = None
        idx = getattr(self, '_editing_transform_idx', None)
        if idx is not None:                 # closed without OK while editing
            self._editing_transform_idx = None
            entries = self.history.entries
            if 0 <= idx < len(entries):
                entries[idx].editing = False
                self.history.seek(idx)
                self._rebuild_all_meshes()
            self.history_changed.emit()
        self.update()

    def _set_transform_body(self, body_id: str):
        body = self.workspace.bodies.get(body_id)
        panel = getattr(self, '_transform_panel', None)
        if body is None or panel is None:
            return
        self._transform_body_id  = body_id
        self._transform_edge_ref = None
        panel.set_body(body_id, body.name)

    # ------------------------------------------------------------------
    # Reopen / edit
    # ------------------------------------------------------------------

    def reopen_transform(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation != "transform" or entry.op is None:
            return
        entry.editing = True
        self._editing_transform_idx = history_idx
        op = entry.op
        if history_idx > 0:
            self.history.seek(history_idx - 1)
            self._rebuild_all_meshes()
        self.history_changed.emit()
        self._show_transform_panel()
        panel = self._transform_panel
        self._set_transform_body(op.source_body_id)
        panel.set_fields(op, entry.params.get("exprs", {}))
        if op.axis_edge_ref is not None:
            self._transform_edge_ref = op.axis_edge_ref
            panel.set_edge("edge (saved)")

    # ------------------------------------------------------------------
    # Pick routing
    # ------------------------------------------------------------------

    def route_body_pick_for_transform(self, body_id: str) -> bool:
        if not getattr(self, '_transform_body_pick_active', False):
            return False
        self._set_transform_body(body_id)
        return True

    def route_edge_pick_for_transform(self, edge_idx: int, body_id: str) -> bool:
        if not getattr(self, '_transform_edge_pick_active', False):
            return False
        panel = getattr(self, '_transform_panel', None)
        if panel is None:
            return False
        if body_id != self._transform_body_id:
            print("[Transform] Pick an edge of the body being moved.")
            return True
        from cad.edge_ref import EdgeRef
        from cad.op_transform import edge_axis
        mesh = self._meshes.get(body_id)
        shape = self.workspace.current_shape(body_id)
        if mesh is None or shape is None or edge_idx >= len(mesh.topo_edges_occ):
            return True
        occ = mesh.topo_edges_occ[edge_idx]
        if edge_axis(occ) is None:
            print("[Transform] The axis edge must be straight or circular.")
            return True
        self._transform_edge_ref = EdgeRef.for_edge(
            self.history, body_id, occ, shape.wrapped, self.history.cursor + 1)
        panel.set_edge(f"edge {edge_idx}")
        return True

    # ------------------------------------------------------------------
    # Preview (the moved body, ghosted)
    # ------------------------------------------------------------------

    def _transform_op(self, fields):
        from cad.op_transform import TransformOp
        f = dict(fields)
        if f.get("axis") == "edge":
            f["axis_edge_ref"] = self._transform_edge_ref
        return TransformOp(source_body_id=self._transform_body_id, **f)

    def _draw_transform_preview(self):
        if getattr(self, '_transform_panel', None) is not None:
            self._draw_transform_gizmo()
        if getattr(self, '_transform_preview_mesh', None) is None:
            return
        saved = (getattr(self, '_loft_preview_mesh', None),
                 getattr(self, '_loft_preview_is_cut', False))
        self._loft_preview_mesh, self._loft_preview_is_cut = \
            self._transform_preview_mesh, False
        try:
            self._draw_loft_preview()
        finally:
            self._loft_preview_mesh, self._loft_preview_is_cut = saved

    def _on_transform_preview_live(self):
        panel = getattr(self, '_transform_panel', None)
        if panel is None:
            return
        f = panel.fields()
        shape = (self.workspace.current_shape(self._transform_body_id)
                 if self._transform_body_id else None)
        if f is None or shape is None:
            self._transform_job().cancel()
            self._transform_preview_mesh = None
            self.update()
            return
        op, history = self._transform_op(f), self.history
        self._transform_rot_axis = None
        if f["mode"] == "rotate":
            try:
                self._transform_rot_axis = tuple(
                    np.asarray(v, float) for v in op._axis(history, shape))
            except Exception:
                pass

        def _compute():
            from viewer.preview_job import preview_mesh_data, pack_preview
            tris, edges = preview_mesh_data(op.apply(history, shape).wrapped, 0.15, 24)
            return pack_preview([tris], [edges])
        self._transform_job().submit(_compute)

    def _transform_job(self):
        from viewer.preview_job import job_for
        return job_for(self, '_transform_preview_job', self._on_transform_mesh_done,
                       "Transform preview")

    def _on_transform_mesh_done(self, result, error):
        if getattr(self, '_transform_panel', None) is None:
            return
        if error:
            print(f"[Transform preview] {error}")
        self._transform_preview_mesh = result
        self.update()

    # ------------------------------------------------------------------
    # Gizmo
    # ------------------------------------------------------------------

    def _transform_gizmo(self):
        """("translate", origin, scale) or ("rotate", centre, axis, radius,
        u, v) — the ring's plane basis — or None when there's nothing to drag."""
        panel = getattr(self, '_transform_panel', None)
        mesh = self._meshes.get(getattr(self, '_transform_body_id', None))
        if panel is None or mesh is None:
            return None
        c = (np.asarray(mesh.bbox_min, float) + np.asarray(mesh.bbox_max, float)) / 2
        scale = self.camera.distance * 0.10
        if panel._mode.currentData() == "translate":
            d = [b.mm_value() or 0.0 for b in (panel._dx, panel._dy, panel._dz)]
            return "translate", c + np.array(d), scale
        ax = getattr(self, '_transform_rot_axis', None)
        if ax is None:
            return None
        p, d = ax
        d = d / np.linalg.norm(d)
        centre = p + d * float(np.dot(c - p, d))
        u = np.cross(d, (1.0, 0.0, 0.0) if abs(d[0]) < 0.9 else (0.0, 1.0, 0.0))
        u /= np.linalg.norm(u)
        return "rotate", centre, d, scale * 1.4, u, np.cross(d, u)

    def _draw_transform_gizmo(self):
        g = self._transform_gizmo()
        if g is None:
            return
        if g[0] == "translate":
            from viewer.drag_arrow import DragArrow
            _m, origin, scale = g
            for a, col in zip(_AXES, _COLORS):
                DragArrow().draw(origin, np.array(a), scale, color=col)
            return
        from OpenGL.GL import (glDisable, glEnable, glColor4f, glLineWidth,
                               glBegin, glEnd, glVertex3f, glPointSize,
                               GL_DEPTH_TEST, GL_LIGHTING, GL_BLEND,
                               GL_LINE_LOOP, GL_POINTS)
        _m, centre, d, r, u, v = g
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glColor4f(*_RING, 0.9)
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        for t in np.linspace(0, 2 * math.pi, 64, endpoint=False):
            glVertex3f(*(float(x) for x in centre + r * (math.cos(t) * u + math.sin(t) * v)))
        glEnd()
        # Handle at the current angle; the axis through the centre.
        ang = math.radians(self._transform_panel._angle.mm_value() or 0.0)
        glPointSize(10.0)
        glBegin(GL_POINTS)
        glVertex3f(*(float(x) for x in centre + r * (math.cos(ang) * u + math.sin(ang) * v)))
        glEnd()
        glLineWidth(1.5)
        glColor4f(*_RING, 0.5)
        from OpenGL.GL import GL_LINES
        glBegin(GL_LINES)
        glVertex3f(*(float(x) for x in centre - d * r * 0.6))
        glVertex3f(*(float(x) for x in centre + d * r * 0.6))
        glEnd()
        glPointSize(1.0)
        glLineWidth(1.0)
        glEnable(GL_DEPTH_TEST)

    def _ring_angle(self, ray_o, ray_d, g):
        """Angle (degrees) round the ring where the ray meets its plane, and
        the hit's distance from the centre; None edge-on."""
        _m, centre, d, _r, u, v = g
        den = float(np.dot(ray_d, d))
        if abs(den) < 1e-6:
            return None
        hit = ray_o + ray_d * float(np.dot(centre - ray_o, d)) / den - centre
        return (math.degrees(math.atan2(float(hit @ v), float(hit @ u))),
                float(np.linalg.norm(hit)))

    def _hit_transform_gizmo(self, mx: int, my: int) -> bool:
        g = self._transform_gizmo()
        if g is None:
            return False
        ray_o, ray_d = self.get_ray(mx, my)
        if ray_o is None:
            return False
        panel = self._transform_panel
        if g[0] == "translate":
            from viewer.drag_arrow import DragArrow
            _m, origin, scale = g
            for k, (a, box) in enumerate(zip(_AXES, (panel._dx, panel._dy, panel._dz))):
                if DragArrow().hit_test(ray_o, ray_d, origin, np.array(a), scale) is None:
                    continue
                s = self._closest_point_on_axis(ray_o, ray_d, origin, np.array(a))
                if s is None:
                    return False
                self._transform_drag = (k, origin, s, box.mm_value() or 0.0)
                break
            else:
                return False
            self._drag_arrow_axis_origin = origin
        else:
            hit = self._ring_angle(ray_o, ray_d, g)
            if hit is None or abs(hit[1] - g[3]) > 0.2 * g[3]:
                return False
            self._transform_drag = ("ring", hit[0], panel._angle.mm_value() or 0.0)
            self._drag_arrow_axis_origin = g[1]
        self._drag_arrow_active = True
        self._drag_arrow_op = 'transform'
        return True

    def _update_transform_drag(self, ray_o, ray_d):
        drag = getattr(self, '_transform_drag', None)
        panel = getattr(self, '_transform_panel', None)
        if drag is None or panel is None:
            return
        if drag[0] == "ring":
            g = self._transform_gizmo()
            hit = self._ring_angle(ray_o, ray_d, g) if g and g[0] == "rotate" else None
            if hit is None:
                return
            delta = (hit[0] - drag[1] + 180.0) % 360.0 - 180.0
            # Unwrap: keep accumulating past ±180 as the drag goes round.
            base = drag[2] + delta
            prev = panel._angle.mm_value() or 0.0
            base += 360.0 * round((prev - base) / 360.0)
            panel._angle.set_mm(round(base, 1))
            panel._changed()
            self.update()
            return
        k, origin, s0, v0 = drag
        s = self._closest_point_on_axis(ray_o, ray_d, origin, np.array(_AXES[k]))
        if s is not None:
            (panel._dx, panel._dy, panel._dz)[k].set_mm(round(v0 + s - s0, 3))
            panel._changed()
            self.update()

    # ------------------------------------------------------------------
    # OK
    # ------------------------------------------------------------------

    def _on_transform_ok(self, fields: dict):
        from gui.expr_spinbox import exprs_extra
        panel = self._transform_panel
        op = self._transform_op(fields)
        extra = exprs_extra(**panel.spinboxes()) or {}
        editing_idx = getattr(self, '_editing_transform_idx', None)
        self._editing_transform_idx = None     # so close doesn't restore
        self._close_transform_panel()
        if editing_idx is None:
            op.commit_async(self, extra or None)
            return
        preserved = self._tear_down_loft_entry_group(editing_idx)
        if preserved and op.copy:
            extra["_preserved_body_ids"] = preserved
        op.commit(self, extra or None)
        ok, err, _ = self.history.replay_from(self.history.cursor + 1)
        if not ok:
            print(f"[Edit] Downstream replay failed: {err}")
        self.history.seek(len(self.history.entries) - 1)
        self._rebuild_all_meshes()
        self.history_changed.emit()
