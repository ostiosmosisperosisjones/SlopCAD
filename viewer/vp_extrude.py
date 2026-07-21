"""
viewer/vp_extrude.py

ExtrudeMixin — panel management, pick routing, edit flow, and geometry
dispatch for extrude / cut / merge operations.

Expects self to have:
  history, workspace, _meshes, _sketch_faces, _selected_sketch_entry,
  _selected_sketch_face, _body_visible, selection,
  _rebuild_body_mesh(), _rebuild_bodies(), _post_push_cascade(),
  history_changed signal.
"""

from __future__ import annotations

from cad.units import format_op_label as _make_label


# ---------------------------------------------------------------------------
# ExtrudeMixin
# ---------------------------------------------------------------------------

class ExtrudeMixin:

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _try_extrude(self):
        if self._selected_sketch_entry is not None:
            self._show_extrude_panel(sketch_idx=self._selected_sketch_entry)
            return
        if self.selection.face_count > 0:
            sf = self.selection.single_face or self.selection.faces[0]
            cb = self.request_extrude_distance
            if cb:
                cb(sf.body_id, sf.face_idx)
                return
            self._show_extrude_panel(body_id=sf.body_id, face_idx=sf.face_idx)
        else:
            self._show_extrude_panel()

    def _show_extrude_panel(self, sketch_idx: int | None = None,
                             body_id: str | None = None,
                             face_idx: int | None = None,
                             editing_entry=None):
        """Show the floating ExtrudePanel over the viewport."""
        import numpy as np
        from gui.extrude_panel import ExtrudePanel
        if hasattr(self, '_extrude_panel') and self._extrude_panel is not None:
            self._extrude_panel.close()

        # Store face target as mutable viewport state so face-pick routing
        # can update it live while the panel is open.
        # _extrude_face_pairs: list of (body_id, face_idx) — supports multi-body.
        self._extrude_sketch_idx  = sketch_idx
        self._extrude_face_pairs  = ([(body_id, face_idx)]
                                     if body_id is not None and face_idx is not None
                                     else [])

        panel = ExtrudePanel(self.workspace, parent=self)

        origin, normal = self._face_origin_and_normal(
            sketch_idx, body_id, face_idx, editing_entry=editing_entry)
        if origin is not None:
            panel.set_face_origin(origin)
        if normal is not None:
            panel.set_face_normal(normal)

        # Populate initial face entries if we already have a face.
        if body_id is not None and face_idx is not None:
            body = self.workspace.bodies.get(body_id)
            if body is not None:
                panel.add_face_entry(body_id, face_idx, f"{body.name}  ·  face {face_idx}")
            else:
                panel.add_face_entry(None, None, "⚠  face lost — pick again", valid=False)
        elif sketch_idx is not None:
            panel.add_face_entry(None, None, f"Sketch {sketch_idx}")

        panel.extrude_requested.connect(self._on_extrude_panel_ok_live)
        panel.cancelled.connect(self._close_extrude_panel)
        panel.picking_edge_changed.connect(self._on_extrude_pick_edge)
        panel.picking_vertex_changed.connect(self._on_extrude_pick_vertex)
        panel.picking_body_changed.connect(self._on_extrude_pick_body)
        panel.picking_face_changed.connect(self._on_extrude_pick_face)
        panel.preview_changed.connect(self._on_extrude_preview_live)
        panel.face_entry_removed.connect(self._on_extrude_face_removed)

        self._extrude_panel        = panel
        self._extrude_pick_active  = False
        self._extrude_vtx_active   = False
        self._extrude_body_active  = False
        self._extrude_face_active  = False
        self._extrude_preview_mesh = None
        self._extrude_arrow_origin = None   # np.ndarray | None
        self._extrude_arrow_dir    = None   # np.ndarray | None  (signed, cut=negative)
        self._position_extrude_panel()
        panel.show()
        panel.setFocus()
        panel._emit_preview()

    def _face_origin_and_normal(self, sketch_idx, body_id, face_idx,
                                 editing_entry=None):
        """
        Return (origin_np, normal_np) for the face being extruded, or (None, None).

        When re-editing an entry, current_shape() may return None for bodies
        that didn't exist before the entry (new-body extrudes).  In that case
        fall back to shape_before stored on the entry itself.
        """
        import numpy as np
        try:
            if sketch_idx is not None:
                entries = self.history.entries
                se = entries[sketch_idx].params.get("sketch_entry") if sketch_idx < len(entries) else None
                if se is not None:
                    origin = np.array(se.plane_origin, dtype=float)
                    x = np.array(se.plane_x_axis, dtype=float)
                    y = np.array(se.plane_y_axis, dtype=float)
                    normal = np.cross(x, y)
                    n = np.linalg.norm(normal)
                    if n > 1e-10:
                        return origin, normal / n
            elif body_id is not None and face_idx is not None:
                shape = self.workspace.current_shape(body_id)
                # Fallback for new-body extrudes: body didn't exist before this
                # entry so current_shape returns None; use the stored shape_before.
                if shape is None and editing_entry is not None:
                    # New-body extrude: shape_before is None on the entry itself.
                    # Find the most recent shape for body_id in history.
                    for e in reversed(self.history.entries):
                        if e.body_id == body_id and e.shape_after is not None:
                            shape = e.shape_after
                            break
                if shape is not None:
                    faces = list(shape.faces())
                    if face_idx < len(faces):
                        from build123d import Plane
                        pl = Plane(faces[face_idx])
                        o  = pl.origin
                        z  = pl.z_dir
                        return (np.array([o.X, o.Y, o.Z], dtype=float),
                                np.array([z.X, z.Y, z.Z], dtype=float))
        except Exception:
            pass
        return None, None

    def _position_extrude_panel(self):
        p = getattr(self, '_extrude_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_extrude_panel(self):
        if hasattr(self, '_extrude_panel') and self._extrude_panel is not None:
            self._extrude_panel.close()
            self._extrude_panel = None
        self._extrude_pick_active  = False
        self._extrude_vtx_active   = False
        self._extrude_body_active  = False
        self._extrude_face_active  = False
        self._extrude_preview_mesh = None
        self._extrude_arrow_origin = None
        self._extrude_arrow_dir    = None
        if getattr(self, '_editing_history_idx', None) is not None:
            self._cancel_extrude_edit()
        self.update()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_extrude_preview(self, dist: float, direction,
                                 sketch_idx, face_pairs):
        """Compute the preview solid and trigger a repaint."""
        from cad.operations.extrude import _do_extrude_solid
        panel        = getattr(self, '_extrude_panel', None)
        start_offset = (panel._start_offset.mm_value() or 0.0) if panel else 0.0
        end_offset   = (panel._end_offset.mm_value()   or 0.0) if panel else 0.0
        draft_angle  = panel.draft_angle_deg() if panel else 0.0
        # Nothing to preview yet (empty spinbox during panel init, or the end
        # offset swallows the whole distance) — bail silently instead of
        # letting the zero-length extrude throw.
        if abs(dist) - end_offset <= 1e-9:
            self._extrude_preview_mesh = None
            self.update()
            return
        try:
            if sketch_idx is not None:
                all_sketch = self._sketch_faces.get(sketch_idx, [])
                if not all_sketch:
                    self._extrude_preview_mesh = None
                    self.update(); return
                fidx_sel = self._selected_sketch_face
                if fidx_sel is not None:
                    preview_faces = [all_sketch[i][0] for i in fidx_sel
                                     if 0 <= i < len(all_sketch)]
                else:
                    preview_faces = [f[0] for f in all_sketch]
            elif face_pairs:
                preview_faces = []
                for bid, fi in face_pairs:
                    shape = self.workspace.current_shape(bid)
                    if shape is None:
                        continue
                    all_f = list(shape.faces())
                    if fi < len(all_f):
                        preview_faces.append(all_f[fi])
                if not preview_faces:
                    self._extrude_preview_mesh = None
                    self.update(); return
            else:
                self._extrude_preview_mesh = None
                self.update(); return

            self._extrude_preview_mesh = [
                _do_extrude_solid(f, dist, direction,
                                  start_offset=start_offset, end_offset=end_offset,
                                  draft_angle=draft_angle)
                for f in preview_faces
            ]
            self._extrude_preview_dist = dist
            self._update_extrude_arrow(dist, direction, sketch_idx, face_pairs,
                                       start_offset)
        except Exception as ex:
            print(f"[Preview] {ex}")
            self._extrude_preview_mesh = None
        self.update()

    def _extrude_face_centroid(self, sketch_idx, face_pairs):
        """Best-effort centroid of the face(s) being extruded, in world space.
        Returns None if nothing usable is available."""
        import numpy as np
        try:
            from build123d import Plane
            faces = []
            if sketch_idx is not None:
                all_sketch = self._sketch_faces.get(sketch_idx, [])
                fidx_sel   = self._selected_sketch_face
                if fidx_sel is not None:
                    faces = [all_sketch[i][0] for i in fidx_sel
                             if 0 <= i < len(all_sketch)]
                else:
                    faces = [f[0] for f in all_sketch]
            elif face_pairs:
                for bid, fi in face_pairs:
                    shape = self.workspace.current_shape(bid)
                    if shape is None:
                        continue
                    all_f = list(shape.faces())
                    if fi < len(all_f):
                        faces.append(all_f[fi])
            if not faces:
                return None
            pts = []
            for f in faces:
                try:
                    o = Plane(f).origin
                    pts.append([o.X, o.Y, o.Z])
                except Exception:
                    pass
            if not pts:
                return None
            return np.mean(np.asarray(pts, dtype=float), axis=0)
        except Exception:
            return None

    def _update_extrude_arrow(self, dist: float, direction,
                               sketch_idx, face_pairs, start_offset: float):
        """Store the world-space arrow base and direction for rendering."""
        import numpy as np
        panel = getattr(self, '_extrude_panel', None)
        if panel is None:
            self._extrude_arrow_origin = None
            self._extrude_arrow_dir    = None
            return

        # Prefer the actual sketch/body face centroid over panel._face_origin.
        # For sketches on a default plane, plane_origin is the world origin —
        # parking the arrow there even when the sketch is drawn far off-axis.
        # Use Plane(face).origin (build123d puts it near the face center) so
        # the arrow stays attached to the part, matching the other ops.
        origin = self._extrude_face_centroid(sketch_idx, face_pairs)
        normal = panel._face_normal
        if origin is None:
            origin = panel._face_origin
        if origin is None or normal is None:
            self._extrude_arrow_origin = None
            self._extrude_arrow_dir    = None
            return

        # direction from the panel is already the signed extrude vector
        # (for a cut it points into the material with dist < 0, giving a net
        # displacement of dist*direction which lands at the cut face).
        if direction is not None:
            arrow_dir = np.asarray(direction, dtype=float)
        else:
            arrow_dir = np.asarray(normal, dtype=float)

        n = np.linalg.norm(arrow_dir)
        if n < 1e-10:
            self._extrude_arrow_origin = None
            self._extrude_arrow_dir    = None
            return
        arrow_dir = arrow_dir / n

        # Place the arrow at the tip of the preview so it always moves with it.
        # direction already encodes the sign (cut → flipped normal), and
        # _do_extrude_solid uses abs(dist) with that direction, so the solid
        # tip is at: face_origin + direction * (start_offset + abs(dist)).
        tip = (np.asarray(origin, dtype=float)
               + arrow_dir * start_offset
               + arrow_dir * abs(dist))
        self._extrude_arrow_origin = tip
        self._extrude_arrow_dir = arrow_dir

    def _draw_extrude_arrow(self):
        """Render the drag arrow on top of the extrude preview."""
        import numpy as np
        from viewer.drag_arrow import DragArrow

        origin = getattr(self, '_extrude_arrow_origin', None)
        direction = getattr(self, '_extrude_arrow_dir', None)
        if origin is None or direction is None:
            return

        dist = getattr(self, '_extrude_preview_dist', 0.0)
        is_cut = dist < 0
        color = (0.95, 0.25, 0.25) if is_cut else (0.95, 0.85, 0.15)

        # Scale arrow relative to camera distance for consistent screen size
        scale = self.camera.distance * 0.10
        scale = max(scale, abs(dist) * 0.18) if dist != 0.0 else scale

        arrow = DragArrow()
        arrow.draw(origin, direction, scale, color=color)

    # ------------------------------------------------------------------
    # Pick routing (called by viewport mouse handler)
    # ------------------------------------------------------------------

    def _on_extrude_preview_live(self, dist: float, direction):
        self._update_extrude_preview(
            dist, direction,
            getattr(self, '_extrude_sketch_idx', None),
            getattr(self, '_extrude_face_pairs', []),
        )

    def _on_extrude_panel_ok_live(self, dist: float, direction, merge_body_id):
        self._on_extrude_panel_ok(
            dist, direction, merge_body_id,
            getattr(self, '_extrude_sketch_idx', None),
            getattr(self, '_extrude_face_pairs', []),
        )

    def _on_extrude_pick_edge(self, active: bool):
        self._extrude_pick_active = active

    def _on_extrude_pick_vertex(self, active: bool):
        self._extrude_vtx_active = active

    def _on_extrude_pick_body(self, active: bool):
        self._extrude_body_active = active

    def _on_extrude_pick_face(self, active: bool):
        self._extrude_face_active = active

    def _on_extrude_face_removed(self, index: int):
        sketch_idx = getattr(self, '_extrude_sketch_idx', None)
        if sketch_idx is not None:
            # Sketch-face mode: remove from _selected_sketch_face list.
            faces = list(self._selected_sketch_face or [])
            if 0 <= index < len(faces):
                faces.pop(index)
            self._selected_sketch_face = faces if faces else None
            panel = getattr(self, '_extrude_panel', None)
            if panel is not None:
                panel._emit_preview()
            return
        pairs = getattr(self, '_extrude_face_pairs', [])
        if 0 <= index < len(pairs):
            pairs = [p for i, p in enumerate(pairs) if i != index]
            self._extrude_face_pairs = pairs
            panel = getattr(self, '_extrude_panel', None)
            if panel is not None:
                self._update_panel_mode_lock(panel, pairs)
                panel._emit_preview()

    def route_face_pick_for_extrude(self, body_id: str, face_idx: int) -> bool:
        if not getattr(self, '_extrude_face_active', False):
            return False
        panel = getattr(self, '_extrude_panel', None)
        pairs = getattr(self, '_extrude_face_pairs', [])

        # Toggle: clicking an already-picked face removes it.
        if (body_id, face_idx) in pairs:
            pairs = [p for p in pairs if p != (body_id, face_idx)]
            self._extrude_face_pairs = pairs
            if panel is not None:
                panel.clear_face_entries()
                for bid, fi in pairs:
                    b = self.workspace.bodies.get(bid)
                    name = b.name if b else bid
                    panel.add_face_entry(bid, fi, f"{name}  ·  face {fi}")
                self._update_panel_mode_lock(panel, pairs)
                panel._emit_preview()
            return True

        self._extrude_sketch_idx = None
        pairs = pairs + [(body_id, face_idx)]
        self._extrude_face_pairs = pairs
        if panel is not None:
            body = self.workspace.bodies.get(body_id)
            name = body.name if body else body_id
            panel.add_face_entry(body_id, face_idx, f"{name}  ·  face {face_idx}")
            # Use the last-picked face for origin/normal hint
            origin, normal = self._face_origin_and_normal(None, body_id, face_idx)
            if origin is not None:
                panel.set_face_origin(origin)
            if normal is not None:
                panel.set_face_normal(normal)
            self._update_panel_mode_lock(panel, pairs)
            panel._emit_preview()
        return True

    def _update_panel_mode_lock(self, panel, face_pairs):
        """Lock out merge/cut modes when faces span multiple bodies."""
        bodies = {bid for bid, _ in face_pairs}
        multi_body = len(bodies) > 1
        if multi_body:
            # Force new-body mode, disable merge and cut options
            panel._radio_extrude.setChecked(True)
            panel._on_mode_changed(0)
            panel._radio_new.setChecked(True)
            panel._op_group.button(0).setChecked(True)
            panel._on_op_changed(0)
            panel._radio_cut.setEnabled(False)
            panel._radio_merge.setEnabled(False)
            panel._radio_extrude.setEnabled(True)
        else:
            panel._radio_cut.setEnabled(True)
            panel._radio_merge.setEnabled(True)

    def route_sketch_face_pick_for_extrude(self, sketch_idx: int, face_idx: int) -> bool:
        """Route a sketch face click to the extrude panel when face picking is active."""
        if not getattr(self, '_extrude_face_active', False):
            return False
        panel = getattr(self, '_extrude_panel', None)

        current_indices = list(self._selected_sketch_face or []) \
            if self._extrude_sketch_idx == sketch_idx else []

        # Toggle: clicking an already-selected face removes it.
        if face_idx in current_indices:
            current_indices.remove(face_idx)
        else:
            current_indices.append(face_idx)

        self._extrude_sketch_idx    = sketch_idx
        self._extrude_face_pairs    = []
        self._selected_sketch_entry = sketch_idx
        self._selected_sketch_face  = current_indices if current_indices else None

        if panel is not None:
            panel.clear_face_entries()
            if current_indices:
                for i in current_indices:
                    panel.add_face_entry(None, None, f"Sketch {sketch_idx}  ·  face {i}")
            else:
                panel.add_face_entry(None, None, f"Sketch {sketch_idx}")
            origin, normal = self._face_origin_and_normal(sketch_idx, None, None)
            if origin is not None:
                panel.set_face_origin(origin)
            if normal is not None:
                panel.set_face_normal(normal)
            self._update_panel_mode_lock(panel, [])
            panel._emit_preview()
        return True

    def route_body_pick_for_extrude(self, body_id: str) -> bool:
        if not getattr(self, '_extrude_body_active', False):
            return False
        body = self.workspace.bodies.get(body_id)
        if body is None:
            return False
        panel = getattr(self, '_extrude_panel', None)
        if panel is not None:
            panel.set_merge_body(body_id, body.name)
        return True

    def route_vertex_pick_for_extrude(self, body_id: str, vertex_idx: int) -> bool:
        import numpy as np
        if not getattr(self, '_extrude_vtx_active', False):
            return False
        mesh = self._meshes.get(body_id)
        if mesh is None or vertex_idx >= len(mesh.topo_verts):
            return False
        pos = np.array(mesh.topo_verts[vertex_idx], dtype=float)
        panel = getattr(self, '_extrude_panel', None)
        if panel is not None:
            panel.set_vertex_target(pos)
        return True

    def route_edge_pick_for_extrude(self, edge_idx: int, body_id: str) -> bool:
        import numpy as np
        if not getattr(self, '_extrude_pick_active', False):
            return False
        mesh = self._meshes.get(body_id)
        if mesh is None or edge_idx >= len(mesh.topo_edges_occ):
            return False
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        try:
            adp = BRepAdaptor_Curve(mesh.topo_edges_occ[edge_idx])
            mid = (adp.FirstParameter() + adp.LastParameter()) * 0.5
            d   = adp.DN(mid, 1)
            vec = np.array([d.X(), d.Y(), d.Z()], dtype=float)
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                return False
            vec /= norm
        except Exception as ex:
            print(f"[Extrude] Edge tangent failed: {ex}"); return False
        panel = getattr(self, '_extrude_panel', None)
        if panel is not None:
            panel.set_direction(vec)
        return True

    # ------------------------------------------------------------------
    # Edit (re-open existing entry)
    # ------------------------------------------------------------------

    def reopen_extrude(self, history_idx: int):
        """
        Double-click an extrude/cut entry to re-edit it.
        Delegates all op-specific panel restoration to entry.op.reopen().
        """
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation not in ("extrude", "cut"):
            return
        if entry.op is None:
            return

        entry.editing = True
        self._editing_history_idx = history_idx
        self._editing_body_id     = entry.body_id

        entry.op.reopen(self, history_idx)

    def _cancel_extrude_edit(self):
        idx = getattr(self, '_editing_history_idx', None)
        self._editing_history_idx = None
        self._cancel_extrude_edit_at(idx)

    def _cancel_extrude_edit_at(self, idx):
        if idx is None:
            return
        entries = self.history.entries
        if idx < len(entries):
            entries[idx].editing = False
        body_id = getattr(self, '_editing_body_id', None)
        self._editing_body_id = None
        if 0 <= idx < len(entries):
            self.history.seek(idx)
            if body_id is not None:
                self._rebuild_body_mesh(body_id)
        self.history_changed.emit()

    def _commit_extrude_edit(self, dist, direction, merge_body_id,
                              sketch_idx, face_pairs, extra,
                              target_vertex, start_off, end_off):
        """
        Commit an edit by:
          1. Seeking to idx-1 (state before the op).
          2. Deleting the original entry group (the main entry + any split-body
             imports or force_new_body entries it produced).
          3. Removing those bodies from the workspace.
          4. Calling new_op.commit() fresh — which handles all push/split/mesh
             logic correctly, just like a first-time commit.
        """
        import numpy as np
        from cad.op_types import FaceExtrudeOp, CrossBodyCutOp, SketchExtrudeOp

        idx = getattr(self, '_editing_history_idx', None)
        if idx is None:
            return
        self._editing_history_idx = None
        self._editing_body_id = None

        entries = self.history.entries
        if idx >= len(entries):
            return

        entry = entries[idx]
        entry.editing = False

        # --- Step 1: collect the group of entries this op produced. ----------
        # The "group" is:
        #   - The main entry at idx.
        #   - Any subsequent import entries with source_entry_id == entry_id
        #     (standard split-body results).
        #   - For force_new_body ops: all subsequent entries that share the
        #     same body_id domain — they have shape_before=None and were
        #     pushed in the same commit call (identified by matching op type
        #     and original op's body neighborhood).
        original_op    = entry.op
        group_indices  = [idx]
        group_body_ids = {entry.body_id}

        entry_id = entry.entry_id
        for j in range(idx + 1, len(entries)):
            e = entries[j]
            # Collect any entry that was produced by this op — both regular
            # split imports (operation="import") and force_new_body children
            # (operation="extrude"/"cut") all share source_entry_id == this entry_id.
            if e.params.get("source_entry_id") == entry_id:
                group_indices.append(j)
                group_body_ids.add(e.body_id)

        # Source bodies — used only by the pre-commit guard below to ensure
        # the op's input still has a valid shape at idx-1.
        body_id  = face_pairs[0][0] if face_pairs else None
        face_idx = face_pairs[0][1] if face_pairs else None
        if sketch_idx is not None:
            se_entry = entries[sketch_idx] if sketch_idx < len(entries) else None
            se = se_entry.params.get("sketch_entry") if se_entry else None
            src_bodies = {se.body_id if se else body_id}
        else:
            src_bodies = {bid for bid, _ in face_pairs} if face_pairs else {body_id}

        # --- Step 2: seek to idx-1 so commit() sees the pre-op state. --------
        self.history.seek(max(idx - 1, 0))

        # Guard: abort if any source body has no valid shape at idx-1.
        if sketch_idx is None and src_bodies:
            for sbid in src_bodies:
                if sbid and self.workspace.current_shape(sbid) is None:
                    print(f"[Edit] Cannot commit: source body '{sbid}' has no "
                          f"valid shape at this point in history.")
                    self.history.seek(idx)
                    entry.editing = False
                    self._rebuild_body_mesh(sbid)
                    self.history_changed.emit()
                    return

        # --- Step 3: delete group entries back-to-front + remove created bodies.
        # A body is removable only if it was *created* by this op group —
        # i.e. its workspace.Body.created_at_entry_id points to one of the
        # group entries. Pre-existing bodies (cut targets, extrude sources,
        # STEP imports) must never be removed, even when they appear in
        # group_body_ids because the op was logged under them.
        child_body_ids = entry.params.get("child_body_ids", [])
        group_entry_ids = {entries[j].entry_id for j in group_indices}
        removable_bodies = set()
        for bid in group_body_ids | set(child_body_ids):
            body = self.workspace.bodies.get(bid)
            if body is not None and body.created_at_entry_id in group_entry_ids:
                removable_bodies.add(bid)

        # Preserve the body uuid(s) so the new op re-creates bodies under the
        # same ids. Without this, downstream ops (fillet/chamfer/thicken/boolean)
        # lose their source_body_id reference on every parametric edit.
        # Also preserve ids for bodies that were consumed by a later op (e.g.
        # boolean union) and are no longer in workspace — they still need the
        # same id so downstream entries in history can find them during replay.
        # Order: main entry's body first, then any split children in entry order.
        preserved_body_ids: list[str] = []
        all_produced = ([entry.body_id] + list(child_body_ids))
        for bid in all_produced:
            if bid and bid not in preserved_body_ids:
                body = self.workspace.bodies.get(bid)
                # Include if: in workspace and created by this op, OR absent
                # from workspace (consumed downstream — still need same id).
                if body is None or body.created_at_entry_id in group_entry_ids:
                    preserved_body_ids.append(bid)

        for j in reversed(group_indices):
            self.history.delete(j)
        for bid in removable_bodies:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

        # --- Step 4: build the new op and commit it fresh. -------------------
        dir_list  = direction.tolist() if direction is not None else None
        is_cut    = dist < 0

        sketch_id = (self.history.index_to_id(sketch_idx)
                     if sketch_idx is not None else None)

        if is_cut and merge_body_id is None:
            # No-target cut: fan out one CrossBodyCutOp per intersecting body.
            # Mirrors the fresh-commit path in _on_extrude_panel_ok so an edit
            # that lands on "self-cut" mode behaves like the original tool.
            self._do_cut_all_intersecting(dist, direction, sketch_idx,
                                          body_id, face_idx, extra)
            new_idx = self.history.cursor
            ok, err, _ = self.history.replay_all_from(new_idx + 1)
            if not ok:
                print(f"[Edit] Downstream replay failed: {err}")
            # Advance cursor to tip so bodies created downstream are visible.
            self.history.seek(len(self.history.entries) - 1)
            self._rebuild_all_meshes()
            self.history_changed.emit()
            return

        if is_cut and merge_body_id not in (None, "__new_body__"):
            src_body_for_cut = (original_op.source_body_id
                                if hasattr(original_op, 'source_body_id') else body_id)
            live_faces = (self._selected_sketch_face
                          if sketch_idx is not None else None)
            new_op = CrossBodyCutOp(
                cut_body_id      = merge_body_id,
                source_body_id   = src_body_for_cut,
                source_face_idx  = face_idx,
                source_sketch_id = sketch_id,
                distance         = abs(dist),
                direction        = dir_list,
                target_vertex    = (extra or {}).get('target_vertex'),
                start_offset     = float((extra or {}).get('start_offset', 0.0)),
                end_offset       = float((extra or {}).get('end_offset', 0.0)),
                draft_angle      = float((extra or {}).get('draft_angle', 0.0)),
                face_indices     = list(live_faces) if live_faces else None,
            )
        elif sketch_idx is not None:
            live_faces = self._selected_sketch_face
            new_op = SketchExtrudeOp(
                from_sketch_id = sketch_id,
                distance       = dist,
                merge_body_id  = (None if merge_body_id in (None, "__new_body__")
                                  else merge_body_id),
                face_indices   = live_faces,
                direction      = dir_list,
                target_vertex  = (extra or {}).get('target_vertex'),
                start_offset   = float((extra or {}).get('start_offset', 0.0)),
                end_offset     = float((extra or {}).get('end_offset', 0.0)),
                draft_angle    = float((extra or {}).get('draft_angle', 0.0)),
                force_new_body = (merge_body_id == "__new_body__"),
            )
        else:
            new_op = FaceExtrudeOp(
                source_body_id = body_id,
                face_idx       = face_idx,
                face_pairs     = list(face_pairs),
                face_indices   = [fi for _, fi in face_pairs],
                distance       = dist,
                direction      = dir_list,
                target_vertex  = (extra or {}).get('target_vertex'),
                start_offset   = float((extra or {}).get('start_offset', 0.0)),
                end_offset     = float((extra or {}).get('end_offset', 0.0)),
                draft_angle    = float((extra or {}).get('draft_angle', 0.0)),
                force_new_body = (merge_body_id == "__new_body__"),
                merge_body_id  = (None if merge_body_id in (None, "__new_body__")
                                  else merge_body_id),
            )

        commit_extra = dict(extra) if extra else {}
        if preserved_body_ids:
            commit_extra["_preserved_body_ids"] = preserved_body_ids
        new_op.commit(self, commit_extra or None)

        # Cascade: replay all body chains after the newly inserted group so
        # downstream ops receive updated shapes (single ordered pass handles
        # cross-body dependencies correctly).
        new_idx = self.history.cursor
        ok, err, _ = self.history.replay_all_from(new_idx + 1)
        if not ok:
            print(f"[Edit] Downstream replay failed: {err}")

        # Advance cursor to tip so bodies created downstream (e.g. a boolean's
        # result body) become visible in the parts panel and viewport.
        self.history.seek(len(self.history.entries) - 1)

        # Full mesh rebuild — cheaper than tracking exactly what changed, and
        # ensures stale GL buffers for removed/replaced bodies are purged.
        self._rebuild_all_meshes()
        self.history_changed.emit()

    # ------------------------------------------------------------------
    # Panel OK handler
    # ------------------------------------------------------------------

    def _on_extrude_panel_ok(self, dist: float, direction, merge_body_id,
                              sketch_idx, face_pairs):
        # Guard against double-fire (e.g. Enter key + default button both firing).
        if not getattr(self, '_extrude_panel', None) and \
                getattr(self, '_editing_history_idx', None) is None:
            return
        panel = self._extrude_panel
        target_vertex = getattr(panel, '_target_vertex', None)
        start_off     = (panel._start_offset.mm_value() or 0.0) if panel else 0.0
        end_off       = (panel._end_offset.mm_value()   or 0.0) if panel else 0.0
        draft_angle   = panel.draft_angle_deg() if panel else 0.0

        editing_idx = getattr(self, '_editing_history_idx', None)
        if editing_idx is not None:
            self._editing_history_idx = None  # prevent cancel on close

        self._close_extrude_panel()
        if dist == 0:
            if editing_idx is not None:
                self._cancel_extrude_edit_at(editing_idx)
            return

        extra = {}
        if target_vertex is not None:
            extra['target_vertex'] = target_vertex.tolist()
        if start_off != 0.0:
            extra['start_offset'] = start_off
        if end_off != 0.0:
            extra['end_offset'] = end_off
        if draft_angle != 0.0:
            extra['draft_angle'] = draft_angle

        if editing_idx is not None:
            self._editing_history_idx = editing_idx  # restore for _commit
            self._commit_extrude_edit(dist, direction, merge_body_id,
                                      sketch_idx, face_pairs, extra,
                                      target_vertex, start_off, end_off)
            return

        is_cut = dist < 0
        # Multi-body always forces new-body — use first pair for cut/sketch compat
        body_id  = face_pairs[0][0] if face_pairs else None
        face_idx = face_pairs[0][1] if face_pairs else None

        if is_cut and merge_body_id not in (None, "__new_body__"):
            self._do_cut_from_body(dist, direction, merge_body_id,
                                   sketch_idx, body_id, face_idx, extra)
            return

        # No-target cut: cut through every body the tool intersects, one
        # CrossBodyCutOp entry per affected body.
        if is_cut and merge_body_id is None:
            self._do_cut_all_intersecting(dist, direction, sketch_idx,
                                          body_id, face_idx, extra)
            return

        force_new_body = (merge_body_id == "__new_body__")
        real_merge_id  = None if force_new_body else merge_body_id

        if sketch_idx is not None:
            self.do_extrude_sketch(sketch_idx, dist,
                                   direction=direction,
                                   merge_body_id=real_merge_id,
                                   force_new_body=force_new_body,
                                   extra_params=extra)
        elif face_pairs:
            self.do_extrude(face_pairs, dist,
                            direction=direction,
                            merge_body_id=real_merge_id,
                            force_new_body=force_new_body,
                            extra_params=extra)

    # ------------------------------------------------------------------
    # Cross-body cut dispatch
    # ------------------------------------------------------------------

    def _do_cut_from_body(self, dist, direction, cut_body_id,
                          sketch_idx, body_id, face_idx, extra):
        """Build CrossBodyCutOp and commit it."""
        import numpy as np
        from cad.op_types import CrossBodyCutOp

        cut_face_indices = None
        if sketch_idx is not None:
            entries = self.history.entries
            se = (entries[sketch_idx].params.get("sketch_entry")
                  if sketch_idx < len(entries) else None)
            src_body_id = se.body_id if se else body_id
            fidx_sel = self._selected_sketch_face
            if fidx_sel is not None:
                all_sketch = self._sketch_faces.get(sketch_idx, [])
                cut_face_indices = [i for i in fidx_sel
                                    if 0 <= i < len(all_sketch)] or None
        else:
            src_body_id = body_id

        sketch_id = (self.history.index_to_id(sketch_idx)
                     if sketch_idx is not None else None)
        dir_list = direction.tolist() if direction is not None else None
        op = CrossBodyCutOp(
            cut_body_id      = cut_body_id,
            source_body_id   = src_body_id,
            source_face_idx  = face_idx,
            source_sketch_id = sketch_id,
            distance         = abs(dist),
            direction        = dir_list,
            target_vertex    = (extra or {}).get('target_vertex'),
            start_offset     = float((extra or {}).get('start_offset', 0.0)),
            end_offset       = float((extra or {}).get('end_offset', 0.0)),
            draft_angle      = float((extra or {}).get('draft_angle', 0.0)),
            face_indices     = cut_face_indices,
        )
        op.commit_async(self, extra)

    def _do_cut_all_intersecting(self, dist, direction, sketch_idx,
                                 body_id, face_idx, extra):
        """
        No-target cut: build the extrude tool solid once, then fan out one
        CrossBodyCutOp per workspace body it intersects (see cad.cut_all).
        """
        import numpy as np
        from cad.op_types import CrossBodyCutOp
        from cad.operations.extrude import _do_extrude_solid
        from cad.cut_all import fan_out_cut
        from build123d import Plane, Compound
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape

        # ---- Resolve tool faces (depends on UI selection) ------------------
        if sketch_idx is not None:
            entries = self.history.entries
            se = (entries[sketch_idx].params.get("sketch_entry")
                  if sketch_idx < len(entries) else None)
            if se is None:
                print("[Cut] No-target cut: sketch entry missing."); return
            src_body_id = se.body_id
            all_sketch = self._sketch_faces.get(sketch_idx, [])
            fidx_sel = self._selected_sketch_face
            if fidx_sel is not None:
                cut_face_indices = [i for i in fidx_sel
                                    if 0 <= i < len(all_sketch)]
            else:
                cut_face_indices = list(range(len(all_sketch)))
            tool_faces = [all_sketch[i][0] for i in cut_face_indices]
        else:
            cut_face_indices = None
            src_body_id = body_id
            src_shape = self.workspace.current_shape(body_id)
            if src_shape is None or face_idx is None:
                print("[Cut] No-target cut: source face missing."); return
            all_faces = list(src_shape.faces())
            if face_idx >= len(all_faces):
                print("[Cut] No-target cut: face_idx out of range."); return
            tool_faces = [all_faces[face_idx]]

        if not tool_faces:
            print("[Cut] No-target cut: no tool faces."); return

        # ---- Build the tool solid once for the bbox prefilter --------------
        tool_dist = abs(dist) + 0.01
        if direction is None:
            z = Plane(tool_faces[0]).z_dir
            tool_dir = -np.array([z.X, z.Y, z.Z], dtype=float)
        else:
            tool_dir = np.asarray(direction, dtype=float)
        start_off    = float((extra or {}).get('start_offset', 0.0))
        end_off      = float((extra or {}).get('end_offset',   0.0))
        draft_angle  = float((extra or {}).get('draft_angle',  0.0))

        try:
            tool_solid = None
            for face in tool_faces:
                s = _do_extrude_solid(face, tool_dist, tool_dir,
                                      start_offset=start_off, end_offset=end_off,
                                      draft_angle=draft_angle)
                if tool_solid is None:
                    tool_solid = s
                else:
                    lst_a = TopTools_ListOfShape(); lst_a.Append(tool_solid.wrapped)
                    lst_b = TopTools_ListOfShape(); lst_b.Append(s.wrapped)
                    fu = BRepAlgoAPI_Fuse()
                    fu.SetArguments(lst_a); fu.SetTools(lst_b)
                    fu.SetRunParallel(True); fu.Build()
                    if fu.IsDone():
                        tool_solid = Compound(fu.Shape())
        except Exception as ex:
            print(f"[Cut] Tool construction failed: {ex}"); return

        sketch_id = (self.history.index_to_id(sketch_idx)
                     if sketch_idx is not None else None)
        dir_list = direction.tolist() if direction is not None else None

        def build_op(bid: str):
            return CrossBodyCutOp(
                cut_body_id      = bid,
                source_body_id   = src_body_id,
                source_face_idx  = face_idx,
                source_sketch_id = sketch_id,
                distance         = abs(dist),
                direction        = dir_list,
                target_vertex    = (extra or {}).get('target_vertex'),
                start_offset     = start_off,
                end_offset       = end_off,
                draft_angle      = draft_angle,
                face_indices     = cut_face_indices,
            )

        fan_out_cut(self, tool_solid, build_op, extra, op_label="Cut")

    def do_extrude(self, face_pairs, distance: float,
                   direction=None, merge_body_id: str | None = None,
                   force_new_body: bool = False,
                   extra_params: dict | None = None):
        """Build FaceExtrudeOp (or delegate to merge path) and commit.

        face_pairs: list of (body_id, face_idx) tuples, or a single body_id str
                    for backward compat (combined with face_idx below).
        """
        from cad.op_types import FaceExtrudeOp

        # Normalise: accept list of (body_id, face_idx) pairs
        if isinstance(face_pairs, str):
            # Legacy: do_extrude(body_id, face_idx, ...) — shouldn't happen anymore
            # but guard just in case
            face_pairs = [(face_pairs, 0)]
        pairs = list(face_pairs)
        if not pairs:
            return
        body_id  = pairs[0][0]
        face_idx = pairs[0][1]

        # Multi-body: always force_new_body, skip merge path
        multi_body = len({bid for bid, _ in pairs}) > 1
        if multi_body:
            force_new_body = True
            merge_body_id  = None

        if merge_body_id is not None and not force_new_body:
            from cad.operations.extrude import _do_extrude_solid
            from build123d import Compound
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
            from OCP.TopTools import TopTools_ListOfShape
            from cad.op_types import _push_result
            shape_before = self.workspace.current_shape(body_id)
            target_shape = self.workspace.current_shape(merge_body_id)
            all_src_faces = list(shape_before.faces())
            face_objs = [all_src_faces[fi] for _, fi in pairs
                         if fi < len(all_src_faces)]
            target_occ   = target_shape.wrapped
            original_solid_count = len(list(target_shape.solids()))
            op_params = {"distance": abs(distance), "merged_from": body_id,
                         "merge_body_id": merge_body_id,
                         "face_pairs": [(bid, fi) for bid, fi in pairs],
                         "source_body_id": body_id}
            if direction is not None:
                op_params["direction"] = direction.tolist()
            if extra_params:
                op_params.update(extra_params)
            _merge_body_id = merge_body_id
            _draft_angle   = float((extra_params or {}).get('draft_angle', 0.0))

            def _compute():
                from build123d import Compound as _C
                from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse as _F
                from OCP.TopTools import TopTools_ListOfShape as _L
                result = target_occ
                for fo in face_objs:
                    tool = _do_extrude_solid(fo, distance, direction,
                                             draft_angle=_draft_angle)
                    lst_a = _L(); lst_a.Append(result)
                    lst_b = _L(); lst_b.Append(tool.wrapped)
                    fuse = _F()
                    fuse.SetArguments(lst_a); fuse.SetTools(lst_b)
                    fuse.SetRunParallel(True); fuse.Build()
                    if not fuse.IsDone():
                        raise RuntimeError("Fuse failed.")
                    result = fuse.Shape()
                return _C(result)

            def _finalize(merged):
                _push_result(self, "extrude", op_params, _merge_body_id,
                             None, target_shape, merged, original_solid_count)

            self.run_op_async("Extrude", _compute, _finalize)
            return

        dir_list = direction.tolist() if direction is not None else None
        op = FaceExtrudeOp(
            source_body_id = body_id,
            face_idx       = face_idx,
            face_pairs     = pairs,
            face_indices   = [fi for _, fi in pairs],
            distance       = distance,
            direction      = dir_list,
            target_vertex  = (extra_params or {}).get('target_vertex'),
            start_offset   = float((extra_params or {}).get('start_offset', 0.0)),
            end_offset     = float((extra_params or {}).get('end_offset', 0.0)),
            draft_angle    = float((extra_params or {}).get('draft_angle', 0.0)),
            force_new_body = force_new_body,
        )
        op.commit_async(self, extra_params)

    def do_extrude_sketch(self, history_idx: int, distance: float,
                          direction=None, merge_body_id: str | None = None,
                          force_new_body: bool = False,
                          extra_params: dict | None = None):
        """Build SketchExtrudeOp and commit it."""
        import numpy as np
        from cad.op_types import SketchExtrudeOp

        sketch_id = self.history.index_to_id(history_idx)
        if sketch_id is None:
            print(f"[Extrude] Sketch at index {history_idx} not found."); return

        dir_list = direction.tolist() if direction is not None else None
        op = SketchExtrudeOp(
            from_sketch_id = sketch_id,
            distance       = distance,
            merge_body_id  = merge_body_id,
            direction      = dir_list,
            target_vertex  = (extra_params or {}).get('target_vertex'),
            start_offset   = float((extra_params or {}).get('start_offset', 0.0)),
            end_offset     = float((extra_params or {}).get('end_offset', 0.0)),
            draft_angle    = float((extra_params or {}).get('draft_angle', 0.0)),
            force_new_body = force_new_body,
        )
        op.commit_async(self, extra_params)


# ---------------------------------------------------------------------------
# Extrude distance dialog (used by request_extrude_distance callbacks)
# ---------------------------------------------------------------------------

def _extrude_distance_dialog(parent) -> float | None:
    """
    Modal dialog with an ExprSpinBox for entering an extrude/cut distance.
    Returns distance in mm (positive = extrude, negative = cut), or None.
    """
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
    from PyQt6.QtCore import Qt
    from gui.expr_spinbox import ExprSpinBox
    from cad.prefs import prefs

    dlg = QDialog(parent)
    dlg.setWindowTitle("Extrude / Cut")
    dlg.setMinimumWidth(280)
    dlg.setStyleSheet("background: #1e1e1e; color: #d4d4d4;")

    layout = QVBoxLayout(dlg)
    layout.setSpacing(10)
    layout.setContentsMargins(16, 14, 16, 12)

    lbl = QLabel("Distance  <span style='color:#666; font-size:11px;'>"
                 "(positive = extrude, negative = cut)</span>")
    lbl.setTextFormat(Qt.TextFormat.RichText)
    layout.addWidget(lbl)

    spinbox = ExprSpinBox(unit=prefs.default_unit)
    layout.addWidget(spinbox)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok |
        QDialogButtonBox.StandardButton.Cancel)
    buttons.setStyleSheet("""
        QPushButton {
            background: #2a2a2a; color: #d4d4d4;
            border: 1px solid #444; border-radius: 3px;
            padding: 4px 14px;
        }
        QPushButton:hover { background: #333; }
        QPushButton:default { border-color: #4a90d9; }
    """)

    def _accept():
        spinbox._on_commit()
        dlg.accept()

    buttons.accepted.connect(_accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)
    spinbox.setFocus()

    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return spinbox.mm_value()
