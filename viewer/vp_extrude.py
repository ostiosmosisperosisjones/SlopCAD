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
from cad.solid_ref import as_compound

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
            p = self._extrude_panel
            self._arm_pick(p._pick_face_btn, p._on_pick_face_toggle)

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
            preview_faces = self._selection_faces(sketch_idx, face_pairs)
            if not preview_faces:
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
            faces = self._selection_faces(sketch_idx, face_pairs)
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

    # ------------------------------------------------------------------
    # Profile selection (shared with revolve): one sketch's loops plus any
    # number of body faces.  _selected_sketch_face holds the sketch's picked
    # loops (None = every loop); face pairs are (body_id, face_idx).
    # ------------------------------------------------------------------

    def _selected_loops(self, sketch_idx) -> list | None:
        """Picked loop indices of the selection's sketch (None = all loops)."""
        sel = self._selected_sketch_face
        if sketch_idx is None or sel is None:
            return None
        n = len(self._sketch_faces.get(sketch_idx, []))
        return [i for i in sel if 0 <= i < n] or None

    def _selection_faces(self, sketch_idx, face_pairs) -> list:
        """Live build123d faces of a profile selection, sketch loops first."""
        faces = []
        if sketch_idx is not None:
            all_sketch = self._sketch_faces.get(sketch_idx, [])
            loops = self._selected_loops(sketch_idx)
            faces += [all_sketch[i][0] for i in
                      (loops if loops is not None else range(len(all_sketch)))]
        for bid, fi in face_pairs or []:
            shape = self.workspace.current_shape(bid)
            if shape is None:
                continue
            all_f = list(shape.faces())
            if 0 <= fi < len(all_f):
                faces.append(all_f[fi])
        return faces

    def _selection_profiles(self, sketch_idx, face_pairs) -> list:
        """ProfileSources for a selection: the sketch (its picked loops
        baked) first, then the body faces in pick order."""
        from cad.profile_source import SketchProfile, BodyFaceProfile
        profs = []
        if sketch_idx is not None:
            sid = self.history.index_to_id(sketch_idx)
            if sid is not None:
                profs.append(SketchProfile(sketch_id=sid,
                                           face_indices=self._selected_loops(sketch_idx)))
        profs += [BodyFaceProfile(body_id=b, face_idx=int(fi))
                  for b, fi in face_pairs or []]
        return profs

    def _live_face_pairs(self, profiles) -> list:
        """(body_id, face_idx) of an op's face profiles on the current shapes
        (re-found by ref; the stored index when the face can't be found)."""
        from cad.profile_source import BodyFaceProfile
        pairs = []
        for p in profiles:
            if not isinstance(p, BodyFaceProfile):
                continue
            fi = p.face_idx
            shape = self.workspace.current_shape(p.body_id)
            if shape is not None and p.face_ref is not None:
                try:
                    found, _f = p.face_ref.find_in(shape)
                    if found is not None:
                        fi = found
                except Exception:
                    pass
            pairs.append((p.body_id, fi))
        return pairs

    def _selection_rows(self, sketch_idx, face_pairs) -> list:
        """Panel rows for a selection: (body_id|None, face_idx|None, label, valid)."""
        rows = []
        if sketch_idx is not None:
            loops = self._selected_loops(sketch_idx)
            if loops:
                rows += [(None, None, f"Sketch {sketch_idx}  ·  face {i}", True)
                         for i in loops]
            else:
                rows.append((None, None, f"Sketch {sketch_idx}", True))
        for bid, fi in face_pairs or []:
            body = self.workspace.bodies.get(bid)
            rows.append((bid, fi, f"{body.name}  ·  face {fi}", True) if body
                        else (None, None, "⚠  face lost — pick again", False))
        return rows

    def _remove_selection_row(self, index, sketch_idx, face_pairs):
        """Drop panel row *index* from a selection → (sketch_idx, face_pairs).
        The sketch's rows come first; removing its last row drops the sketch."""
        n_sketch = 0
        if sketch_idx is not None:
            loops = self._selected_loops(sketch_idx)
            n_sketch = len(loops) if loops else 1
        if index < n_sketch:
            loops = self._selected_loops(sketch_idx)
            if loops and len(loops) > 1:
                loops.pop(index)
                self._selected_sketch_face = loops
            else:
                sketch_idx = None
                self._selected_sketch_face = None
            return sketch_idx, list(face_pairs)
        k = index - n_sketch
        return sketch_idx, [p for i, p in enumerate(face_pairs) if i != k]

    def _refresh_extrude_rows(self):
        panel = getattr(self, '_extrude_panel', None)
        if panel is None:
            return
        panel.clear_face_entries()
        for bid, fi, label, valid in self._selection_rows(
                self._extrude_sketch_idx, self._extrude_face_pairs):
            panel.add_face_entry(bid, fi, label, valid=valid)
        self._update_panel_mode_lock(panel, self._extrude_face_pairs)

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
        self._extrude_sketch_idx, self._extrude_face_pairs = \
            self._remove_selection_row(index, self._extrude_sketch_idx,
                                       getattr(self, '_extrude_face_pairs', []))
        panel = getattr(self, '_extrude_panel', None)
        if panel is not None:
            self._update_panel_mode_lock(panel, self._extrude_face_pairs)
            panel._emit_preview()

    def route_face_pick_for_extrude(self, body_id: str, face_idx: int) -> bool:
        """Toggle a body face in the selection (alongside any sketch loops)."""
        if not getattr(self, '_extrude_face_active', False):
            return False
        panel = getattr(self, '_extrude_panel', None)
        pairs = list(getattr(self, '_extrude_face_pairs', []))
        added = (body_id, face_idx) not in pairs
        pairs = pairs + [(body_id, face_idx)] if added else \
            [p for p in pairs if p != (body_id, face_idx)]
        self._extrude_face_pairs = pairs
        if panel is not None:
            self._refresh_extrude_rows()
            if added:
                # Use the last-picked face for origin/normal hint
                origin, normal = self._face_origin_and_normal(None, body_id, face_idx)
                if origin is not None:
                    panel.set_face_origin(origin)
                if normal is not None:
                    panel.set_face_normal(normal)
            panel._emit_preview()
        return True

    def _update_panel_mode_lock(self, panel, face_pairs):
        """Lock out merge/cut modes when the profiles span several bodies
        (their faces' bodies plus the sketch's parent body)."""
        bodies = {bid for bid, _ in face_pairs}
        sketch_idx = getattr(self, '_extrude_sketch_idx', None)
        if sketch_idx is not None and face_pairs:
            entries = self.history.entries
            se = (entries[sketch_idx].params.get("sketch_entry")
                  if sketch_idx < len(entries) else None)
            if se is not None and se.body_id:
                bodies.add(se.body_id)
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
        """Toggle a sketch loop in the selection (alongside any body faces).
        Loops of a different sketch replace the previous sketch's loops."""
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
        self._selected_sketch_entry = sketch_idx
        self._selected_sketch_face  = current_indices if current_indices else None

        if panel is not None:
            self._refresh_extrude_rows()
            origin, normal = self._face_origin_and_normal(sketch_idx, None, None)
            if origin is not None:
                panel.set_face_origin(origin)
            if normal is not None:
                panel.set_face_normal(normal)
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
            panel.set_vertex_target(pos, _picked_vertex_source(
                self.history, body_id, mesh.shape, pos))
        return True

    def route_edge_pick_for_extrude(self, edge_idx: int, body_id: str) -> bool:
        import numpy as np
        if not getattr(self, '_extrude_pick_active', False):
            return False
        mesh = self._meshes.get(body_id)
        if mesh is None or edge_idx >= len(mesh.topo_edges_occ):
            return False
        from cad.op_extrude import edge_tangent
        occ = mesh.topo_edges_occ[edge_idx]
        try:
            vec = edge_tangent(occ)
        except Exception as ex:
            print(f"[Extrude] Edge tangent failed: {ex}"); return False
        if vec is None:
            return False
        panel = getattr(self, '_extrude_panel', None)
        if panel is not None:
            panel.set_direction(vec, _picked_edge_source(self.history, body_id,
                                                         mesh.shape, occ))
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
        #   - For a no-target cut: its fan-out siblings (one per body cut).
        # Split imports (operation="import") and force_new_body children
        # (operation="extrude"/"cut") share source_entry_id == this entry_id.
        group_indices  = self.history.op_group(idx)
        group_body_ids = {entries[j].body_id for j in group_indices}

        # Profiles of the edited selection (built before the group is deleted)
        # and the face bodies the pre-commit guard checks.
        profiles   = self._selection_profiles(sketch_idx, face_pairs)
        src_bodies = {bid for bid, _ in face_pairs or []}

        # --- Step 2: seek to idx-1 so commit() sees the pre-op state. --------
        self.history.seek(max(idx - 1, 0))

        # Guard: abort if any source body has no valid shape at idx-1.
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

        keep_id = (self.history.entries[min(group_indices)].entry_id, min(group_indices))
        for j in reversed(group_indices):
            self.history.delete(j)
        # The re-commit keeps the entry id: downstream refs name it (revolve /
        # extrude cap and chamfer face origins of planes and edge picks).
        self.history.reuse_entry_id(*keep_id)
        for bid in removable_bodies:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

        # --- Step 4: build the new op and commit it fresh. -------------------
        if dist < 0 and merge_body_id is None:
            # No-target cut: fan out one CrossBodyCutOp per intersecting body.
            # Mirrors the fresh-commit path in _on_extrude_panel_ok so an edit
            # that lands on "self-cut" mode behaves like the original tool.
            self._do_cut_all_intersecting(dist, direction, profiles, extra)
            new_idx = self.history.cursor
            ok, err, _ = self.history.replay_all_from(new_idx + 1)
            if not ok:
                print(f"[Edit] Downstream replay failed: {err}")
            # Advance cursor to tip so bodies created downstream are visible.
            self.history.seek(len(self.history.entries) - 1)
            self._rebuild_all_meshes()
            self.history_changed.emit()
            return

        new_op = self._build_extrude_op(profiles, dist, direction, merge_body_id, extra)
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

    def _build_extrude_op(self, profiles, dist, direction, merge_body_id, extra):
        """The op for a panel commit (cad.extrude_commit.extrude_op, shared
        with the scripting API)."""
        from cad.extrude_commit import extrude_op
        return extrude_op(profiles, dist, direction, merge_body_id, extra)

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
        by_vertex     = panel is not None and panel._target_group.checkedId() == 1
        target_vertex = panel._target_vertex if by_vertex else None
        target_source = panel._target_source if by_vertex else None
        vertex_side   = -1 if by_vertex and panel._flip else 1
        dir_source    = (panel._direction_source
                         if panel is not None and panel._direction is not None else None)
        start_off     = (panel._start_offset.mm_value() or 0.0) if panel else 0.0
        end_off       = (panel._end_offset.mm_value()   or 0.0) if panel else 0.0
        draft_angle   = panel.draft_angle_deg() if panel else 0.0
        exprs         = _panel_bindings(panel, target_vertex is None)

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
            extra['vertex_side'] = vertex_side
            if target_source is not None:
                extra['vertex_source'] = target_source
        if dir_source is not None:
            extra['direction_source'] = dir_source
        if start_off != 0.0:
            extra['start_offset'] = start_off
        if end_off != 0.0:
            extra['end_offset'] = end_off
        if draft_angle != 0.0:
            extra['draft_angle'] = draft_angle
        if exprs:
            extra['exprs'] = exprs

        if editing_idx is not None:
            self._editing_history_idx = editing_idx  # restore for _commit
            self._commit_extrude_edit(dist, direction, merge_body_id,
                                      sketch_idx, face_pairs, extra,
                                      target_vertex, start_off, end_off)
            return

        profiles = self._selection_profiles(sketch_idx, face_pairs)
        if not profiles:
            print("[Extrude] Nothing selected.")
            return

        # No-target cut: cut through every body the tool intersects, one
        # CrossBodyCutOp entry per affected body.
        if dist < 0 and merge_body_id is None:
            self._do_cut_all_intersecting(dist, direction, profiles, extra)
            return

        self._build_extrude_op(profiles, dist, direction, merge_body_id,
                               extra).commit_async(self, extra)

    # ------------------------------------------------------------------
    # Cross-body cut dispatch
    # ------------------------------------------------------------------

    def _do_cut_all_intersecting(self, dist, direction, profiles, extra):
        """No-target cut: one CrossBodyCutOp per body the tool intersects
        (cad.extrude_commit.cut_all_intersecting, shared with the API)."""
        from cad.extrude_commit import cut_all_intersecting
        cut_all_intersecting(self, dist, direction, profiles, extra)

    def do_extrude(self, face_pairs, distance: float,
                   direction=None, merge_body_id: str | None = None,
                   force_new_body: bool = False,
                   extra_params: dict | None = None):
        """Extrude body faces: face_pairs is a list of (body_id, face_idx).
        Faces on several bodies always make new bodies."""
        pairs = list(face_pairs)
        if not pairs:
            return
        if len({bid for bid, _ in pairs}) > 1:
            force_new_body, merge_body_id = True, None
        self.do_extrude_profiles(self._selection_profiles(None, pairs), distance,
                                 direction, merge_body_id, force_new_body,
                                 extra_params)

    def do_extrude_sketch(self, history_idx: int, distance: float,
                          direction=None, merge_body_id: str | None = None,
                          force_new_body: bool = False,
                          extra_params: dict | None = None):
        """Extrude a sketch's picked loops (all loops when none are picked)."""
        if self.history.index_to_id(history_idx) is None:
            print(f"[Extrude] Sketch at index {history_idx} not found."); return
        self.do_extrude_profiles(self._selection_profiles(history_idx, []), distance,
                                 direction, merge_body_id, force_new_body,
                                 extra_params)

    def do_extrude_profiles(self, profiles, distance: float, direction=None,
                            merge_body_id: str | None = None,
                            force_new_body: bool = False,
                            extra_params: dict | None = None):
        """Build an ExtrudeOp over *profiles* and commit it."""
        merge = "__new_body__" if force_new_body else merge_body_id
        self._build_extrude_op(list(profiles), distance, direction, merge,
                               extra_params).commit_async(self, extra_params)


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


def _picked_edge_source(history, body_id, shape, occ_edge) -> dict | None:
    """The picked direction edge as a serialized BodyEdgeSource, or None."""
    from cad.edge_ref import EdgeRef, BodyEdgeSource
    from cad.serializer import _edge_source_to_dict
    try:
        occ = getattr(shape, "wrapped", shape)
        ref = EdgeRef.for_edge(history, body_id, occ_edge, occ, history.cursor + 1)
        return None if ref is None else _edge_source_to_dict(
            BodyEdgeSource(body_id, ref, ref.origin))
    except Exception as ex:
        print(f"[Extrude] picked direction won't follow edits: {ex}")
        return None


def _picked_vertex_source(history, body_id, shape, pos) -> dict | None:
    """The picked vertex as a serialized source, or None if it has no edges."""
    from cad.edge_ref import vertex_source_at
    from cad.serializer import _edge_source_to_dict
    try:
        return _edge_source_to_dict(vertex_source_at(
            history, body_id, shape, pos, history.cursor + 1))
    except Exception as ex:
        print(f"[Extrude] picked vertex won't follow edits: {ex}")
        return None


def _panel_bindings(panel, distance_from_field: bool) -> dict:
    """Parameter bindings from the extrude panel's fields → params["exprs"].
    Distance is only bound when it came from the field (not a picked vertex)."""
    if panel is None:
        return {}
    fields = {"start_offset": panel._start_offset,
              "end_offset":   panel._end_offset}
    if distance_from_field:
        fields["distance"] = panel._spinbox
    return {k: b for k, w in fields.items() if (b := w.binding())}
