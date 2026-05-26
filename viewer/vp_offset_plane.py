"""
viewer/vp_offset_plane.py

OffsetPlaneMixin — panel lifecycle, face-pick routing, and op dispatch for
the Offset Plane datum operation.
"""

from __future__ import annotations


class OffsetPlaneMixin:

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _show_offset_plane_panel(self):
        from gui.offset_plane_panel import OffsetPlanePanel
        if getattr(self, '_offset_plane_panel', None) is not None:
            return
        panel = OffsetPlanePanel(self.workspace, parent=self)
        panel.confirmed.connect(self._on_offset_plane_confirmed)
        panel.cancelled.connect(self._close_offset_plane_panel)
        panel.picking_face_changed.connect(self._on_offset_plane_pick_face)
        panel.preview_changed.connect(self._on_offset_plane_preview)
        self._offset_plane_panel       = panel
        self._offset_plane_face_active = False
        self._offset_plane_preview     = None  # (origin_np, normal_np, distance, size)
        self._position_offset_plane_panel()
        panel.show()
        panel.setFocus()

    def _position_offset_plane_panel(self):
        p = getattr(self, '_offset_plane_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_offset_plane_panel(self):
        p = getattr(self, '_offset_plane_panel', None)
        if p is not None:
            p.close()
        self._offset_plane_panel       = None
        self._offset_plane_face_active = False
        self._offset_plane_preview     = None
        # Clear edit flag on the entry being edited so the history row stops
        # showing the "editing" highlight.
        idx = getattr(self, '_editing_offset_plane_idx', None)
        if idx is not None:
            self._editing_offset_plane_idx = None
            entries = self.history.entries
            if 0 <= idx < len(entries):
                entries[idx].editing = False
            self.history_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # Reopen / edit
    # ------------------------------------------------------------------

    def reopen_offset_plane(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation != "offset_plane" or entry.op is None:
            return
        entry.editing = True
        self._editing_offset_plane_idx = history_idx

        self._show_offset_plane_panel()
        panel = self._offset_plane_panel
        if panel is None:
            return

        # Restore widgets from the stored op.
        panel.load_from_op(entry.op, parent_label=self._infer_parent_label(entry.op))
        self.history_changed.emit()

    def _infer_parent_label(self, op) -> str | None:
        """Best-effort human label for the parent shown in Face mode after reopen."""
        from cad.plane_ref import OffsetPlaneSource, FacePlaneSource
        ps = op.plane_source
        if not isinstance(ps, OffsetPlaneSource):
            return None
        parent = ps.parent
        if isinstance(parent, FacePlaneSource):
            body = self.workspace.bodies.get(parent.body_id)
            return f"Face on {body.name}" if body else None
        # For nested offset / sketch sources, label is just descriptive.
        return None

    # ------------------------------------------------------------------
    # Pick routing
    # ------------------------------------------------------------------

    def _on_offset_plane_pick_face(self, active: bool):
        self._offset_plane_face_active = active

    def route_face_pick_for_offset_plane(self, body_id: str, face_idx: int) -> bool:
        if not getattr(self, '_offset_plane_face_active', False):
            return False
        panel = getattr(self, '_offset_plane_panel', None)
        if panel is None:
            return False
        body = self.workspace.bodies.get(body_id)
        if body is None:
            return False
        # Verify the face is planar — non-planar faces can't be parents.
        mesh = self._meshes.get(body_id)
        if mesh is None or face_idx >= len(mesh.occt_faces):
            return False
        from build123d import Plane
        try:
            Plane(mesh.occt_faces[face_idx])
        except Exception:
            print(f"[OffsetPlane] Face {face_idx} on {body.name} is not planar.")
            return True
        panel.set_picked_face(body_id, face_idx, body.name)
        return True

    def route_sketch_face_pick_for_offset_plane(self, sketch_idx: int) -> bool:
        """Route a sketch-face pick through to the offset-plane panel."""
        if not getattr(self, '_offset_plane_face_active', False):
            return False
        panel = getattr(self, '_offset_plane_panel', None)
        if panel is None:
            return False
        entries = self.history.entries
        if sketch_idx < 0 or sketch_idx >= len(entries):
            return False
        entry = entries[sketch_idx]
        se = entry.params.get("sketch_entry") if entry.operation == "sketch" else None
        if se is None or se.plane_source is None:
            return False
        panel.set_picked_sketch_plane(se.plane_source, f"Sketch {sketch_idx}")
        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_offset_plane_preview(self):
        """Resolve the panel's pending plane source and stash a preview pose."""
        import numpy as np
        panel = getattr(self, '_offset_plane_panel', None)
        if panel is None:
            self._offset_plane_preview = None
            self.update()
            return
        ps = self._build_plane_source_from_panel(panel)
        if ps is None:
            self._offset_plane_preview = None
            self.update()
            return
        try:
            b3d_plane = ps.resolve(self.history, self.history.cursor + 1)
        except Exception:
            self._offset_plane_preview = None
            self.update()
            return
        origin = np.array([b3d_plane.origin.X, b3d_plane.origin.Y,
                           b3d_plane.origin.Z], dtype=float)
        normal = np.array([b3d_plane.z_dir.X, b3d_plane.z_dir.Y,
                           b3d_plane.z_dir.Z], dtype=float)
        x_dir  = np.array([b3d_plane.x_dir.X, b3d_plane.x_dir.Y,
                           b3d_plane.x_dir.Z], dtype=float)
        size   = self._offset_plane_size(ps)
        self._offset_plane_preview = (origin, normal, x_dir, size)
        self.update()

    def _offset_plane_size(self, plane_source) -> float:
        """Auto-size: face-parent → parent face bbox extent; world → scene-ish."""
        import numpy as np
        from cad.plane_ref import OffsetPlaneSource, FacePlaneSource
        # Walk to the root non-offset parent
        node = plane_source
        while isinstance(node, OffsetPlaneSource):
            node = node.parent
        if isinstance(node, FacePlaneSource):
            shape = self.workspace.current_shape(node.body_id)
            if shape is not None:
                try:
                    idx, face = node.face_ref.find_in(shape)
                    if face is not None:
                        bb = face.bounding_box()
                        # Take the largest extent and pad
                        ext = max(bb.size.X, bb.size.Y, bb.size.Z) * 0.6
                        return max(20.0, ext)
                except Exception:
                    pass
        # Fallback for world-parented (or face lookup failed): scene-ish
        if self._meshes:
            all_mins = np.vstack([m.bbox_min for m in self._meshes.values()])
            all_maxs = np.vstack([m.bbox_max for m in self._meshes.values()])
            ext = float(np.linalg.norm(all_maxs.max(axis=0) - all_mins.min(axis=0))) * 0.4
            return max(40.0, ext)
        return 50.0

    # ------------------------------------------------------------------
    # Build plane source from panel state
    # ------------------------------------------------------------------

    def _build_plane_source_from_panel(self, panel):
        """Resolve panel state to a real SketchPlaneSource (with FaceRef)."""
        from cad.plane_ref import (WorldPlaneSource, FacePlaneSource,
                                    OffsetPlaneSource)
        from cad.face_ref import FaceRef
        ps_dict = panel.build_plane_source_dict()
        if ps_dict is None:
            return None
        dist     = float(ps_dict["distance"])
        parent_d = ps_dict["parent"]
        if parent_d["type"] == "world":
            parent = WorldPlaneSource(parent_d["axis"])
            return OffsetPlaneSource(parent, dist)
        if parent_d["type"] == "sketch_live":
            parent = panel.parent_sketch_source()
            if parent is None:
                return None
            return OffsetPlaneSource(parent, dist)
        if parent_d["type"] == "loaded_source":
            parent = panel.loaded_parent_source()
            if parent is None:
                return None
            return OffsetPlaneSource(parent, dist)
        if parent_d["type"] == "face_pending":
            body_id  = parent_d["body_id"]
            face_idx = parent_d["face_idx"]
            shape = self.workspace.current_shape(body_id)
            if shape is None:
                return None
            faces = list(shape.faces())
            if face_idx >= len(faces):
                return None
            face_ref = FaceRef.from_b3d_face(faces[face_idx])
            if face_ref is None:
                return None
            parent = FacePlaneSource(body_id, face_ref)
            return OffsetPlaneSource(parent, dist)
        return None

    # ------------------------------------------------------------------
    # OK handler
    # ------------------------------------------------------------------

    def _on_offset_plane_confirmed(self, _ps_dict, name: str):
        from cad.op_types import OffsetPlaneOp
        panel = getattr(self, '_offset_plane_panel', None)
        if panel is None:
            return
        plane_source = self._build_plane_source_from_panel(panel)
        if plane_source is None:
            print("[OffsetPlane] Cannot create: incomplete parent face / distance.")
            return

        editing_idx = getattr(self, '_editing_offset_plane_idx', None)
        if editing_idx is not None:
            self._commit_offset_plane_edit(editing_idx, plane_source, name)
            return

        # Fresh create
        if not name:
            count = sum(1 for e in self.history.entries
                        if e.operation == "offset_plane")
            name = f"Plane {count + 1}"
        op = OffsetPlaneOp(plane_source=plane_source, name=name)
        op.commit(self, None)
        self._close_offset_plane_panel()
        self.update()

    def _commit_offset_plane_edit(self, idx: int, plane_source, name: str):
        """Mutate the existing op + params in place so downstream sketches that
        share the plane_source object reference pick up the new distance on
        replay.  Tearing down + re-creating would orphan those references.
        """
        from cad.op_types import OffsetPlaneOp
        entries = self.history.entries
        if idx >= len(entries):
            return
        entry = entries[idx]
        old_op = entry.op
        if not isinstance(old_op, OffsetPlaneOp):
            return

        # Mutate the existing plane_source so any sketch that holds a reference
        # to it sees the new distance.  Only mutate the *distance* field of an
        # OffsetPlaneSource — if the parent type changed (e.g. World → Face)
        # the user picked a fresh parent and we must replace the source object
        # outright, which means downstream sketches drift to the old one.
        from cad.plane_ref import OffsetPlaneSource
        old_ps = old_op.plane_source
        if (isinstance(old_ps, OffsetPlaneSource)
                and isinstance(plane_source, OffsetPlaneSource)
                and type(old_ps.parent) is type(plane_source.parent)):
            old_ps.distance = float(plane_source.distance)
            # Mutate the parent too in case its internal fields changed
            # (e.g. axis change WorldPlaneSource("XY") → WorldPlaneSource("XZ")).
            for attr, val in vars(plane_source.parent).items():
                setattr(old_ps.parent, attr, val)
            # Keep the same op object — just update its name.
            if name:
                old_op.name = name
        else:
            # Parent topology changed; replace the source.  Downstream sketches
            # that referenced the old source will keep using it — that's the
            # accepted limitation.
            old_op.plane_source = plane_source
            if name:
                old_op.name = name

        # Refresh params and label
        entry.params = old_op.to_params()
        from cad.units import format_op_label as _lbl
        entry.label = _lbl("offset_plane", entry.params)
        entry.editing = False
        self._editing_offset_plane_idx = None

        # Replay downstream so sketches/lofts that depend on this plane recompute.
        ok, err, _ = self.history.replay_from(idx)
        if not ok:
            print(f"[OffsetPlane edit] Downstream replay failed: {err}")
        # Advance cursor to tip so bodies created downstream are visible.
        self.history.seek(len(self.history.entries) - 1)
        self._rebuild_all_meshes()
        self._close_offset_plane_panel()
        self.history_changed.emit()
        self.update()
