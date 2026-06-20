"""
viewer/sketch_modal.py

SketchModalMixin — enter, re-enter, exit, and commit sketch mode.

Keeps sketch lifecycle out of viewport.py.
Expects self to have: camera, workspace, history, _meshes, _sketch,
_selected_sketch_entry, _editing_sketch_history_idx, _sketch_faces,
selection, hover, _rebuild_sketch_faces(), _rebuild_all_meshes(),
_post_push_cascade(), history_changed/selection_changed/sketch_mode_changed
signals.
"""

from __future__ import annotations
from build123d import Plane


def _plane_from_sketch_entry(se):
    """Reconstruct a build123d Plane from a SketchEntry's baked plane cache."""
    return Plane(
        origin = tuple(se.plane_origin.tolist()),
        x_dir  = tuple(se.plane_x_axis.tolist()),
        z_dir  = tuple(se.plane_normal.tolist()),
    )


class SketchModalMixin:

    def _enter_sketch(self, body_id: str, face_idx: int):
        from cad.sketch import SketchMode
        from cad.face_ref import FaceRef
        from cad.plane_ref import FacePlaneSource
        mesh = self._meshes.get(body_id)
        if mesh is None:
            return
        try:
            b3d_plane = Plane(mesh.occt_faces[face_idx])
        except Exception:
            print("[Sketch] Face is not planar — cannot enter sketch mode.")
            return
        n, wo = b3d_plane.z_dir, b3d_plane.origin
        xd, yd = b3d_plane.x_dir, b3d_plane.y_dir
        self.camera.snap_to_normal(n.X, n.Y, n.Z, origin=(wo.X, wo.Y, wo.Z),
                                   x_dir=(xd.X, xd.Y, xd.Z),
                                   y_dir=(yd.X, yd.Y, yd.Z))
        face_ref     = FaceRef.from_b3d_face(mesh.occt_faces[face_idx])
        plane_source = FacePlaneSource(body_id, face_ref) if face_ref else None
        self._sketch = SketchMode(b3d_plane, body_id, face_idx,
                                  plane_source=plane_source)
        self._selected_sketch_entry = None
        self._selected_sketch_face  = None
        self.selection.clear()
        self.hover.clear()
        self.selection_changed.emit()
        self.sketch_mode_changed.emit(True)
        self.update()
        print(f"[Sketch] Entered sketch on face {face_idx} of "
              f"{self.workspace.bodies[body_id].name}")

    def _reenter_sketch(self, history_idx: int):
        """Re-open a committed SketchEntry for editing."""
        import copy
        from cad.sketch import SketchMode
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        se = entry.params.get("sketch_entry")
        if se is None:
            return
        b3d_plane = _plane_from_sketch_entry(se)
        n, wo = b3d_plane.z_dir, b3d_plane.origin
        xd, yd = b3d_plane.x_dir, b3d_plane.y_dir
        self.camera.snap_to_normal(n.X, n.Y, n.Z, origin=(wo.X, wo.Y, wo.Z),
                                   x_dir=(xd.X, xd.Y, xd.Z),
                                   y_dir=(yd.X, yd.Y, yd.Z))
        mode = SketchMode(b3d_plane, se.body_id, se.face_idx,
                          plane_source=se.plane_source)
        existing = copy.deepcopy(se.entities)
        mode._entity_snapshots = copy.deepcopy(se.undo_snapshots)
        mode.entities    = existing
        mode.constraints = copy.deepcopy(se.constraints)
        self._sketch = mode
        self._editing_sketch_history_idx = history_idx
        self._selected_sketch_entry = None
        self._selected_sketch_face  = None
        self.selection.clear()
        self.hover.clear()
        self.selection_changed.emit()
        self.sketch_mode_changed.emit(True)
        self.update()
        print(f"[Sketch] Re-entered sketch entry {history_idx} "
              f"({len(mode.entities)} entities)")

    def _exit_sketch(self):
        if self._sketch is None:
            return
        self._sketch = None
        self._editing_sketch_history_idx = None
        self.sketch_mode_changed.emit(False)
        if hasattr(self, '_line_hud'):
            self._line_hud.hide()
        # Clear any pending drag state so stale mouseRelease events don't
        # attempt to call _edit_dimension_label on a None sketch.
        self._dragging_label        = None
        self._drag_start_screen     = None
        self._drag_constraint       = None
        self._drag_perp_world       = None
        self._drag_line_mid         = None
        self._drag_is_diameter      = False
        self._drag_chord_world      = None
        self._drag_arc_center_world = None
        self._drag_arc_radius       = None
        self.update()
        print("[Sketch] Exited sketch mode.")

    def _on_dimension_requested(self, entity_idx: int, current_val: float,
                                entity_type: str = 'line'):
        """Show a spinbox for the user to enter a length or diameter constraint."""
        from PyQt6.QtWidgets import QInputDialog
        from cad.sketch import SketchConstraint
        from cad.prefs import prefs
        from cad.units import from_mm, to_mm
        unit = prefs.default_unit

        is_arc = (entity_type == 'arc')
        label  = f"Diameter ({unit}):" if is_arc else f"Length ({unit}):"
        title  = "Set Diameter" if is_arc else "Set Length"
        con_type = 'diameter' if is_arc else 'distance'

        display_val = from_mm(current_val, unit)
        val_display, ok = QInputDialog.getDouble(
            self, title, label,
            display_val, 0.0, 1_000_000, prefs.display_decimals,
        )
        if not ok:
            return
        val = to_mm(val_display, unit)

        self._sketch.push_undo_snapshot()
        # Replace any existing same-type constraint on this entity.
        self._sketch.constraints = [
            c for c in self._sketch.constraints
            if not (c.type == con_type and c.indices == (entity_idx,))
        ]
        self._sketch.constraints.append(
            SketchConstraint(con_type, (entity_idx,), val)
        )
        from cad.sketch import SketchEntry, LineEntity, ArcEntity
        tmp = SketchEntry.from_sketch_mode(self._sketch)
        if tmp.apply_all_constraints():
            for i, ent in enumerate(tmp.entities):
                if isinstance(ent, LineEntity):
                    self._sketch.entities[i].p0 = ent.p0.copy()
                    self._sketch.entities[i].p1 = ent.p1.copy()
                elif isinstance(ent, ArcEntity):
                    self._sketch.entities[i].center      = ent.center.copy()
                    self._sketch.entities[i].radius      = ent.radius
                    self._sketch.entities[i].start_angle = ent.start_angle
                    self._sketch.entities[i].end_angle   = ent.end_angle
        else:
            # Solver failed — roll back the constraint and snapshot.
            self._sketch.constraints.pop()
            self._sketch._entity_snapshots.pop()
        self._rebuild_sketch_faces()
        self.sketch_mode_changed.emit(True)
        self.update()

    def _complete_sketch_delete(self, editing_idx: int):
        """Delete a sketch entry (and cascade) when re-edited to empty."""
        self._sketch = None
        self._editing_sketch_history_idx = None
        self.sketch_mode_changed.emit(False)
        self.do_delete(editing_idx)
        self._rebuild_sketch_faces()
        self.update()
        print(f"[Sketch] Deleted empty sketch entry {editing_idx}.")

    def _complete_sketch(self):
        if self._sketch is None:
            return
        sketch = self._sketch
        from cad.sketch import LineEntity, ArcEntity, ReferenceEntity, SketchEntry
        lines = [e for e in sketch.entities if isinstance(e, LineEntity)]
        arcs  = [e for e in sketch.entities if isinstance(e, ArcEntity)]
        refs  = [e for e in sketch.entities if isinstance(e, ReferenceEntity)]
        editing_idx = self._editing_sketch_history_idx
        if not lines and not arcs and not refs and editing_idx is not None:
            # Empty sketch on re-edit — delete the entry and cascade
            self._complete_sketch_delete(editing_idx)
            return
        entity_count = len(lines) + len(arcs) + len(refs)
        new_se = SketchEntry.from_sketch_mode(sketch)

        if editing_idx is not None:
            entries = self.history.entries
            if editing_idx < len(entries):
                entries[editing_idx].params["sketch_entry"] = new_se
                self._sketch = None
                self._editing_sketch_history_idx = None
                self.sketch_mode_changed.emit(False)
                self._rebuild_sketch_faces()
                ok, err, _ = self.history.replay_from(editing_idx)
                if not ok:
                    print(f"[Sketch] Replay after re-entry failed: {err}")
                sketch_entry_id = entries[editing_idx].entry_id
                dep_ok, dep_err, _ = self.history.replay_sketch_dependents(sketch_entry_id)
                if not dep_ok:
                    print(f"[Sketch] Dependent replay failed: {dep_err}")
                # Advance cursor to tip so bodies created downstream (e.g. a
                # boolean's result body) are visible after the edit.
                self.history.seek(len(self.history.entries) - 1)
                self._rebuild_all_meshes()
                self.history_changed.emit()
                self.update()
                print(f"[Sketch] Updated sketch entry {editing_idx}, "
                      f"replayed downstream.")
                return

        body_id = sketch.body_id
        if body_id is None:
            body = self.workspace.add_body(
                self.workspace.next_part_name(), None)
            body_id = body.id
            sketch.body_id = body_id
            new_se.body_id  = body_id

        from cad.face_ref import FaceRef
        mesh = self._meshes.get(body_id)
        face_ref = (FaceRef.from_b3d_face(mesh.occt_faces[sketch.face_idx])
                    if mesh and sketch.face_idx >= 0 else None)
        current_shape = self.workspace.current_shape(body_id)
        self.history.push(
            label        = f"Sketch  ({entity_count} entities)",
            operation    = "sketch",
            params       = {"sketch_entry": new_se},
            body_id      = body_id,
            face_ref     = face_ref,
            shape_before = current_shape,
            shape_after  = current_shape,
        )
        self._sketch = None
        self._editing_sketch_history_idx = None
        self.sketch_mode_changed.emit(False)
        self._rebuild_sketch_faces()
        self._post_push_cascade(body_id)
        self.history_changed.emit()
        self.update()
        print(f"[Sketch] Committed {entity_count} entities to history.")

    def _enter_sketch_on_offset_plane(self, entry_idx: int):
        """Enter sketch mode on an OffsetPlaneOp datum entry."""
        from cad.sketch import SketchMode
        if self._sketch is not None:
            return
        entries = self.history.entries
        if entry_idx < 0 or entry_idx >= len(entries):
            return
        entry = entries[entry_idx]
        op    = entry.op
        if op is None or entry.operation != "offset_plane":
            return
        try:
            b3d_plane = op.plane_source.resolve(self.history, entry_idx + 1)
        except Exception as ex:
            print(f"[Sketch] Cannot sketch on '{entry.label}': {ex}")
            return
        n, wo = b3d_plane.z_dir, b3d_plane.origin
        xd, yd = b3d_plane.x_dir, b3d_plane.y_dir
        self.camera.snap_to_normal(n.X, n.Y, n.Z, origin=(wo.X, wo.Y, wo.Z),
                                   x_dir=(xd.X, xd.Y, xd.Z),
                                   y_dir=(yd.X, yd.Y, yd.Z))
        self._sketch = SketchMode(b3d_plane, None, -1,
                                  plane_source=op.plane_source)
        self._selected_sketch_entry = None
        self._selected_sketch_face  = None
        self._selected_plane_idx    = None
        self.selection.clear()
        self.hover.clear()
        self.selection_changed.emit()
        self.sketch_mode_changed.emit(True)
        self.update()
        print(f"[Sketch] Entered sketch on '{entry.label}'")

    def _enter_sketch_on_plane(self, axis: str):
        """Enter sketch mode on one of the three world planes."""
        from cad.sketch import SketchMode
        from cad.plane_ref import WorldPlaneSource
        if self._sketch is not None:
            return
        plane_map = {"XY": Plane.XY, "XZ": Plane.XZ, "YZ": Plane.YZ}
        b3d_plane = plane_map.get(axis)
        if b3d_plane is None:
            return
        n, wo = b3d_plane.z_dir, b3d_plane.origin
        xd, yd = b3d_plane.x_dir, b3d_plane.y_dir
        self.camera.snap_to_normal(n.X, n.Y, n.Z, origin=(wo.X, wo.Y, wo.Z),
                                   x_dir=(xd.X, xd.Y, xd.Z),
                                   y_dir=(yd.X, yd.Y, yd.Z))
        plane_source = WorldPlaneSource(axis)
        self._sketch = SketchMode(b3d_plane, None, -1,
                                  plane_source=plane_source)
        self._selected_sketch_entry = None
        self._selected_sketch_face  = None
        self.selection.clear()
        self.hover.clear()
        self.selection_changed.emit()
        self.sketch_mode_changed.emit(True)
        self.update()
        print(f"[Sketch] Entered sketch on world plane {axis}")
