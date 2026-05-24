"""
viewer/vp_fillet3d.py

Fillet3DMixin — 3D edge fillet panel, preview, and commit.

Expects self to have:
  history, workspace, _meshes, selection,
  _rebuild_body_mesh(), _post_push_cascade(),
  history_changed signal.
"""

from __future__ import annotations
from PyQt6.QtCore import pyqtSlot, QMetaObject, Qt, Q_ARG


class Fillet3DMixin:

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _try_fillet(self):
        """Open the 3D fillet panel, pre-populated with the current selection."""
        edges = self.selection.edges
        faces = self.selection.faces

        if edges:
            body_id     = edges[0].body_id
            edge_indices = [e.edge_idx for e in edges if e.body_id == body_id]
            face_indices = []
        elif faces:
            body_id = faces[0].body_id
            if any(f.body_id != body_id for f in faces):
                print("[Fillet] All selected faces must be on the same body.")
                return
            face_indices = [f.face_idx for f in faces]
            edge_indices = []
        else:
            body_id      = None
            face_indices = []
            edge_indices = []
        self._show_fillet3d_panel(body_id, face_indices, edge_indices=edge_indices)

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _show_fillet3d_panel(self, body_id: str | None, face_indices: list,
                              editing_entry=None, radius: float = 1.0,
                              edge_indices: list | None = None):
        from gui.fillet3d_panel import Fillet3DPanel

        if getattr(self, '_fillet3d_panel', None) is not None:
            old = self._fillet3d_panel
            old._preview_timer.stop()
            try:
                old.preview_changed.disconnect(self._update_fillet3d_preview)
            except Exception:
                pass
            old.close()
            self._fillet3d_panel = None

        self._fillet3d_body_id        = body_id
        self._fillet3d_face_indices   = list(face_indices)
        self._fillet3d_edge_indices   = list(edge_indices) if edge_indices else []
        self._fillet3d_preview_token  = None
        self._fillet3d_preview_mesh   = None
        self._fillet3d_computing      = False   # True while worker thread is running
        self._fillet3d_pending        = None    # (face_indices, edge_occs, radius) to retry
        self._fillet3d_arrow_base     = None   # fixed face/edge centroid
        self._fillet3d_arrow_origin   = None   # base + dir*radius (moves with drag)
        self._fillet3d_arrow_dir      = None
        self._fillet3d_pick_face      = False
        self._fillet3d_pick_edge      = False
        self._editing_fillet3d_idx    = (editing_entry and
                                          self.history.entries.index(editing_entry)
                                          if editing_entry is not None else None)

        panel = Fillet3DPanel(self.workspace, parent=self)
        panel.set_radius(radius)

        # Pre-populate face and edge lists from initial selection
        if body_id is not None:
            body = self.workspace.bodies.get(body_id)
            name = body.name if body else body_id
            shape = self.workspace.current_shape(body_id)
            all_faces = list(shape.faces()) if shape is not None else []
            for fi in face_indices:
                label = self._fillet3d_face_label(body_id, fi, all_faces)
                panel.add_face_entry(body_id, fi, label)
            for ei in self._fillet3d_edge_indices:
                panel.add_edge_entry(body_id, ei, f"{name}  ·  edge {ei}")

        panel.confirmed.connect(self._on_fillet3d_ok)
        panel.cancelled.connect(self._close_fillet3d_panel)
        panel.preview_changed.connect(self._update_fillet3d_preview)
        panel.face_entry_removed.connect(self._on_fillet3d_face_removed)
        panel.edge_entry_removed.connect(self._on_fillet3d_edge_removed)
        panel.picking_face_changed.connect(self._on_fillet3d_pick_face)
        panel.picking_edge_changed.connect(self._on_fillet3d_pick_edge)

        self._fillet3d_panel = panel
        self._position_fillet3d_panel()
        self._update_fillet3d_arrow()
        panel.show()
        panel.setFocus()
        panel._emit_preview()

    def _fillet3d_face_label(self, body_id: str, face_idx: int,
                              all_faces: list) -> str:
        body = self.workspace.bodies.get(body_id)
        name = body.name if body else body_id
        return f"{name}  ·  face {face_idx}"

    def _position_fillet3d_panel(self):
        p = getattr(self, '_fillet3d_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_fillet3d_panel(self):
        panel = getattr(self, '_fillet3d_panel', None)
        if panel is not None:
            panel._preview_timer.stop()
            panel.end_pick_face()
            panel.end_pick_edge()
            for sig, slot in [
                (panel.preview_changed,     self._update_fillet3d_preview),
                (panel.face_entry_removed,  self._on_fillet3d_face_removed),
                (panel.edge_entry_removed,  self._on_fillet3d_edge_removed),
            ]:
                try:
                    sig.disconnect(slot)
                except Exception:
                    pass
            panel.close()
            self._fillet3d_panel = None

        if getattr(self, '_editing_fillet3d_idx', None) is not None:
            self._cancel_fillet3d_edit()

        self._fillet3d_preview_token = None
        self._fillet3d_preview_mesh  = None
        self._fillet3d_computing     = False
        self._fillet3d_pending       = None
        self._fillet3d_arrow_base    = None
        self._fillet3d_arrow_origin  = None
        self._fillet3d_arrow_dir     = None
        self._fillet3d_pick_face     = False
        self._fillet3d_pick_edge     = False
        self.update()

    # ------------------------------------------------------------------
    # Face picking
    # ------------------------------------------------------------------

    def _on_fillet3d_pick_face(self, active: bool):
        self._fillet3d_pick_face = active

    def _on_fillet3d_pick_edge(self, active: bool):
        self._fillet3d_pick_edge = active

    def _on_fillet3d_face_removed(self, index: int):
        indices = getattr(self, '_fillet3d_face_indices', [])
        if 0 <= index < len(indices):
            indices.pop(index)
        self._fillet3d_face_indices = indices
        if not indices and not getattr(self, '_fillet3d_edge_indices', []):
            self._fillet3d_body_id = None
        self._fillet3d_preview_mesh = None
        panel = getattr(self, '_fillet3d_panel', None)
        if panel is not None:
            panel._emit_preview()
        self.update()

    def _on_fillet3d_edge_removed(self, index: int):
        indices = getattr(self, '_fillet3d_edge_indices', [])
        if 0 <= index < len(indices):
            indices.pop(index)
        self._fillet3d_edge_indices = indices
        if not indices and not getattr(self, '_fillet3d_face_indices', []):
            self._fillet3d_body_id = None
        self._fillet3d_preview_mesh = None
        panel = getattr(self, '_fillet3d_panel', None)
        if panel is not None:
            panel._emit_preview()
        self.update()

    def route_face_pick_for_fillet3d(self, body_id: str, face_idx: int) -> bool:
        panel = getattr(self, '_fillet3d_panel', None)
        if panel is None or not getattr(self, '_fillet3d_pick_face', False):
            return False
        if not self._fillet3d_lock_body(body_id):
            return True

        indices = getattr(self, '_fillet3d_face_indices', [])
        if face_idx in indices:
            panel.remove_face_entry(indices.index(face_idx))
        else:
            indices.append(face_idx)
            self._fillet3d_face_indices = indices
            shape = self.workspace.current_shape(body_id)
            all_faces = list(shape.faces()) if shape is not None else []
            panel.add_face_entry(body_id, face_idx,
                                 self._fillet3d_face_label(body_id, face_idx, all_faces))
            self._update_fillet3d_arrow()
            panel._emit_preview()
        return True

    def route_edge_pick_for_fillet3d(self, edge_idx: int, body_id: str) -> bool:
        panel = getattr(self, '_fillet3d_panel', None)
        if panel is None or not getattr(self, '_fillet3d_pick_edge', False):
            return False
        if not self._fillet3d_lock_body(body_id):
            return True

        indices = getattr(self, '_fillet3d_edge_indices', [])
        if edge_idx in indices:
            panel.remove_edge_entry(indices.index(edge_idx))
        else:
            indices.append(edge_idx)
            self._fillet3d_edge_indices = indices
            body = self.workspace.bodies.get(body_id)
            name = body.name if body else body_id
            panel.add_edge_entry(body_id, edge_idx, f"{name}  ·  edge {edge_idx}")
            self._update_fillet3d_arrow()
            panel._emit_preview()
        return True

    def _fillet3d_lock_body(self, body_id: str) -> bool:
        """Lock to first picked body. Returns False (consumed, rejected) if mismatch."""
        locked = getattr(self, '_fillet3d_body_id', None)
        if locked is None:
            self._fillet3d_body_id = body_id
        elif body_id != locked:
            print("[Fillet] All selections must be on the same body.")
            return False
        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_fillet3d_preview(self, radius: float):
        body_id      = getattr(self, '_fillet3d_body_id', None)
        face_indices = getattr(self, '_fillet3d_face_indices', [])
        edge_indices = getattr(self, '_fillet3d_edge_indices', [])
        if body_id is None or (not face_indices and not edge_indices):
            self._fillet3d_preview_mesh = None
            self.update()
            return

        shape = self.workspace.current_shape(body_id)
        if shape is None or radius <= 0:
            self._fillet3d_preview_mesh = None
            self.update()
            return

        # Resolve TopoDS_Edge objects for direct edge picks (main-thread safe)
        edge_occs = []
        live_mesh = self._meshes.get(body_id)
        if live_mesh is not None:
            for ei in edge_indices:
                if ei < len(live_mesh.topo_edges_occ):
                    edge_occs.append(live_mesh.topo_edges_occ[ei])

        params = (list(face_indices), edge_occs, radius)

        if getattr(self, '_fillet3d_computing', False):
            # A thread is already running — stash latest params, it will re-fire
            self._fillet3d_pending = params
            return

        self._fillet3d_computing = True
        self._fillet3d_pending   = None
        self._launch_fillet3d_thread(shape, params)

    def _launch_fillet3d_thread(self, shape, params):
        import threading
        from cad.operations.fillet import fillet_preview

        face_indices, edge_occs, radius = params
        token = object()
        self._fillet3d_preview_token = token

        def _compute():
            try:
                result = fillet_preview(shape, face_indices, edge_occs, radius)
                from viewer.mesh import Mesh
                preview_mesh = Mesh(result)
            except Exception:
                preview_mesh = None
            QMetaObject.invokeMethod(
                self, "_fillet3d_preview_done",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(object, token),
                Q_ARG(object, preview_mesh),
            )

        threading.Thread(target=_compute, daemon=True).start()

    @pyqtSlot(object, object)
    def _fillet3d_preview_done(self, token, preview_mesh):
        if getattr(self, '_fillet3d_preview_token', None) is not token:
            self._fillet3d_computing = False
            return
        if preview_mesh is not None:
            self.makeCurrent()
            try:
                preview_mesh.upload()
            except Exception:
                preview_mesh = None
        self._fillet3d_preview_mesh = preview_mesh
        self._update_fillet3d_arrow()
        self.update()

        pending = getattr(self, '_fillet3d_pending', None)
        self._fillet3d_pending   = None
        self._fillet3d_computing = False
        if pending is not None:
            shape = self.workspace.current_shape(getattr(self, '_fillet3d_body_id', None))
            if shape is not None:
                self._fillet3d_computing = True
                self._launch_fillet3d_thread(shape, pending)

    def _update_fillet3d_arrow(self):
        """Derive base + direction from selection; then position tip at current radius."""
        import numpy as np
        body_id      = getattr(self, '_fillet3d_body_id', None)
        face_indices = getattr(self, '_fillet3d_face_indices', [])
        edge_indices = getattr(self, '_fillet3d_edge_indices', [])

        def _clear():
            self._fillet3d_arrow_base   = None
            self._fillet3d_arrow_origin = None
            self._fillet3d_arrow_dir    = None

        if body_id is None or (not face_indices and not edge_indices):
            _clear(); return

        shape = self.workspace.current_shape(body_id)
        if shape is None:
            _clear(); return

        try:
            if face_indices:
                all_faces = list(shape.faces())
                fi = face_indices[0]
                if fi >= len(all_faces):
                    raise IndexError
                from build123d import Plane
                pl = Plane(all_faces[fi])
                base   = np.array([pl.origin.X, pl.origin.Y, pl.origin.Z])
                normal = np.array([pl.z_dir.X,  pl.z_dir.Y,  pl.z_dir.Z])
            else:
                mesh = self._meshes.get(body_id)
                if mesh is None or edge_indices[0] >= len(mesh.topo_edges):
                    raise IndexError
                pts  = mesh.topo_edges[edge_indices[0]]
                base = np.array(pts[len(pts) // 2], dtype=float)
                efn  = getattr(mesh, 'topo_edge_face_normals', None)
                if efn and edge_indices[0] < len(efn):
                    adj    = np.array(efn[edge_indices[0]], dtype=float)
                    normal = adj.mean(axis=0) if adj.ndim == 2 else adj.flatten()[:3]
                else:
                    normal = np.array([0.0, 0.0, 1.0])

            n = np.linalg.norm(normal)
            if n < 1e-10:
                raise ValueError
            normal /= n
            self._fillet3d_arrow_base = base
            self._fillet3d_arrow_dir  = normal
        except Exception:
            _clear(); return

        self._reposition_fillet3d_arrow()

    def _reposition_fillet3d_arrow(self):
        """Move the arrow tip to base + dir * radius (called after each drag step)."""
        import numpy as np
        base   = getattr(self, '_fillet3d_arrow_base', None)
        normal = getattr(self, '_fillet3d_arrow_dir',  None)
        if base is None or normal is None:
            self._fillet3d_arrow_origin = None
            return
        panel  = getattr(self, '_fillet3d_panel', None)
        radius = (panel._spinbox.mm_value() or 1.0) if panel else 1.0
        self._fillet3d_arrow_origin = np.asarray(base) + np.asarray(normal) * radius

    def _draw_fillet3d_preview(self):
        origin    = getattr(self, '_fillet3d_arrow_origin', None)
        direction = getattr(self, '_fillet3d_arrow_dir',    None)
        if origin is None or direction is None:
            return
        from viewer.drag_arrow import DragArrow
        panel  = getattr(self, '_fillet3d_panel', None)
        radius = (panel._spinbox.mm_value() or 1.0) if panel else 1.0
        scale  = self._arrow_scale(radius)
        DragArrow().draw(origin, direction, scale, color=(0.40, 0.85, 0.95))

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    @pyqtSlot(float)
    def _on_fillet3d_ok(self, radius: float):
        body_id      = getattr(self, '_fillet3d_body_id', None)
        face_indices = getattr(self, '_fillet3d_face_indices', [])
        edge_indices = getattr(self, '_fillet3d_edge_indices', [])
        editing_idx  = getattr(self, '_editing_fillet3d_idx', None)

        if editing_idx is not None:
            self._editing_fillet3d_idx = None
        self._close_fillet3d_panel()

        if body_id is None or (not face_indices and not edge_indices):
            return

        from cad.op_types import FaceFilletOp
        if editing_idx is not None:
            self._editing_fillet3d_idx = editing_idx
            self._commit_fillet3d_edit(body_id, face_indices, edge_indices, radius)
            return
        FaceFilletOp(source_body_id=body_id,
                     face_indices=face_indices,
                     edge_indices=edge_indices,
                     radius=radius).commit_async(self)

    # ------------------------------------------------------------------
    # Edit / reopen
    # ------------------------------------------------------------------

    def reopen_fillet(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation != "fillet" or entry.op is None:
            return
        op = entry.op
        entry.editing = True
        self._editing_fillet3d_idx = history_idx

        if history_idx > 0:
            self.history.seek(history_idx - 1)
            self._rebuild_body_mesh(op.source_body_id)
        self.history_changed.emit()

        self._show_fillet3d_panel(
            op.source_body_id, list(op.face_indices),
            editing_entry=entry, radius=op.radius)

        # Restore edge picks (panel already open)
        panel = getattr(self, '_fillet3d_panel', None)
        if panel is not None and op.edge_indices:
            self._fillet3d_edge_indices = list(op.edge_indices)
            body = self.workspace.bodies.get(op.source_body_id)
            name = body.name if body else op.source_body_id
            for ei in op.edge_indices:
                panel.add_edge_entry(op.source_body_id, ei,
                                     f"{name}  ·  edge {ei}")

    def _commit_fillet3d_edit(self, body_id: str, face_indices: list,
                               edge_indices: list, radius: float):
        from cad.op_types import FaceFilletOp

        idx = getattr(self, '_editing_fillet3d_idx', None)
        if idx is None:
            return
        self._editing_fillet3d_idx = None

        entries = self.history.entries
        if idx >= len(entries):
            return

        entry = entries[idx]
        entry.editing = False
        entry_id = entry.entry_id

        # Collect group (the fillet entry + any split children)
        group_indices  = [idx]
        group_body_ids = {entry.body_id}
        for j in range(idx + 1, len(entries)):
            if entries[j].params.get("source_entry_id") == entry_id:
                group_indices.append(j)
                group_body_ids.add(entries[j].body_id)

        self.history.seek(max(idx - 1, 0))

        if self.workspace.current_shape(body_id) is None:
            print(f"[Fillet edit] Source body '{body_id}' has no shape at this point.")
            self.history.seek(idx)
            entry.editing = False
            self._rebuild_body_mesh(body_id)
            self.history_changed.emit()
            return

        # Only remove bodies *created* by this op group (split children).
        # The source body and any unrelated bodies must stay.
        group_entry_ids = {entries[j].entry_id for j in group_indices}
        removable = set()
        for bid in group_body_ids:
            body = self.workspace.bodies.get(bid)
            if body is not None and body.created_at_entry_id in group_entry_ids:
                removable.add(bid)
        for j in reversed(group_indices):
            self.history.delete(j)
        for bid in removable:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

        FaceFilletOp(source_body_id=body_id,
                     face_indices=face_indices,
                     edge_indices=edge_indices,
                     radius=radius).commit_async(self)

    def _cancel_fillet3d_edit(self):
        idx = getattr(self, '_editing_fillet3d_idx', None)
        self._editing_fillet3d_idx = None
        if idx is None:
            return
        entries = self.history.entries
        if idx < len(entries):
            entries[idx].editing = False
        self.history.seek(idx)
        body_id = getattr(self, '_fillet3d_body_id', None)
        if body_id is not None:
            self._rebuild_body_mesh(body_id)
        self.history_changed.emit()
