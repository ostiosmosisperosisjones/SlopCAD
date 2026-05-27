"""
viewer/vp_chamfer.py

ChamferMixin — 3D edge chamfer panel, preview, and commit.

Mirrors the structure of Fillet3DMixin but uses ChamferOp + the chamfer_edges
helper, and carries an extra angle + flip-reference-face parameter.
"""

from __future__ import annotations
from PyQt6.QtCore import pyqtSlot, QMetaObject, Qt, Q_ARG


class ChamferMixin:

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _try_chamfer(self):
        """Open the chamfer panel pre-populated with the current selection."""
        edges = self.selection.edges
        faces = self.selection.faces

        if edges:
            body_id      = edges[0].body_id
            edge_indices = [e.edge_idx for e in edges if e.body_id == body_id]
            face_indices = []
        elif faces:
            body_id = faces[0].body_id
            if any(f.body_id != body_id for f in faces):
                print("[Chamfer] All selected faces must be on the same body.")
                return
            face_indices = [f.face_idx for f in faces]
            edge_indices = []
        else:
            body_id      = None
            face_indices = []
            edge_indices = []
        self._show_chamfer_panel(body_id, face_indices, edge_indices=edge_indices)

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _show_chamfer_panel(self, body_id: str | None, face_indices: list,
                             editing_entry=None, distance: float = 1.0,
                             angle_deg: float = 45.0, flip: bool = False,
                             edge_indices: list | None = None):
        from gui.chamfer_panel import ChamferPanel

        if getattr(self, '_chamfer_panel', None) is not None:
            old = self._chamfer_panel
            old._preview_timer.stop()
            try:
                old.preview_changed.disconnect(self._update_chamfer_preview)
            except Exception:
                pass
            old.close()
            self._chamfer_panel = None

        self._chamfer_body_id        = body_id
        self._chamfer_face_indices   = list(face_indices)
        self._chamfer_edge_indices   = list(edge_indices) if edge_indices else []
        self._chamfer_preview_token  = None
        self._chamfer_preview_mesh   = None
        self._chamfer_computing      = False
        self._chamfer_pending        = None
        self._chamfer_pick_face      = False
        self._chamfer_pick_edge      = False
        self._chamfer_arrow_base     = None
        self._chamfer_arrow_origin   = None
        self._chamfer_arrow_dir      = None
        self._editing_chamfer_idx    = (
            self.history.entries.index(editing_entry)
            if editing_entry is not None else None)

        panel = ChamferPanel(self.workspace, parent=self)
        panel.set_distance(distance)
        panel.set_angle(angle_deg)
        panel.set_flip(flip)

        if body_id is not None:
            body = self.workspace.bodies.get(body_id)
            name = body.name if body else "Body"
            shape = self.workspace.current_shape(body_id)
            all_faces = list(shape.faces()) if shape is not None else []
            for fi in face_indices:
                panel.add_face_entry(body_id, fi,
                                      self._chamfer_face_label(body_id, fi, all_faces))
            for ei in self._chamfer_edge_indices:
                panel.add_edge_entry(body_id, ei, f"{name}  ·  edge {ei}")

        panel.confirmed.connect(self._on_chamfer_ok)
        panel.cancelled.connect(self._close_chamfer_panel)
        panel.preview_changed.connect(self._update_chamfer_preview)
        panel.face_entry_removed.connect(self._on_chamfer_face_removed)
        panel.edge_entry_removed.connect(self._on_chamfer_edge_removed)
        panel.picking_face_changed.connect(self._on_chamfer_pick_face)
        panel.picking_edge_changed.connect(self._on_chamfer_pick_edge)

        self._chamfer_panel = panel
        self._position_chamfer_panel()
        self._update_chamfer_arrow()
        panel.show()
        panel.setFocus()
        panel._emit_preview()

    def _chamfer_face_label(self, body_id: str, face_idx: int,
                              all_faces: list) -> str:
        body = self.workspace.bodies.get(body_id)
        name = body.name if body else "Body"
        return f"{name}  ·  face {face_idx}"

    def _position_chamfer_panel(self):
        p = getattr(self, '_chamfer_panel', None)
        if p is None:
            return
        margin = 16
        origin = self.mapToGlobal(self.rect().topLeft())
        p.move(origin.x() + margin, origin.y() + margin)

    def _close_chamfer_panel(self):
        panel = getattr(self, '_chamfer_panel', None)
        if panel is not None:
            panel._preview_timer.stop()
            panel.end_pick_face()
            panel.end_pick_edge()
            for sig, slot in [
                (panel.preview_changed,    self._update_chamfer_preview),
                (panel.face_entry_removed, self._on_chamfer_face_removed),
                (panel.edge_entry_removed, self._on_chamfer_edge_removed),
            ]:
                try:
                    sig.disconnect(slot)
                except Exception:
                    pass
            panel.close()
            self._chamfer_panel = None

        if getattr(self, '_editing_chamfer_idx', None) is not None:
            self._cancel_chamfer_edit()

        self._chamfer_preview_token = None
        self._chamfer_preview_mesh  = None
        self._chamfer_computing     = False
        self._chamfer_pending       = None
        self._chamfer_pick_face     = False
        self._chamfer_pick_edge     = False
        self._chamfer_arrow_base    = None
        self._chamfer_arrow_origin  = None
        self._chamfer_arrow_dir     = None
        self.update()

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def _on_chamfer_pick_face(self, active: bool):
        self._chamfer_pick_face = active

    def _on_chamfer_pick_edge(self, active: bool):
        self._chamfer_pick_edge = active

    def _on_chamfer_face_removed(self, index: int):
        indices = getattr(self, '_chamfer_face_indices', [])
        if 0 <= index < len(indices):
            indices.pop(index)
        self._chamfer_face_indices = indices
        if not indices and not getattr(self, '_chamfer_edge_indices', []):
            self._chamfer_body_id = None
        self._chamfer_preview_mesh = None
        self._update_chamfer_arrow()
        panel = getattr(self, '_chamfer_panel', None)
        if panel is not None:
            panel._emit_preview()
        self.update()

    def _on_chamfer_edge_removed(self, index: int):
        indices = getattr(self, '_chamfer_edge_indices', [])
        if 0 <= index < len(indices):
            indices.pop(index)
        self._chamfer_edge_indices = indices
        if not indices and not getattr(self, '_chamfer_face_indices', []):
            self._chamfer_body_id = None
        self._chamfer_preview_mesh = None
        self._update_chamfer_arrow()
        panel = getattr(self, '_chamfer_panel', None)
        if panel is not None:
            panel._emit_preview()
        self.update()

    def route_face_pick_for_chamfer(self, body_id: str, face_idx: int) -> bool:
        panel = getattr(self, '_chamfer_panel', None)
        if panel is None or not getattr(self, '_chamfer_pick_face', False):
            return False
        if not self._chamfer_lock_body(body_id):
            return True

        indices = getattr(self, '_chamfer_face_indices', [])
        if face_idx in indices:
            panel.remove_face_entry(indices.index(face_idx))
        else:
            indices.append(face_idx)
            self._chamfer_face_indices = indices
            shape = self.workspace.current_shape(body_id)
            all_faces = list(shape.faces()) if shape is not None else []
            panel.add_face_entry(body_id, face_idx,
                                 self._chamfer_face_label(body_id, face_idx, all_faces))
            self._update_chamfer_arrow()
            panel._emit_preview()
        return True

    def route_edge_pick_for_chamfer(self, edge_idx: int, body_id: str) -> bool:
        panel = getattr(self, '_chamfer_panel', None)
        if panel is None or not getattr(self, '_chamfer_pick_edge', False):
            return False
        if not self._chamfer_lock_body(body_id):
            return True

        indices = getattr(self, '_chamfer_edge_indices', [])
        if edge_idx in indices:
            panel.remove_edge_entry(indices.index(edge_idx))
        else:
            indices.append(edge_idx)
            self._chamfer_edge_indices = indices
            body = self.workspace.bodies.get(body_id)
            name = body.name if body else "Body"
            panel.add_edge_entry(body_id, edge_idx, f"{name}  ·  edge {edge_idx}")
            self._update_chamfer_arrow()
            panel._emit_preview()
        return True

    def _chamfer_lock_body(self, body_id: str) -> bool:
        locked = getattr(self, '_chamfer_body_id', None)
        if locked is None:
            self._chamfer_body_id = body_id
        elif body_id != locked:
            print("[Chamfer] All selections must be on the same body.")
            return False
        return True

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_chamfer_preview(self):
        body_id      = getattr(self, '_chamfer_body_id', None)
        face_indices = getattr(self, '_chamfer_face_indices', [])
        edge_indices = getattr(self, '_chamfer_edge_indices', [])
        if body_id is None or (not face_indices and not edge_indices):
            self._chamfer_preview_mesh = None
            self.update()
            return

        shape = self.workspace.current_shape(body_id)
        if shape is None:
            self._chamfer_preview_mesh = None
            self.update()
            return

        panel = getattr(self, '_chamfer_panel', None)
        if panel is None:
            return
        dist  = panel._distance_spin.mm_value()
        angle = float(panel._angle_spin.value())
        flip  = panel._flip_chk.isChecked()
        if dist is None or dist <= 0 or not (0.0 < angle < 90.0):
            self._chamfer_preview_mesh = None
            self.update()
            return

        # Resolve TopoDS_Edge objects for direct edge picks (main-thread only)
        edge_occs = []
        live_mesh = self._meshes.get(body_id)
        if live_mesh is not None:
            for ei in edge_indices:
                if ei < len(live_mesh.topo_edges_occ):
                    edge_occs.append(live_mesh.topo_edges_occ[ei])

        params = (list(face_indices), edge_occs, float(dist), angle, flip)

        if getattr(self, '_chamfer_computing', False):
            self._chamfer_pending = params
            return

        panel.clear_entry_errors()
        self._chamfer_computing = True
        self._chamfer_pending   = None
        self._launch_chamfer_thread(shape, params)

    def _launch_chamfer_thread(self, shape, params):
        import threading
        from cad.operations.chamfer import chamfer_edges

        face_indices, edge_occs, dist, angle, flip = params
        token = object()
        self._chamfer_preview_token = token

        # Resolve face-derived edges on the main thread before handing off
        # (build123d traversal must not run from a worker).
        # edge_owners[i] = ('face', face_entry_idx) | ('edge', edge_entry_idx)
        face_edges  = []
        edge_owners = []
        seen = {}                       # id(occ_edge) → owner tuple
        try:
            all_faces = list(shape.faces())
            for face_entry_i, fi in enumerate(face_indices):
                if fi < len(all_faces):
                    for e in all_faces[fi].edges():
                        if id(e.wrapped) not in seen:
                            owner = ('face', face_entry_i)
                            seen[id(e.wrapped)] = owner
                            face_edges.append(e.wrapped)
                            edge_owners.append(owner)
        except Exception:
            pass
        all_edges = list(face_edges)
        for edge_entry_i, e in enumerate(edge_occs):
            if id(e) in seen:
                continue
            owner = ('edge', edge_entry_i)
            seen[id(e)] = owner
            all_edges.append(e)
            edge_owners.append(owner)

        def _compute():
            # Errors come from chamfer_edges itself — no second validate pass.
            # The list is positionally aligned with all_edges/edge_owners.
            per_edge: list[tuple[int, str | None]] = []
            try:
                result = chamfer_edges(shape, all_edges, dist, angle, flip,
                                        per_edge_errors=per_edge)
                from viewer.mesh import Mesh
                preview_mesh = Mesh(result)
                kernel_err   = None
            except Exception as ex:
                preview_mesh = None
                kernel_err   = str(ex)
            # Reshape per_edge → flat list aligned with all_edges; if the
            # native path succeeded chamfer_edges fills None for every entry.
            errors = [None] * len(all_edges)
            for i, msg in per_edge:
                if 0 <= i < len(errors):
                    errors[i] = msg
            QMetaObject.invokeMethod(
                self, "_chamfer_preview_done",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(object, token),
                Q_ARG(object, preview_mesh),
                Q_ARG(object, (edge_owners, errors, kernel_err)),
            )

        threading.Thread(target=_compute, daemon=True).start()

    @pyqtSlot(object, object, object)
    def _chamfer_preview_done(self, token, preview_mesh, validation):
        if getattr(self, '_chamfer_preview_token', None) is not token:
            self._chamfer_computing = False
            return
        if preview_mesh is not None:
            self.makeCurrent()
            try:
                preview_mesh.upload()
            except Exception:
                preview_mesh = None
        self._chamfer_preview_mesh = preview_mesh
        self._apply_chamfer_validation(validation)
        self._reposition_chamfer_arrow()
        self.update()

        pending = getattr(self, '_chamfer_pending', None)
        self._chamfer_pending   = None
        self._chamfer_computing = False
        if pending is not None:
            shape = self.workspace.current_shape(
                getattr(self, '_chamfer_body_id', None))
            if shape is not None:
                self._chamfer_computing = True
                self._launch_chamfer_thread(shape, pending)

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def _on_chamfer_ok(self, distance: float, angle_deg: float, flip: bool):
        body_id      = getattr(self, '_chamfer_body_id', None)
        face_indices = getattr(self, '_chamfer_face_indices', [])
        edge_indices = getattr(self, '_chamfer_edge_indices', [])
        editing_idx  = getattr(self, '_editing_chamfer_idx', None)

        if editing_idx is not None:
            self._editing_chamfer_idx = None
        self._close_chamfer_panel()

        if body_id is None or (not face_indices and not edge_indices):
            return

        from cad.op_types import ChamferOp
        if editing_idx is not None:
            self._editing_chamfer_idx = editing_idx
            self._commit_chamfer_edit(body_id, face_indices, edge_indices,
                                       distance, angle_deg, flip)
            return
        ChamferOp(source_body_id      = body_id,
                  face_indices        = face_indices,
                  edge_indices        = edge_indices,
                  distance            = distance,
                  angle_deg           = angle_deg,
                  flip_reference_face = flip).commit_async(self)

    # ------------------------------------------------------------------
    # Edit / reopen
    # ------------------------------------------------------------------

    def reopen_chamfer(self, history_idx: int):
        entries = self.history.entries
        if history_idx >= len(entries):
            return
        entry = entries[history_idx]
        if entry.operation != "chamfer" or entry.op is None:
            return
        op = entry.op
        entry.editing = True
        self._editing_chamfer_idx = history_idx

        if history_idx > 0:
            self.history.seek(history_idx - 1)
            self._rebuild_body_mesh(op.source_body_id)
        self.history_changed.emit()

        self._show_chamfer_panel(
            op.source_body_id, list(op.face_indices),
            editing_entry=entry,
            distance=op.distance, angle_deg=op.angle_deg,
            flip=op.flip_reference_face)

        panel = getattr(self, '_chamfer_panel', None)
        if panel is not None and op.edge_indices:
            self._chamfer_edge_indices = list(op.edge_indices)
            body = self.workspace.bodies.get(op.source_body_id)
            name = body.name if body else "Body"
            for ei in op.edge_indices:
                panel.add_edge_entry(op.source_body_id, ei,
                                      f"{name}  ·  edge {ei}")

    def _commit_chamfer_edit(self, body_id: str, face_indices: list,
                              edge_indices: list, distance: float,
                              angle_deg: float, flip: bool):
        from cad.op_types import ChamferOp

        idx = getattr(self, '_editing_chamfer_idx', None)
        if idx is None:
            return
        self._editing_chamfer_idx = None

        entries = self.history.entries
        if idx >= len(entries):
            return

        entry = entries[idx]
        entry.editing = False
        entry_id = entry.entry_id

        # Collect the entry group + any split-children.
        group_indices  = [idx]
        group_body_ids = {entry.body_id}
        for j in range(idx + 1, len(entries)):
            if entries[j].params.get("source_entry_id") == entry_id:
                group_indices.append(j)
                group_body_ids.add(entries[j].body_id)

        self.history.seek(max(idx - 1, 0))

        if self.workspace.current_shape(body_id) is None:
            print(f"[Chamfer edit] Source body '{body_id}' has no shape here.")
            self.history.seek(idx)
            entry.editing = False
            self._rebuild_body_mesh(body_id)
            self.history_changed.emit()
            return

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

        ChamferOp(source_body_id      = body_id,
                  face_indices        = face_indices,
                  edge_indices        = edge_indices,
                  distance            = distance,
                  angle_deg           = angle_deg,
                  flip_reference_face = flip).commit_async(self)

    def _cancel_chamfer_edit(self):
        idx = getattr(self, '_editing_chamfer_idx', None)
        self._editing_chamfer_idx = None
        if idx is None:
            return
        entries = self.history.entries
        if idx < len(entries):
            entries[idx].editing = False
        self.history.seek(idx)
        body_id = getattr(self, '_chamfer_body_id', None)
        if body_id is not None:
            self._rebuild_body_mesh(body_id)
        self.history_changed.emit()

    # ------------------------------------------------------------------
    # Per-entry validation feedback
    # ------------------------------------------------------------------

    def _apply_chamfer_validation(self, validation):
        """Mark face/edge entries red if their edges failed to chamfer.

        validation = (edge_owners, errors, kernel_err)
        edge_owners[i] = ('face', face_entry_idx) | ('edge', edge_entry_idx)
        errors[i]      = str | None
        kernel_err     = panel-level message if the combined chamfer itself failed
        """
        panel = getattr(self, '_chamfer_panel', None)
        if panel is None or validation is None:
            return
        edge_owners, errors, kernel_err = validation

        # Aggregate per-owner: keep the first error message we see per entry.
        face_errs: dict[int, str] = {}
        edge_errs: dict[int, str] = {}
        any_individual = False
        for owner, err in zip(edge_owners, errors):
            if err is None:
                continue
            any_individual = True
            kind, idx = owner
            bucket = face_errs if kind == 'face' else edge_errs
            bucket.setdefault(idx, err)

        for idx, msg in face_errs.items():
            panel.set_face_entry_error(idx, msg)
        for idx, msg in edge_errs.items():
            panel.set_edge_entry_error(idx, msg)

        # Kernel-level failure with no per-edge culprit: flag every entry
        # so the user sees *something* is wrong (e.g. distance too large).
        if kernel_err and not any_individual:
            for i in range(len(panel._face_list)):
                panel.set_face_entry_error(i, kernel_err)
            for i in range(len(panel._edge_list)):
                panel.set_edge_entry_error(i, kernel_err)

    # ------------------------------------------------------------------
    # Drag arrow
    # ------------------------------------------------------------------

    def _update_chamfer_arrow(self):
        """Derive base + direction from the first picked face/edge."""
        import numpy as np
        body_id      = getattr(self, '_chamfer_body_id', None)
        face_indices = getattr(self, '_chamfer_face_indices', [])
        edge_indices = getattr(self, '_chamfer_edge_indices', [])

        def _clear():
            self._chamfer_arrow_base   = None
            self._chamfer_arrow_origin = None
            self._chamfer_arrow_dir    = None

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
                from cad.face_ref import _occ_face_anchor_and_normal
                face = all_faces[fi]
                anchor, normal = _occ_face_anchor_and_normal(face.wrapped)
                if anchor is None or normal is None:
                    raise ValueError("no surface anchor / normal")
                base = anchor
            else:
                mesh = self._meshes.get(body_id)
                if mesh is None or edge_indices[0] >= len(mesh.topo_edges):
                    raise IndexError
                pts  = mesh.topo_edges[edge_indices[0]]
                base = np.array(pts[len(pts) // 2], dtype=float)
                from viewer.mesh import edge_outward_normal
                normal = edge_outward_normal(mesh, edge_indices[0])

            n = np.linalg.norm(normal)
            if n < 1e-10:
                raise ValueError
            normal /= n
            self._chamfer_arrow_base = base
            self._chamfer_arrow_dir  = normal
        except Exception:
            _clear(); return

        self._reposition_chamfer_arrow()

    def _reposition_chamfer_arrow(self):
        import numpy as np
        base   = getattr(self, '_chamfer_arrow_base', None)
        normal = getattr(self, '_chamfer_arrow_dir',  None)
        if base is None or normal is None:
            self._chamfer_arrow_origin = None
            return
        panel = getattr(self, '_chamfer_panel', None)
        dist  = (panel._distance_spin.mm_value() or 1.0) if panel else 1.0
        self._chamfer_arrow_origin = np.asarray(base) + np.asarray(normal) * dist

    def _draw_chamfer_preview(self):
        origin    = getattr(self, '_chamfer_arrow_origin', None)
        direction = getattr(self, '_chamfer_arrow_dir',    None)
        if origin is None or direction is None:
            return
        from viewer.drag_arrow import DragArrow
        panel = getattr(self, '_chamfer_panel', None)
        dist  = (panel._distance_spin.mm_value() or 1.0) if panel else 1.0
        scale = self._arrow_scale(dist)
        DragArrow().draw(origin, direction, scale, color=(0.40, 0.85, 0.95))
