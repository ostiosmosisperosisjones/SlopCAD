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
from PyQt6.QtCore import pyqtSlot

class RevolveMixin:

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def _try_revolve(self):
        if self._selected_sketch_entry is not None:
            self._show_revolve_panel(sketch_idx=self._selected_sketch_entry)
            return
        if self.selection.face_count > 0:
            sf = self.selection.single_face or self.selection.faces[0]
            self._show_revolve_panel(body_id=sf.body_id, face_idx=sf.face_idx)
        else:
            self._show_revolve_panel()

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

        if body_id is not None and face_idx is not None:
            body = self.workspace.bodies.get(body_id)
            label = f"{body.name}  ·  face {face_idx}" if body else "⚠  face lost"
            panel.add_face_entry(body_id, face_idx, label,
                                 valid=body is not None)
        elif sketch_idx is not None:
            panel.add_face_entry(None, None, f"Sketch {sketch_idx}")

        panel.revolve_requested.connect(self._on_revolve_panel_ok)
        panel.cancelled.connect(self._close_revolve_panel)
        panel.picking_axis_changed.connect(self._on_revolve_pick_axis)
        panel.picking_face_changed.connect(self._on_revolve_pick_face)
        panel.picking_body_changed.connect(self._on_revolve_pick_body)
        panel.preview_changed.connect(self._on_revolve_preview)

        self._revolve_panel          = panel
        self._revolve_axis_active    = False
        self._revolve_face_active    = False
        self._revolve_body_active    = False
        self._revolve_preview_mesh   = None

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
            self._revolve_panel.close()
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

        faces = []
        if sketch_idx is not None:
            all_sketch = self._sketch_faces.get(sketch_idx, [])
            fidx_sel   = self._selected_sketch_face
            faces = ([all_sketch[i][0] for i in fidx_sel if 0 <= i < len(all_sketch)]
                     if fidx_sel is not None else [f[0] for f in all_sketch])
        elif face_pairs:
            for bid, fi in face_pairs:
                shape = self.workspace.current_shape(bid)
                if shape is None:
                    continue
                all_f = list(shape.faces())
                if fi < len(all_f):
                    faces.append(all_f[fi])

        if not faces:
            self._revolve_cage_wires   = None
            self._revolve_arrow_origin = None
            self._revolve_arrow_dir    = None
            self.update()
            return

        # Re-extract cage wires only when axis or faces change, not every angle tick.
        cache_key = (tuple(axis_pt), tuple(axis_d), tuple(id(f) for f in faces))
        if cache_key != getattr(self, '_revolve_cage_key', None):
            self._revolve_cage_wires = _extract_face_wires(faces)
            self._revolve_cage_key   = cache_key
            self._revolve_preview_mesh = None  # axis/face changed — old mesh is wrong

        # Store for use by mesh compute on release.
        self._revolve_last_preview_params = (faces, axis_pt, axis_d, angle)

        self._update_revolve_arrow(faces, axis_pt, axis_d, angle)

        # During arrow drag show cage only. When not dragging (spinbox, axis pick)
        # kick off a mesh compute so the user sees a proper solid preview.
        if not (getattr(self, '_drag_arrow_active', False)
                and getattr(self, '_drag_arrow_op', None) == 'revolve'):
            self._revolve_compute_mesh(faces, axis_pt, axis_d, angle)

        self.update()

    def _revolve_compute_mesh(self, faces, axis_pt, axis_d, angle):
        """Spawn a background worker to build the mesh. One at a time — newer
        params overwrite pending; stale results are dropped via gen counter."""
        gen = getattr(self, '_revolve_mesh_gen', 0) + 1
        self._revolve_mesh_gen     = gen
        self._revolve_mesh_pending = (faces, axis_pt, axis_d, angle, gen)
        if getattr(self, '_revolve_mesh_inflight', False):
            return
        self._revolve_mesh_inflight = True
        self._revolve_mesh_gen      = gen

        def _compute():
            from cad.operations.revolve import _do_revolve_solid
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.BRep import BRep_Tool
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
            from OCP.TopoDS import TopoDS
            from OCP.TopLoc import TopLoc_Location
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            from OCP.GCPnts import GCPnts_UniformAbscissa
            solids   = [_do_revolve_solid(f, axis_pt, axis_d, angle) for f in faces]
            tris_list  = []
            edges_list = []
            identity   = TopLoc_Location()
            for solid in solids:
                wrapped = solid.wrapped
                BRepMesh_IncrementalMesh(wrapped, 1.0)
                tris = []
                exp = TopExp_Explorer(wrapped, TopAbs_FACE)
                while exp.More():
                    face = TopoDS.Face_s(exp.Current())
                    tri  = BRep_Tool.Triangulation_s(face, identity)
                    if tri is not None:
                        for i in range(1, tri.NbTriangles() + 1):
                            n1, n2, n3 = tri.Triangle(i).Get()
                            for ni in (n1, n2, n3):
                                p = tri.Node(ni)
                                tris.append((p.X(), p.Y(), p.Z()))
                    exp.Next()
                tris_list.append(tris)
                edges = []
                exp2 = TopExp_Explorer(wrapped, TopAbs_EDGE)
                while exp2.More():
                    edge = TopoDS.Edge_s(exp2.Current())
                    try:
                        adp  = BRepAdaptor_Curve(edge)
                        disc = GCPnts_UniformAbscissa()
                        disc.Initialize(adp, 16)
                        if disc.IsDone() and disc.NbPoints() >= 2:
                            pts = []
                            for pi in range(1, disc.NbPoints() + 1):
                                p = adp.Value(disc.Parameter(pi))
                                pts.append((p.X(), p.Y(), p.Z()))
                            edges.append(pts)
                    except Exception:
                        pass
                    exp2.Next()
                edges_list.append(edges)
            return tris_list, edges_list

        def _deliver(result, finished_gen):
            self._revolve_mesh_inflight = False
            if getattr(self, '_revolve_mesh_gen', 0) == finished_gen:
                self._revolve_preview_mesh = result
                self.update()
            # Fire once more if newer params arrived while we were computing.
            pending = getattr(self, '_revolve_mesh_pending', None)
            if pending is not None and pending[4] != finished_gen:
                pfaces, ppt, pd, pangle, _ = pending
                self._revolve_compute_mesh(pfaces, ppt, pd, pangle)

        def _worker():
            try:
                result = _compute()
                err    = None
            except Exception as ex:
                result = None
                err    = ex
            if err is not None:
                print(f"[Revolve preview] {err}")
            from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(
                self, "_revolve_mesh_deliver",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(object, _deliver),
                Q_ARG(object, result),
                Q_ARG(object, gen),
            )

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    @pyqtSlot(object, object, object)
    def _revolve_mesh_deliver(self, deliver_fn, result, finished_gen):
        deliver_fn(result, finished_gen)

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
        if r < 1e-10:
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
        from OpenGL.GL import (glDisable, glEnable, glColor4f, glBegin, glEnd,
                               glVertex3f, glLineWidth, glBlendFunc,
                               GL_LIGHTING, GL_DEPTH_TEST, GL_CULL_FACE,
                               GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                               GL_TRIANGLES, GL_LINE_STRIP)
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
            tris_list, edges_list = mesh_data
            glColor4f(r, g, b, op)
            for tris in tris_list:
                if tris:
                    glBegin(GL_TRIANGLES)
                    for (x, y, z) in tris:
                        glVertex3f(x, y, z)
                    glEnd()
            glColor4f(*edge_color)
            glLineWidth(1.4)
            for edges in edges_list:
                for pts in edges:
                    glBegin(GL_LINE_STRIP)
                    for (x, y, z) in pts:
                        glVertex3f(x, y, z)
                    glEnd()
            glLineWidth(1.0)
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
            from cad.sketch import LineEntity
            ent = se.entities[entity_idx]
            if not isinstance(ent, LineEntity):
                print("[Revolve] Axis pick requires a line entity.")
                return False

            origin = np.array(se.plane_origin,  dtype=float)
            x_axis = np.array(se.plane_x_axis,  dtype=float)
            y_axis = np.array(se.plane_y_axis,  dtype=float)

            p0 = origin + float(ent.p0[0]) * x_axis + float(ent.p0[1]) * y_axis
            p1 = origin + float(ent.p1[0]) * x_axis + float(ent.p1[1]) * y_axis
            direction = p1 - p0
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                print("[Revolve] Degenerate sketch line — cannot use as axis.")
                return False
            direction /= norm

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
        try:
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            adp = BRepAdaptor_Curve(mesh.topo_edges_occ[edge_idx])
            mid = (adp.FirstParameter() + adp.LastParameter()) * 0.5
            pt  = adp.Value(mid)
            d   = adp.DN(mid, 1)
            axis_pt  = np.array([pt.X(), pt.Y(), pt.Z()], dtype=float)
            axis_dir = np.array([d.X(),  d.Y(),  d.Z()],  dtype=float)
            norm = np.linalg.norm(axis_dir)
            if norm < 1e-10:
                return False
            axis_dir /= norm
        except Exception as ex:
            print(f"[Revolve] Edge axis failed: {ex}")
            return False

        panel.set_axis(axis_pt, axis_dir)
        return True

    def route_face_pick_for_revolve(self, body_id: str, face_idx: int) -> bool:
        if not getattr(self, '_revolve_face_active', False):
            return False
        panel = getattr(self, '_revolve_panel', None)
        if panel is None:
            return False
        body = self.workspace.bodies.get(body_id)
        if body is None:
            return False
        label = f"{body.name}  ·  face {face_idx}"
        panel.add_face_entry(body_id, face_idx, label)
        self._revolve_face_pairs.append((body_id, face_idx))
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
        from cad.op_types import (SketchRevolveOp, FaceRevolveOp,
                                  CrossBodyRevolveCutOp)

        sketch_idx  = getattr(self, '_revolve_sketch_idx', None)
        face_pairs  = getattr(self, '_revolve_face_pairs', [])
        editing_idx = getattr(self, '_editing_history_idx', None)

        axis_pt  = list(map(float, axis_point))
        axis_d   = list(map(float, axis_dir))
        angle    = float(angle_deg)

        # Prevent _close_revolve_panel from cancelling the edit
        if editing_idx is not None:
            self._editing_history_idx = None

        self._close_revolve_panel()

        # Editing path: defer all branching to _commit_revolve_edit so the
        # seek/delete/replay bookkeeping stays in one place.
        if editing_idx is not None:
            self._editing_history_idx = editing_idx  # restore for _commit
            self._commit_revolve_edit_from_panel(
                editing_idx, angle, axis_pt, axis_d, merge_body_id, is_cut,
                sketch_idx, face_pairs)
            return

        # Cut paths --------------------------------------------------------
        if is_cut:
            if sketch_idx is None:
                # Face-driven revolve cuts not yet supported (would need
                # CrossBodyRevolveCutOp to accept a body face profile).
                print("[Revolve cut] Face-driven revolve cuts not supported yet.")
                return
            entries = self.history.entries
            if sketch_idx >= len(entries):
                print("[Revolve] Invalid sketch index.")
                return
            sketch_id = entries[sketch_idx].entry_id
            se = entries[sketch_idx].params.get("sketch_entry")
            src_body_id = se.body_id if se else None
            if merge_body_id in (None, "__new_body__"):
                self._do_revolve_cut_all_intersecting(
                    angle, axis_pt, axis_d, sketch_id, src_body_id)
                return
            op = CrossBodyRevolveCutOp(
                cut_body_id      = merge_body_id,
                source_body_id   = src_body_id,
                source_sketch_id = sketch_id,
                angle_deg        = angle,
                axis_point       = axis_pt,
                axis_dir         = axis_d,
            )
            op.commit(self)
            return

        # Revolve (merge / new body) ---------------------------------------
        if sketch_idx is not None:
            entries = self.history.entries
            if sketch_idx >= len(entries):
                print("[Revolve] Invalid sketch index.")
                return
            force_new = (merge_body_id is None or merge_body_id == "__new_body__")
            new_op = SketchRevolveOp(
                from_sketch_id = entries[sketch_idx].entry_id,
                angle_deg      = angle,
                axis_point     = axis_pt,
                axis_dir       = axis_d,
                merge_body_id  = None if force_new else merge_body_id,
            )
        elif face_pairs:
            body_id, face_idx = face_pairs[0]
            new_op = FaceRevolveOp(
                source_body_id = body_id,
                face_idx       = face_idx,
                angle_deg      = angle,
                axis_point     = axis_pt,
                axis_dir       = axis_d,
            )
        else:
            print("[Revolve] No profile selected.")
            return

        new_op.commit_async(self)

    def _do_revolve_cut_all_intersecting(self, angle_deg, axis_pt, axis_d,
                                          sketch_id, src_body_id):
        """
        No-target revolve cut: build the revolved tool solid once, fan out
        one CrossBodyRevolveCutOp per workspace body it intersects.
        """
        import numpy as np
        from cad.op_types import CrossBodyRevolveCutOp
        from cad.operations.revolve import _do_revolve_solid
        from cad.cut_all import fan_out_cut
        from build123d import Compound
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape

        # Resolve sketch profile faces from the cache
        sketch_idx = self.history.id_to_index(sketch_id)
        if sketch_idx is None:
            print("[Revolve cut] Sketch entry missing."); return
        all_sketch = self._sketch_faces.get(sketch_idx, [])
        if not all_sketch:
            print("[Revolve cut] Sketch has no faces."); return
        fidx_sel = self._selected_sketch_face
        if fidx_sel is not None:
            tool_faces = [all_sketch[i][0] for i in fidx_sel
                          if 0 <= i < len(all_sketch)]
        else:
            tool_faces = [f[0] for f in all_sketch]
        if not tool_faces:
            print("[Revolve cut] No tool faces."); return

        axis_pt_arr  = np.asarray(axis_pt, dtype=float)
        axis_dir_arr = np.asarray(axis_d,  dtype=float)

        try:
            tool_solid = None
            for face in tool_faces:
                s = _do_revolve_solid(face, axis_pt_arr, axis_dir_arr, angle_deg)
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
            print(f"[Revolve cut] Tool construction failed: {ex}"); return

        def build_op(bid: str):
            return CrossBodyRevolveCutOp(
                cut_body_id      = bid,
                source_body_id   = src_body_id,
                source_sketch_id = sketch_id,
                angle_deg        = angle_deg,
                axis_point       = list(axis_pt),
                axis_dir         = list(axis_d),
            )

        fan_out_cut(self, tool_solid, build_op, None, op_label="Revolve cut")

    def _commit_revolve_edit_from_panel(self, editing_idx, angle, axis_pt,
                                         axis_d, merge_body_id, is_cut,
                                         sketch_idx, face_pairs):
        """
        Bridge from panel OK to the edit-commit routine: builds the new op
        (or dispatches to fan-out for no-target cuts) so _commit_revolve_edit
        deletes the old entry group correctly in every mode.
        """
        from cad.op_types import (SketchRevolveOp, FaceRevolveOp,
                                  CrossBodyRevolveCutOp)

        # No-target / self cut: special-case because we need to call
        # _do_revolve_cut_all_intersecting which pushes per-body entries
        # instead of producing a single new_op.
        if is_cut and merge_body_id in (None, "__new_body__"):
            if sketch_idx is None:
                print("[Revolve cut] Face-driven revolve cuts not supported yet.")
                return
            entries = self.history.entries
            if sketch_idx >= len(entries):
                return
            sketch_id = entries[sketch_idx].entry_id
            se = entries[sketch_idx].params.get("sketch_entry")
            src_body_id = se.body_id if se else None

            self._editing_history_idx = None
            self._delete_revolve_edit_group(editing_idx)
            self._do_revolve_cut_all_intersecting(
                angle, axis_pt, axis_d, sketch_id, src_body_id)
            new_idx = self.history.cursor
            ok, err, _ = self.history.replay_all_from(new_idx + 1)
            if not ok:
                print(f"[Revolve Edit] Downstream replay failed: {err}")
            # Advance cursor to tip so bodies created downstream are visible.
            self.history.seek(len(self.history.entries) - 1)
            self._rebuild_all_meshes()
            self.history_changed.emit()
            return

        # Cross-body cut with a specific target
        if is_cut:
            if sketch_idx is None:
                print("[Revolve cut] Face-driven revolve cuts not supported yet.")
                return
            entries = self.history.entries
            sketch_id = entries[sketch_idx].entry_id
            se = entries[sketch_idx].params.get("sketch_entry")
            src_body_id = se.body_id if se else None
            new_op = CrossBodyRevolveCutOp(
                cut_body_id      = merge_body_id,
                source_body_id   = src_body_id,
                source_sketch_id = sketch_id,
                angle_deg        = angle,
                axis_point       = axis_pt,
                axis_dir         = axis_d,
            )
            self._commit_revolve_edit(editing_idx, new_op)
            return

        # Plain revolve (merge or new body)
        if sketch_idx is not None:
            entries = self.history.entries
            force_new = (merge_body_id is None or merge_body_id == "__new_body__")
            new_op = SketchRevolveOp(
                from_sketch_id = entries[sketch_idx].entry_id,
                angle_deg      = angle,
                axis_point     = axis_pt,
                axis_dir       = axis_d,
                merge_body_id  = None if force_new else merge_body_id,
            )
        elif face_pairs:
            body_id, face_idx = face_pairs[0]
            new_op = FaceRevolveOp(
                source_body_id = body_id,
                face_idx       = face_idx,
                angle_deg      = angle,
                axis_point     = axis_pt,
                axis_dir       = axis_d,
            )
        else:
            print("[Revolve] No profile selected.")
            return

        self._commit_revolve_edit(editing_idx, new_op)

    def _delete_revolve_edit_group(self, idx: int):
        """Seek to idx-1, delete the group, remove created bodies (shared
        bookkeeping used by both _commit_revolve_edit and the no-target cut
        edit path)."""
        entries = self.history.entries
        if idx >= len(entries):
            return
        entry = entries[idx]
        entry.editing = False

        entry_id = entry.entry_id
        group_indices  = [idx]
        group_body_ids = {entry.body_id}
        for j in range(idx + 1, len(entries)):
            e = entries[j]
            if e.params.get("source_entry_id") == entry_id:
                group_indices.append(j)
                group_body_ids.add(e.body_id)

        self.history.seek(max(idx - 1, 0))

        child_body_ids = entry.params.get("child_body_ids", [])
        group_entry_ids = {entries[j].entry_id for j in group_indices}
        removable = set()
        for bid in group_body_ids | set(child_body_ids):
            body = self.workspace.bodies.get(bid)
            if body is not None and body.created_at_entry_id in group_entry_ids:
                removable.add(bid)
        for j in reversed(group_indices):
            self.history.delete(j)
        for bid in removable:
            if bid in self.workspace.bodies:
                self.workspace.remove_body(bid)

    def _commit_revolve_edit(self, idx: int, new_op):
        self._editing_history_idx = None
        self._editing_body_id = None
        self._delete_revolve_edit_group(idx)

        new_op.commit(self)

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
            edge = TopoDS.Edge_s(exp.Current())
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
