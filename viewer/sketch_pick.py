"""
viewer/sketch_pick.py

SketchPickMixin — manages pickable/renderable sketch faces.

Owns:
  _rebuild_sketch_faces()  — called after any history mutation
  _pick_sketch_face()      — ray-tests committed sketch faces
  _draw_sketch_faces()     — renders semi-transparent filled polygons
  _snap_radius_mm()        — converts screen pixels → world mm for snap

Expects self to have: history, _sketch_faces, _sketch,
_selected_sketch_entry, camera, _modelview, _projection, _viewport,
devicePixelRatio(), width(), height()
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import *


class SketchPickMixin:

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def _rebuild_sketch_faces(self):
        """
        Rebuild pickable/renderable faces from all committed sketch entries
        up to the current history cursor.  Called after any history mutation.
        """
        self._sketch_faces.clear()
        cursor = self.history.cursor
        for i, entry in enumerate(self.history.entries):
            if i > cursor:
                break
            if entry.operation != "sketch":
                continue
            se = entry.params.get("sketch_entry")
            if se is None or entry.error:
                continue
            try:
                segs             = se.line_segments()
                loops            = se.closed_loops()
                faces, regions   = se.build_faces()
                print(f"[Sketch] Entry {i}: {len(segs)} segments, "
                      f"{len(loops)} closed loops, {len(faces)} faces built")
                if faces:
                    # Store as list of (face, outer_uvs, hole_uvs_list)
                    self._sketch_faces[i] = [
                        (face, reg[0], reg[1])
                        for face, reg in zip(faces, regions)
                    ]
            except Exception as ex:
                print(f"[Sketch] Could not build faces for entry {i}: {ex}")

    # ------------------------------------------------------------------
    # Pick
    # ------------------------------------------------------------------

    def _pick_sketch_face(self, ray_origin: np.ndarray, ray_dir: np.ndarray
                          ) -> tuple[int | None, int | None]:
        """
        Ray-test all committed sketch faces.
        Returns (history_index, face_index_within_entry) or (None, None).

        Among all hit faces, prefer the one with the smallest area — this
        ensures that nested faces (which are geometrically disjoint rings/
        regions) resolve to the innermost region under the cursor rather
        than a larger outer face that shares the same ray-parameter t.
        """
        from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter
        from OCP.gp import gp_Lin, gp_Pnt, gp_Dir

        d = ray_dir
        o = ray_origin
        try:
            line = gp_Lin(
                gp_Pnt(float(o[0]), float(o[1]), float(o[2])),
                gp_Dir(float(d[0]), float(d[1]), float(d[2])),
            )
        except Exception:
            return None, None

        # Collect all ray hits with their UV intersection point
        from cad.sketch import _uv_point_in_loop

        hits = []   # (t, hidx, fidx, outer_uvs, hole_uvs_list)
        for hidx, face_list in self._sketch_faces.items():
            se = self.history.entries[hidx].params.get("sketch_entry")
            if se is not None and not se.visible:
                continue
            for fidx, (face, outer_uvs, hole_uvs) in enumerate(face_list):
                try:
                    inter = BRepIntCurveSurface_Inter()
                    inter.Init(face.wrapped, line, 1e-6)
                    while inter.More():
                        t = inter.W()
                        if t > 0:
                            hits.append((t, hidx, fidx, outer_uvs, hole_uvs))
                        inter.Next()
                except Exception:
                    pass

        if not hits:
            return None, None

        # All sketch faces are coplanar — t-distance is meaningless for
        # disambiguation.  Project the closest hit point to UV and find
        # the face whose region (outer polygon minus holes) contains it.
        best_t    = min(h[0] for h in hits)
        hit_world = np.array([o[0], o[1], o[2]]) + best_t * np.array([d[0], d[1], d[2]])

        # Collect UV point once per entry (all faces in an entry share the plane)
        se_cache = {}
        uv_cache = {}  # hidx -> uv_pt

        best_hidx = None
        best_fidx = None
        best_area = np.inf

        for t, hidx, fidx, outer_uvs, hole_uvs in hits:
            if hidx not in uv_cache:
                try:
                    se = self.history.entries[hidx].params.get("sketch_entry")
                    if se is not None:
                        delta = hit_world - se.plane_origin
                        uv_cache[hidx] = (float(np.dot(delta, se.plane_x_axis)),
                                          float(np.dot(delta, se.plane_y_axis)))
                except Exception:
                    pass

            uv_pt = uv_cache.get(hidx)
            if uv_pt is None:
                continue

            # Must be inside outer polygon and outside all hole polygons
            if not _uv_point_in_loop(uv_pt, outer_uvs):
                continue
            if any(_uv_point_in_loop(uv_pt, h) for h in hole_uvs):
                continue

            def _uv_area(uvs):
                a = 0.0
                for k in range(len(uvs)):
                    x0,y0=uvs[k]; x1,y1=uvs[(k+1)%len(uvs)]
                    a += x0*y1 - x1*y0
                return abs(a)*0.5

            area = _uv_area(outer_uvs)
            if area < best_area:
                best_area = area
                best_hidx = hidx
                best_fidx = fidx

        # Fallback: if UV test excluded everything, pick closest hit
        if best_hidx is None and hits:
            _, best_hidx, best_fidx, _, _ = min(hits, key=lambda h: h[0])

        if best_hidx is None:
            return None, None

        # Occlusion check: reject if any mesh is closer along the ray than
        # the sketch plane (coplanar face-level sketches are allowed through).
        if self._sketch_plane_occluded(o, d, best_t):
            return None, None

        return best_hidx, best_fidx

    def _sketch_plane_occluded(self, ray_origin: np.ndarray,
                               ray_dir: np.ndarray, plane_t: float) -> bool:
        """
        Return True if any visible body mesh has a ray hit closer than
        plane_t (i.e. solid geometry is between the camera and the plane).
        A small epsilon allows coplanar face-level sketches through.
        """
        from cad.picker import pick_face
        _COPLANAR_EPS = 1e-3
        for body_id, mesh in self._meshes.items():
            body = self.workspace.bodies.get(body_id)
            if body and not body.visible:
                continue
            result = pick_face(mesh, ray_origin, ray_dir, return_t=True)
            if result and result[1] < plane_t - _COPLANAR_EPS:
                return True
        return False

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def _draw_sketch_faces(self):
        """
        Draw committed sketch faces as semi-transparent filled polygons
        with boundary outlines.

        Each face is drawn twice:
          1. GL_LEQUAL  — pixels that pass the depth test (visible part),
             at normal alpha.
          2. GL_GREATER — pixels that fail (occluded by geometry), at
             greatly reduced alpha so the plane ghost is visible but
             clearly behind the object.
        """
        if not self._sketch_faces:
            return

        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GCPnts import GCPnts_UniformAbscissa

        glDisable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)      # never write sketch planes into depth buffer
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Shift the sketch plane slightly toward the camera so it wins over a
        # coplanar body face without z-fighting.  Only applies to the visible
        # pass (GL_LEQUAL) where depth comparison matters.
        glEnable(GL_POLYGON_OFFSET_FILL)
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)

        def _uv_area(uvs):
            a = 0.0
            for k in range(len(uvs)):
                x0, y0 = uvs[k]; x1, y1 = uvs[(k+1) % len(uvs)]
                a += x0 * y1 - x1 * y0
            return abs(a) * 0.5

        draw_list = []
        for hidx, face_list in self._sketch_faces.items():
            se = self.history.entries[hidx].params.get("sketch_entry")
            if se is not None and not se.visible:
                continue
            for fidx, (face, outer_uvs, hole_uvs) in enumerate(face_list):
                draw_list.append((hidx, fidx, face, _uv_area(outer_uvs)))
        draw_list.sort(key=lambda x: x[3], reverse=True)

        def _draw_tris(facing, alpha):
            glBegin(GL_TRIANGLES)
            for tri_idx in range(1, facing.NbTriangles() + 1):
                tri = facing.Triangle(tri_idx)
                n1, n2, n3 = tri.Get()
                for ni in (n1, n2, n3):
                    node = facing.Node(ni)
                    glVertex3f(node.X(), node.Y(), node.Z())
            glEnd()

        def _draw_edges(face_wrapped, line_alpha, line_width):
            glLineWidth(line_width)
            exp = TopExp_Explorer(face_wrapped, TopAbs_EDGE)
            while exp.More():
                edge = exp.Current()
                try:
                    adaptor = BRepAdaptor_Curve(edge)
                    disc    = GCPnts_UniformAbscissa()
                    disc.Initialize(adaptor, 32)
                    if disc.IsDone() and disc.NbPoints() >= 2:
                        glBegin(GL_LINE_STRIP)
                        for pi in range(1, disc.NbPoints() + 1):
                            p = adaptor.Value(disc.Parameter(pi))
                            glVertex3f(p.X(), p.Y(), p.Z())
                        glEnd()
                except Exception:
                    pass
                exp.Next()
            glLineWidth(1.0)

        for hidx, fidx, face, _ in draw_list:
            entry_selected = (hidx == self._selected_sketch_entry)
            sel_faces = self._selected_sketch_face
            is_selected = entry_selected and (
                sel_faces is None or
                fidx in sel_faces
            )
            try:
                BRepMesh_IncrementalMesh(face.wrapped, 0.1)
                location = face.wrapped.Location()
                facing   = BRep_Tool.Triangulation_s(face.wrapped, location)
                if facing is None:
                    continue

                fill_alpha  = 0.45 if is_selected else 0.20
                edge_alpha  = 0.90 if is_selected else 0.50
                line_width  = 1.8  if is_selected else 1.2
                occ_scale   = 0.18  # occluded regions are much dimmer

                # --- Pass 1: visible (in front of or at geometry depth) ------
                glDepthFunc(GL_LEQUAL)
                glColor4f(0.29, 0.43, 0.54, fill_alpha)
                _draw_tris(facing, fill_alpha)

                if is_selected:
                    glColor4f(0.48, 0.70, 0.84, edge_alpha)
                else:
                    glColor4f(0.29, 0.43, 0.54, edge_alpha)
                _draw_edges(face.wrapped, edge_alpha, line_width)

                # --- Pass 2: occluded (behind geometry) ----------------------
                glDepthFunc(GL_GREATER)
                glColor4f(0.29, 0.43, 0.54, fill_alpha * occ_scale)
                _draw_tris(facing, fill_alpha * occ_scale)

                occ_edge_alpha = edge_alpha * occ_scale
                if is_selected:
                    glColor4f(0.48, 0.70, 0.84, occ_edge_alpha)
                else:
                    glColor4f(0.29, 0.43, 0.54, occ_edge_alpha)
                _draw_edges(face.wrapped, occ_edge_alpha, max(line_width - 0.4, 0.8))

            except Exception as ex:
                print(f"[Sketch] draw_sketch_faces error: {ex}")

        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_POLYGON_OFFSET_LINE)
        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)

    def _draw_extrude_preview(self):
        """
        Draw semi-transparent extrude preview solids.
        Blue for positive (add), red for negative (cut).

        The solids are tessellated to flat vertex buffers ONCE per preview
        change (keyed by the solid-list identity) and cached on the widget.
        Re-meshing the OCCT solid and walking it in immediate mode every frame
        was the dominant lag during a drag (>100 ms/frame on a ~1.2 m drafted
        extrude); the preview geometry only changes when the user drags, so
        there is no reason to rebuild it on every repaint.
        """
        solids = getattr(self, '_extrude_preview_mesh', None)
        if not solids:
            self._extrude_preview_buf = None
            return

        dist = getattr(self, '_extrude_preview_dist', 0.0)
        is_cut = dist < 0

        buf = self._extrude_preview_render_buffers(solids)
        if buf is None:
            return

        glDisable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        from cad.prefs import prefs as _prefs
        op = _prefs.op_preview_opacity
        if is_cut:
            fill_color   = (0.75, 0.18, 0.18, op)
            edge_color   = (1.00, 0.35, 0.35, min(op + 0.35, 1.0))
        else:
            r, g, b = _prefs.op_preview_color
            fill_color   = (r, g, b, op)
            edge_color   = (min(r + 0.23, 1.0), min(g + 0.30, 1.0), min(b + 0.15, 1.0), min(op + 0.35, 1.0))

        tris, edges = buf
        glEnableClientState(GL_VERTEX_ARRAY)
        if len(tris):
            glColor4f(*fill_color)
            glVertexPointer(3, GL_FLOAT, 0, tris)
            glDrawArrays(GL_TRIANGLES, 0, len(tris) // 3)
        if len(edges):
            glColor4f(*edge_color)
            glLineWidth(1.4)
            glVertexPointer(3, GL_FLOAT, 0, edges)
            glDrawArrays(GL_LINES, 0, len(edges) // 3)
            glLineWidth(1.0)
        glDisableClientState(GL_VERTEX_ARRAY)

        glDisable(GL_BLEND)
        glEnable(GL_CULL_FACE)
        glEnable(GL_LIGHTING)

        self._draw_extrude_arrow()

    def _extrude_preview_render_buffers(self, solids):
        """Return (tri_verts, edge_verts) flat float32 arrays for `solids`,
        rebuilding only when the solid list changes. Cached on the widget so
        repeated repaints during a drag are nearly free."""
        import numpy as np
        cache = getattr(self, '_extrude_preview_buf', None)
        if cache is not None and cache[0] is solids:
            return cache[1]

        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
        from OCP.TopoDS import TopoDS
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GCPnts import GCPnts_UniformAbscissa

        tri_pts: list = []
        edge_pts: list = []
        for solid in solids:
            try:
                wrapped = solid.wrapped
                # Size-adaptive deflection: absolute 0.15 mm exploded triangle
                # counts on metre-scale parts. 0.2% of the bbox diagonal keeps
                # the preview cheap and visually identical at any scale.
                from viewer.mesh import _bbox_diagonal
                defl = max(0.15, _bbox_diagonal(solid) * 0.002)
                BRepMesh_IncrementalMesh(wrapped, defl)

                exp = TopExp_Explorer(wrapped, TopAbs_FACE)
                while exp.More():
                    face = TopoDS.Face_s(exp.Current())
                    loc  = face.Location()
                    tri  = BRep_Tool.Triangulation_s(face, loc)
                    if tri is not None:
                        trsf = None if loc.IsIdentity() else loc.Transformation()
                        for i in range(1, tri.NbTriangles() + 1):
                            n1, n2, n3 = tri.Triangle(i).Get()
                            for ni in (n1, n2, n3):
                                p = tri.Node(ni)
                                if trsf is not None:
                                    p = p.Transformed(trsf)
                                tri_pts.append((p.X(), p.Y(), p.Z()))
                    exp.Next()

                exp2 = TopExp_Explorer(wrapped, TopAbs_EDGE)
                while exp2.More():
                    try:
                        # Explorer yields TopoDS_Shape; BRepAdaptor_Curve needs
                        # a TopoDS_Edge — without the cast it raises TypeError,
                        # which the old code silently swallowed (dropping every
                        # preview edge). Cast so edges actually render.
                        edge = TopoDS.Edge_s(exp2.Current())
                        adaptor = BRepAdaptor_Curve(edge)
                        disc    = GCPnts_UniformAbscissa()
                        disc.Initialize(adaptor, 24)
                        if disc.IsDone() and disc.NbPoints() >= 2:
                            prev = None
                            for pi in range(1, disc.NbPoints() + 1):
                                p = adaptor.Value(disc.Parameter(pi))
                                cur = (p.X(), p.Y(), p.Z())
                                # GL_LINES: emit each segment as a pair
                                if prev is not None:
                                    edge_pts.append(prev)
                                    edge_pts.append(cur)
                                prev = cur
                    except Exception:
                        pass
                    exp2.Next()
            except Exception as ex:
                print(f"[Preview] build error: {ex}")

        tris  = np.array(tri_pts,  dtype=np.float32).reshape(-1) if tri_pts else np.zeros(0, np.float32)
        edges = np.array(edge_pts, dtype=np.float32).reshape(-1) if edge_pts else np.zeros(0, np.float32)
        buf = (tris, edges)
        self._extrude_preview_buf = (solids, buf)
        return buf

    # ------------------------------------------------------------------
    # Screen-space snap radius
    # ------------------------------------------------------------------

    def _snap_radius_mm(self, snap_pixels: float = 14.0) -> float:
        """
        Convert a fixed pixel radius to world-mm at the sketch plane depth.
        Gives consistent snap feel regardless of zoom level.
        """
        from OpenGL.GLU import gluUnProject

        if self._modelview is None or self._sketch is None:
            return self.camera.distance * 0.015
        try:
            dpr = self.devicePixelRatio()
            cx  = self.width()  * 0.5 * dpr
            cy  = self.height() * 0.5 * dpr

            near0 = np.array(gluUnProject(cx,       cy, 0.0,
                             self._modelview, self._projection, self._viewport))
            near1 = np.array(gluUnProject(cx + 1.0, cy, 0.0,
                             self._modelview, self._projection, self._viewport))
            far0  = np.array(gluUnProject(cx,       cy, 1.0,
                             self._modelview, self._projection, self._viewport))
            far1  = np.array(gluUnProject(cx + 1.0, cy, 1.0,
                             self._modelview, self._projection, self._viewport))

            plane = self._sketch.plane
            _, pt0 = plane.ray_intersect(near0, far0 - near0)
            _, pt1 = plane.ray_intersect(near1, far1 - near1)

            if pt0 is not None and pt1 is not None:
                mm_per_px = float(np.linalg.norm(pt1 - pt0))
                return max(0.01, mm_per_px * snap_pixels)
        except Exception:
            pass
        return self.camera.distance * 0.015
