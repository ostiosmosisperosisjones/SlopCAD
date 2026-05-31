"""
viewer/mesh.py

Mesh in true world coordinates — no centering or normalization.
Vertices are exactly as OCCT/build123d gives them (millimeters).
The camera fits itself to the scene bounding box instead.
"""

import numpy as np
from ocp_tessellate.tessellator import tessellate
from OpenGL.GL import *


class Mesh:
    def __init__(self, shape, deviation: float | None = None):
        # Use a stable cache key based on the shape's OCCT hash so the
        # tessellator can reuse cached results for identical shapes (e.g. undo/redo).
        cache_key = f"mesh_{hash(shape.wrapped)}"
        self.occt_faces = list(shape.faces())
        self.shape      = shape

        # Tessellation deviation is a CHORDAL tolerance in millimetres, so a
        # fixed absolute value (e.g. 0.01) demands a fraction-of-a-percent
        # relative precision on a large part — which explodes the triangle
        # count and tessellation time (a ~1.9 m part hit 800k triangles / 25 s
        # at deviation=0.01). Scale the deviation to the part's bounding-box
        # diagonal so quality is roughly constant regardless of size, with an
        # absolute floor so small parts stay crisp and an angular tolerance
        # that keeps curves smooth either way.
        diag = _bbox_diagonal(shape)
        base = max(0.05, diag * 0.001)   # 0.1% of size, ≥0.05 mm
        # Retry with progressively looser tolerances if the tessellator returns
        # a face count that doesn't match shape.faces() (ocp_tessellate bug).
        # A caller-supplied deviation skips to that quality tier immediately.
        _attempts = [
            dict(deviation=base,       quality=base,       angular_tolerance=0.3),
            dict(deviation=base * 3,   quality=base * 3,   angular_tolerance=0.5),
            dict(deviation=base * 8,   quality=base * 8,   angular_tolerance=0.8),
        ]
        if deviation is not None:
            _attempts = [dict(deviation=deviation, quality=deviation,
                              angular_tolerance=deviation * 5)] + _attempts
        from cad.profiler import profiler
        tess = None
        for attempt in _attempts:
            with profiler.section("mesh.tessellate"):
                t = tessellate(shape.wrapped, cache_key=cache_key, **attempt)
            tri_per_face = np.array(t['triangles_per_face'], dtype=np.int32)
            if len(tri_per_face) == len(self.occt_faces):
                tess = t
                self.triangles_per_face = tri_per_face
                break
            # bust the cache key so next attempt re-tessellates
            cache_key = cache_key + "_r"

        if tess is None:
            raise RuntimeError(
                f"Face count mismatch after all retry attempts: tessellator gave "
                f"{len(tri_per_face)} faces but shape.faces() gave {len(self.occt_faces)}."
            )

        self.verts   = np.array(tess['vertices'], dtype=np.float32).reshape(-1, 3)
        self.tris    = np.array(tess['triangles'], dtype=np.uint32).reshape(-1, 3)
        self.normals = np.array(tess['normals'],   dtype=np.float32).reshape(-1, 3)
        self.edges   = np.array(tess['edges'],     dtype=np.float32).reshape(-1, 3)
        self.tri_count = len(self.tris) * 3

        # World-space bounding box
        self.bbox_min = self.verts.min(axis=0)
        self.bbox_max = self.verts.max(axis=0)

        # True topological vertices — actual CAD corners where edges meet.
        with profiler.section("mesh.topo_verts"):
            self.topo_verts = _extract_topo_verts(shape)

        # True topological edges — each is a (N,3) polyline in world mm.
        # topo_edges_occ is the parallel list of raw TopoDS_Edge objects,
        # used when projecting reference geometry to preserve true curve type.
        with profiler.section("mesh.topo_edges"):
            self.topo_edges, self.topo_edges_occ, self.topo_edge_face_normals = \
                _extract_topo_edges(shape)

        # Legacy compatibility
        self.center = np.array([0.0, 0.0, 0.0])
        self.scale  = 1.0

        self.vbo_verts   = None
        self.vbo_normals = None
        self.vbo_edges   = None
        self.ebo         = None

    def upload(self):
        self.vbo_verts = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_verts)
        glBufferData(GL_ARRAY_BUFFER, self.verts.nbytes, self.verts, GL_STATIC_DRAW)

        self.vbo_normals = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glBufferData(GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL_STATIC_DRAW)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        indices = self.tris.flatten().astype(np.uint32)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        self.vbo_edges = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_edges)
        glBufferData(GL_ARRAY_BUFFER, self.edges.nbytes,
                     self.edges.astype(np.float32), GL_STATIC_DRAW)
        self.edge_count = len(self.edges)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_verts)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
        glNormalPointer(GL_FLOAT, 0, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glDrawElements(GL_TRIANGLES, self.tri_count, GL_UNSIGNED_INT, None)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw_edges(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_edges)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_LINES, 0, self.edge_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def get_face_triangle_range(self, face_idx):
        start = int(self.triangles_per_face[:face_idx].sum())
        count = int(self.triangles_per_face[face_idx])
        return start, count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_diagonal(shape) -> float:
    """Bounding-box diagonal length in mm, used to size tessellation tolerance.

    Falls back to a neutral default if the bbox can't be computed (degenerate
    or empty shape) so callers always get a usable, finite number."""
    try:
        bb = shape.bounding_box()
        dx, dy, dz = bb.max.X - bb.min.X, bb.max.Y - bb.min.Y, bb.max.Z - bb.min.Z
        diag = (dx * dx + dy * dy + dz * dz) ** 0.5
        if diag > 1e-6 and np.isfinite(diag):
            return float(diag)
    except Exception:
        pass
    return 100.0


def _extract_topo_verts(shape) -> np.ndarray:
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopoDS import TopoDS

    seen = set()
    pts  = []

    explorer = TopExp_Explorer(shape.wrapped, TopAbs_VERTEX)
    while explorer.More():
        vert = TopoDS.Vertex_s(explorer.Current())
        pnt  = BRep_Tool.Pnt_s(vert)
        key  = (round(pnt.X(), 4), round(pnt.Y(), 4), round(pnt.Z(), 4))
        if key not in seen:
            seen.add(key)
            pts.append([pnt.X(), pnt.Y(), pnt.Z()])
        explorer.Next()

    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def _extract_topo_edges(shape) -> tuple[list[np.ndarray], list, list]:
    """
    Extract topological edges from a build123d shape via OCCT.

    Uses curvature-adaptive sampling (GCPnts_TangentialDeflection) so straight
    lines get 2 points and curves get only as many as needed to look smooth.

    Returns
    -------
    topo_edges            : list of (N, 3) float32 arrays — discretised polylines
    topo_edges_occ        : list of TopoDS_Edge — the raw OCCT edge, parallel to above
    topo_edge_face_normals: list of (M, 3) float32 arrays — world normals of the
                            faces adjacent to each edge (M is 1 or 2 normally)
    """
    from OCP.TopExp import TopExp_Explorer, TopExp
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.GCPnts import GCPnts_TangentialDeflection
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    # Build edge→faces map, iterating it directly (avoids shape-identity
    # lookup issues with Contains() on oriented explorer edges).
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(
        shape.wrapped, TopAbs_EDGE, TopAbs_FACE, edge_face_map
    )

    # Build a dict keyed by rounded edge midpoint → list of face normals,
    # by walking the map directly (map keys are canonical/unoriented shapes).
    midpt_to_normals: dict[tuple, list] = {}
    for map_idx in range(1, edge_face_map.Extent() + 1):
        map_edge = TopoDS.Edge_s(edge_face_map.FindKey(map_idx))
        try:
            adp = BRepAdaptor_Curve(map_edge)
            mid_param = (adp.FirstParameter() + adp.LastParameter()) * 0.5
            mid_pt = adp.Value(mid_param)
            mkey = (round(mid_pt.X(), 3), round(mid_pt.Y(), 3), round(mid_pt.Z(), 3))
        except Exception:
            continue

        normals = []
        face_list = edge_face_map.FindFromIndex(map_idx)
        for face_shape in face_list:
            face = TopoDS.Face_s(face_shape)
            try:
                # Planar fast path → Plane(face).z_dir; falls back to a
                # surface-derivative normal sampled at the face's UV center
                # for non-planar faces.
                from build123d import Face as B3dFace, Plane as B3dPlane
                try:
                    pl = B3dPlane(B3dFace(face))
                    zd = pl.z_dir
                    normals.append([zd.X, zd.Y, zd.Z])
                except Exception:
                    from cad.face_ref import _occ_face_anchor_and_normal
                    _, n = _occ_face_anchor_and_normal(face)
                    if n is not None:
                        normals.append([float(n[0]), float(n[1]), float(n[2])])
            except Exception:
                pass
        midpt_to_normals[mkey] = normals

    edges        = []
    edges_occ    = []
    edge_normals = []
    seen         = set()

    explorer = TopExp_Explorer(shape.wrapped, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())

        try:
            adaptor = BRepAdaptor_Curve(edge)
            first   = adaptor.FirstParameter()
            last    = adaptor.LastParameter()

            # Adaptive: angular deflection 0.2 rad (~11°), chordal 0.05 mm
            discretiser = GCPnts_TangentialDeflection(adaptor, 0.2, 0.05)

            pts = []
            n = discretiser.NbPoints()
            if n >= 2:
                for i in range(1, n + 1):
                    p = adaptor.Value(discretiser.Parameter(i))
                    pts.append([p.X(), p.Y(), p.Z()])
            else:
                p0 = adaptor.Value(first)
                p1 = adaptor.Value(last)
                pts = [[p0.X(), p0.Y(), p0.Z()],
                       [p1.X(), p1.Y(), p1.Z()]]

            if len(pts) < 2:
                explorer.Next()
                continue

            arr = np.array(pts, dtype=np.float32)

            # Deduplicate by true geometric midpoint (mean of all points)
            mid = arr.mean(axis=0)
            key = (round(float(mid[0]), 3),
                   round(float(mid[1]), 3),
                   round(float(mid[2]), 3))
            if key in seen:
                explorer.Next()
                continue
            seen.add(key)
            edges.append(arr)
            edges_occ.append(edge)

            # Look up face normals via the pre-built midpoint dict
            mid_param = (first + last) * 0.5
            mid_pt    = adaptor.Value(mid_param)
            mkey = (round(mid_pt.X(), 3), round(mid_pt.Y(), 3), round(mid_pt.Z(), 3))
            normals = midpt_to_normals.get(mkey, [])
            edge_normals.append(
                np.array(normals, dtype=np.float32) if normals
                else np.zeros((0, 3), dtype=np.float32)
            )

        except Exception:
            pass

        explorer.Next()

    return edges, edges_occ, edge_normals


def edge_outward_normal(mesh, edge_idx: int) -> np.ndarray:
    """Return a unit outward direction for an edge — for placing a drag arrow.

    Sums the unit normals of the edge's adjacent faces. When the two adjacent
    normals are anti-parallel (so the naive mean cancels to zero), falls back
    to the first normal. Returns +Z if no normals are available.
    """
    efn = getattr(mesh, 'topo_edge_face_normals', None)
    if not efn or edge_idx >= len(efn):
        return np.array([0.0, 0.0, 1.0])
    adj = np.array(efn[edge_idx], dtype=float)
    if adj.ndim != 2 or adj.shape[0] == 0:
        return np.array([0.0, 0.0, 1.0])
    units = []
    for row in adj:
        ln = np.linalg.norm(row)
        if ln > 1e-10:
            units.append(row / ln)
    if not units:
        return np.array([0.0, 0.0, 1.0])
    summed = np.sum(units, axis=0)
    if np.linalg.norm(summed) < 1e-6:
        return units[0]
    return summed / np.linalg.norm(summed)
