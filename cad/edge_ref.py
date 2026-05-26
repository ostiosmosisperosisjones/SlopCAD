"""
cad/edge_ref.py

Stable edge references and abstract edge sources for parametric include replay.

EdgeRef          — geometry fingerprint that re-finds an edge after topology changes
EdgeSource       — abstract base: resolve → (world_pts, occ_edges)
BodyEdgeSource   — edge on a specific body (most common include case)
SketchEdgeSource — line entity from a previously committed sketch
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# EdgeRef — stable fingerprint
# ---------------------------------------------------------------------------

@dataclass
class EdgeRef:
    """
    Identifies an edge by midpoint, arc length, and tangent direction at
    the midpoint.  The tangent is canonicalised (first non-zero component
    is positive) so the same physical edge always produces the same ref
    regardless of OCCT orientation.

    Tolerances in find_in() are intentionally generous to survive boolean
    operations that may slightly perturb edge positions.

    face_sigs (optional) — fingerprints of the two faces that share this
    edge, used as a topology-aware fallback when geometric matching fails
    (e.g. when draft moves the edge past mid_tol). Each sig is a dict:
    {'category': 'plane'|'curved', 'centroid': (x,y,z), 'area': float}.
    """
    midpoint:  tuple    # (x, y, z) world mm
    length:    float    # arc length mm
    tangent:   tuple    # (tx, ty, tz) unit vector, canonicalised
    face_sigs: list = field(default_factory=list)   # optional adjacency hint

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_occ_edge(cls, occ_edge, parent_shape=None) -> "EdgeRef | None":
        """Build an EdgeRef from a raw TopoDS_Edge. Returns None on failure.

        If parent_shape (raw TopoDS_Shape) is supplied, the ref also records
        signatures for the two adjacent faces — used as a topology-aware
        fallback in find_in() when geometric matching can't survive an edit.
        """
        try:
            from OCP.BRepAdaptor import BRepAdaptor_Curve
            from OCP.GCPnts import GCPnts_AbscissaPoint

            adaptor = BRepAdaptor_Curve(occ_edge)
            u0   = adaptor.FirstParameter()
            u1   = adaptor.LastParameter()
            umid = (u0 + u1) * 0.5

            p = adaptor.Value(umid)
            midpoint = (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))

            length = GCPnts_AbscissaPoint.Length_s(adaptor, u0, u1, 1e-6)

            tv  = adaptor.DN(umid, 1)
            t   = np.array([tv.X(), tv.Y(), tv.Z()], dtype=np.float64)
            tn  = np.linalg.norm(t)
            if tn < 1e-10:
                return None
            t /= tn
            # Canonicalise: flip if first significant component is negative
            for c in t:
                if abs(c) > 1e-6:
                    if c < 0:
                        t = -t
                    break
            tangent = tuple(np.round(t, 8).tolist())

            face_sigs = []
            if parent_shape is not None:
                face_sigs = _adjacent_face_sigs(parent_shape, occ_edge)

            return cls(midpoint=midpoint,
                       length=round(float(length), 6),
                       tangent=tangent,
                       face_sigs=face_sigs)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_in(self, shape,
                mid_tol: float = 0.1,
                len_tol: float = 0.5,
                tan_tol: float = 0.02,
                ) -> tuple[int, object, list] | tuple[None, None, None]:
        """
        Find the matching edge in a build123d shape.

        Tries strict geometric matching first (cheap, exact for replays
        without topology change); falls back to face-adjacency matching
        via face_sigs when geometric matching misses (typical when an
        upstream edit perturbs the edge — e.g. draft, scaling).

        Returns (edge_index, TopoDS_Edge, world_pts) where world_pts is a
        list of [x, y, z] floats sampled along the edge.
        Returns (None, None, None) if no match is found.
        """
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GCPnts import GCPnts_AbscissaPoint

        ref_mid = np.array(self.midpoint)
        ref_tan = np.array(self.tangent)
        best_dist = float("inf")
        best = (None, None, None)

        for idx, edge in enumerate(shape.edges()):
            occ = edge.wrapped
            try:
                adaptor = BRepAdaptor_Curve(occ)
                u0   = adaptor.FirstParameter()
                u1   = adaptor.LastParameter()
                umid = (u0 + u1) * 0.5

                p   = adaptor.Value(umid)
                mid = np.array([p.X(), p.Y(), p.Z()])
                dist = float(np.linalg.norm(mid - ref_mid))
                if dist > mid_tol:
                    continue

                length = GCPnts_AbscissaPoint.Length_s(adaptor, u0, u1, 1e-6)
                if abs(length - self.length) > len_tol:
                    continue

                tv = adaptor.DN(umid, 1)
                t  = np.array([tv.X(), tv.Y(), tv.Z()])
                tn = np.linalg.norm(t)
                if tn > 1e-10:
                    t /= tn
                    for c in t:
                        if abs(c) > 1e-6:
                            if c < 0:
                                t = -t
                            break
                    if float(np.linalg.norm(t - ref_tan)) > tan_tol:
                        continue

                if dist < best_dist:
                    best_dist = dist
                    best = (idx, occ, _sample_edge(adaptor, u0, u1))

            except Exception:
                continue

        if best[0] is not None:
            return best

        # Geometric match missed — fall back to face-pair adjacency if we have
        # face signatures recorded. Survives upstream edits that move the edge
        # past mid_tol/len_tol but preserve the adjacent face roles.
        if len(self.face_sigs) >= 2:
            return _find_edge_via_face_sigs(shape, self.face_sigs[0], self.face_sigs[1])

        return best


def _sample_edge(adaptor, u0: float, u1: float, n: int = 32) -> list:
    """Sample n+1 evenly-spaced world points along an edge adaptor."""
    pts = []
    for i in range(n + 1):
        u = u0 + (u1 - u0) * i / n
        p = adaptor.Value(u)
        pts.append([p.X(), p.Y(), p.Z()])
    return pts


# ---------------------------------------------------------------------------
# Adjacency-based edge lookup
# ---------------------------------------------------------------------------

# Score threshold above which an adjacency match is considered too weak
# to trust. Empirically a single-face match contributes ~10–50 depending
# on how much the upstream edit deformed it; 200 is generous but excludes
# total mismatches.
_ADJACENCY_SCORE_CUTOFF = 200.0


def _surface_category(occ_face) -> str:
    """Coarse surface family — 'plane' or 'curved'. Stable across draft,
    which turns a cylinder into a cone but keeps planes planar."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    return 'plane' if BRepAdaptor_Surface(occ_face).GetType() == GeomAbs_Plane else 'curved'


def _face_signature(occ_face) -> dict:
    from cad.face_ref import _occ_face_props
    centroid, area = _occ_face_props(occ_face)
    return {
        'category': _surface_category(occ_face),
        'centroid': [float(c) for c in centroid],
        'area':     float(area),
    }


def _adjacent_face_sigs(parent_shape_occ, occ_edge) -> list:
    """Return signatures of the (≤2) faces adjacent to occ_edge in parent_shape_occ."""
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(parent_shape_occ, TopAbs_EDGE, TopAbs_FACE, m)
    for i in range(1, m.Extent() + 1):
        if m.FindKey(i).IsSame(occ_edge):
            return [_face_signature(TopoDS.Face_s(f))
                    for f in m.FindFromIndex(i)]
    return []


def _score_face_match(sig: dict, occ_face) -> float:
    """Lower is better; +inf if surface category mismatches."""
    if _surface_category(occ_face) != sig['category']:
        return float('inf')
    from cad.face_ref import _occ_face_props
    centroid, area = _occ_face_props(occ_face)
    cdist = float(np.linalg.norm(np.array(centroid) - np.array(sig['centroid'])))
    adiff = abs(area - sig['area']) / max(sig['area'], 1.0)
    return cdist + 50.0 * adiff


def _find_edge_via_face_sigs(shape, sig_a: dict, sig_b: dict):
    """
    Locate the edge in shape (build123d) whose adjacent faces best match
    sig_a and sig_b. Returns (idx, TopoDS_Edge, world_pts) like find_in().
    """
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    shape_occ = shape.wrapped if hasattr(shape, 'wrapped') else shape
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape_occ, TopAbs_EDGE, TopAbs_FACE, m)

    best_score = float('inf')
    best_edge  = None
    for i in range(1, m.Extent() + 1):
        edge = TopoDS.Edge_s(m.FindKey(i))
        fl   = list(m.FindFromIndex(i))
        if len(fl) < 2:
            continue
        f0 = TopoDS.Face_s(fl[0])
        f1 = TopoDS.Face_s(fl[1])
        s1 = _score_face_match(sig_a, f0) + _score_face_match(sig_b, f1)
        s2 = _score_face_match(sig_a, f1) + _score_face_match(sig_b, f0)
        s  = min(s1, s2)
        if s < best_score:
            best_score = s
            best_edge  = edge

    if best_edge is None or best_score > _ADJACENCY_SCORE_CUTOFF:
        return (None, None, None)

    # Find the build123d edge wrapping this TopoDS_Edge so callers get a
    # valid index for downstream code that uses shape.edges() ordering.
    idx = None
    for i, e in enumerate(shape.edges()):
        if e.wrapped.IsSame(best_edge):
            idx = i
            break

    adaptor = BRepAdaptor_Curve(best_edge)
    u0 = adaptor.FirstParameter(); u1 = adaptor.LastParameter()
    world_pts = _sample_edge(adaptor, u0, u1)
    return (idx, best_edge, world_pts)


# ---------------------------------------------------------------------------
# EdgeSource — abstract
# ---------------------------------------------------------------------------

class EdgeSource(ABC):
    """
    Abstract reference to a geometric edge that can be re-resolved at any
    point in the replay timeline.

    resolve(history, before_index) → (world_pts, occ_edges)
      world_pts  : list of [x, y, z]  — used for UV projection
      occ_edges  : list of TopoDS_Edge | None  — for exact face construction
    Raises RuntimeError on failure so the caller can surface the error.
    """

    @abstractmethod
    def resolve(self, history, before_index: int) -> tuple[list, list | None]:
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        ...


# ---------------------------------------------------------------------------
# BodyEdgeSource
# ---------------------------------------------------------------------------

class BodyEdgeSource(EdgeSource):
    """An edge on a specific body, re-found each replay via EdgeRef matching."""

    def __init__(self, body_id: str, edge_ref: EdgeRef):
        self.body_id  = body_id
        self.edge_ref = edge_ref

    def resolve(self, history, before_index: int) -> tuple[list, list | None]:
        shape = history._shape_for_body_at(self.body_id, before_index)
        if shape is None:
            raise RuntimeError(
                f"BodyEdgeSource: no shape for body '{self.body_id}' "
                f"before history index {before_index}")
        _, occ_edge, world_pts = self.edge_ref.find_in(shape)
        if occ_edge is None:
            raise RuntimeError(
                f"BodyEdgeSource: could not relocate edge in body '{self.body_id}'")
        return world_pts, [occ_edge]

    def to_dict(self) -> dict:
        return {"type": "body_edge", "body_id": self.body_id}


# ---------------------------------------------------------------------------
# SketchEdgeSource
# ---------------------------------------------------------------------------

class SketchEdgeSource(EdgeSource):
    """
    A LineEntity (or ArcEntity) from a previously committed sketch entry.

    Identified by the source sketch's stable entry_id UUID, so reorder/
    delete/insert in the history list never breaks the reference.

    On resolve(), converts the UV-space entity back to world space using
    the source SketchEntry's (possibly already-updated) plane cache, so it
    automatically inherits upstream sketch plane changes.
    """

    def __init__(self, source_entry_id: str, entity_idx: int):
        self.source_entry_id = source_entry_id
        self.entity_idx      = entity_idx

    def resolve(self, history, before_index: int) -> tuple[list, list | None]:
        from cad.sketch import LineEntity

        sketch_idx = history.id_to_index(self.source_entry_id)
        if sketch_idx is None:
            raise RuntimeError(
                f"SketchEdgeSource: source entry '{self.source_entry_id}' not found")
        entry = history.entries[sketch_idx]
        se = entry.params.get("sketch_entry")
        if se is None:
            raise RuntimeError(
                f"SketchEdgeSource: entry '{self.source_entry_id}' has no sketch_entry")
        if self.entity_idx >= len(se.entities):
            raise RuntimeError(
                f"SketchEdgeSource: entity_idx {self.entity_idx} out of range")
        ent = se.entities[self.entity_idx]

        def _uv_to_world(uv):
            return (se.plane_origin
                    + float(uv[0]) * se.plane_x_axis
                    + float(uv[1]) * se.plane_y_axis)

        from cad.sketch import ArcEntity
        if isinstance(ent, LineEntity):
            p0 = _uv_to_world(ent.p0)
            p1 = _uv_to_world(ent.p1)
            return [p0.tolist(), p1.tolist()], None
        elif isinstance(ent, ArcEntity):
            import math
            from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
            from OCP.Geom import Geom_Circle
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
            cx, cy = float(ent.center[0]), float(ent.center[1])
            c3d = _uv_to_world(ent.center)
            ax2 = gp_Ax2(
                gp_Pnt(*c3d.tolist()),
                gp_Dir(float(se.plane_normal[0]),
                       float(se.plane_normal[1]),
                       float(se.plane_normal[2])),
                gp_Dir(float(se.plane_x_axis[0]),
                       float(se.plane_x_axis[1]),
                       float(se.plane_x_axis[2])),
            )
            geom_circ = Geom_Circle(ax2, ent.radius)
            span = ent.end_angle - ent.start_angle
            if abs(span - 2 * math.pi) < 1e-9 or abs(span) < 1e-9:
                occ_edge = BRepBuilderAPI_MakeEdge(geom_circ).Edge()
            else:
                occ_edge = BRepBuilderAPI_MakeEdge(
                    geom_circ, ent.start_angle, ent.end_angle).Edge()
            p0 = _uv_to_world(ent.p0)
            p1 = _uv_to_world(ent.p1)
            return [p0.tolist(), p1.tolist()], [occ_edge]
        else:
            raise RuntimeError(
                f"SketchEdgeSource: entity {self.entity_idx} in entry "
                f"'{self.source_entry_id}' has unsupported type {type(ent).__name__}")

    def to_dict(self) -> dict:
        return {
            "type":            "sketch_edge",
            "source_entry_id": self.source_entry_id,
            "entity_idx":      self.entity_idx,
        }
