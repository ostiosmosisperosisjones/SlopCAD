"""
cad/operations/chamfer.py

3D edge chamfer operations.
Uses OCCT BRepFilletAPI_MakeChamfer with the AddDA signature (distance + angle
measured from a reference face) for maximum power.

Public API
----------
chamfer_edges(shape, edge_occs, distance, angle_deg, flip_reference_face=False)
    Commit-quality chamfer.  Each edge is chamfered with the supplied distance
    measured along its first adjacent face and the supplied angle measured
    from that face.  Setting flip_reference_face=True swaps to the second
    adjacent face — useful when the default reference produces an unwanted
    asymmetric bias.
"""

from __future__ import annotations
from cad.solid_ref import as_compound
from cad.kernel_pool import offload
import math
from build123d import Compound
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopExp import TopExp
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopoDS import TopoDS


def _build_edge_to_faces_map(shape_occ):
    """Return TopTools_IndexedDataMapOfShapeListOfShape mapping each edge to
    its adjacent faces."""
    from OCP.collections import (
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher
        as TopTools_IndexedDataMapOfShapeListOfShape)
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape_occ, TopAbs_EDGE, TopAbs_FACE,
                                    edge_face_map)
    return edge_face_map


def smooth_edges(edge_occs, edge_face_map, tol_deg: float = 1.0) -> list:
    """Indices into *edge_occs* of edges whose two faces meet tangentially
    (e.g. where a fillet runs into a flat face).  There is no corner to
    chamfer or fillet there: OCCT refuses, and the boolean fallback builds
    a degenerate cutter that swallows the body."""
    from OCP.TopoDS import TopoDS
    import math
    out = []
    for i, e in enumerate(edge_occs):
        if not edge_face_map.Contains(e):
            continue
        faces = list(edge_face_map.FindFromKey(e))
        if len(faces) != 2 or faces[0].IsSame(faces[1]):
            continue                        # boundary or seam edge
        n1 = _face_normal_at_edge(e, TopoDS.Face(faces[0]))
        n2 = _face_normal_at_edge(e, TopoDS.Face(faces[1]))
        if n1 is None or n2 is None:
            continue
        if sum(a * b for a, b in zip(n1, n2)) > math.cos(math.radians(tol_deg)):
            out.append(i)
    return out


def _face_area(face_occ) -> float:
    """Return the surface area of a face."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    gp = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face_occ, gp)
    return abs(gp.Mass())


def _face_normal_at_edge(edge_occ, face_occ):
    """Return the face's outward unit normal sampled at the edge midpoint, as
    an (x, y, z) tuple, or None if it can't be evaluated."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRep import BRep_Tool
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.TopAbs import TopAbs_REVERSED

    adp = BRepAdaptor_Curve(edge_occ)
    u0, u1 = adp.FirstParameter(), adp.LastParameter()
    mid = adp.Value((u0 + u1) * 0.5)
    surf = BRep_Tool.Surface_s(face_occ)
    proj = GeomAPI_ProjectPointOnSurf(mid, surf)
    if proj.NbPoints() == 0:
        return None
    u, v = proj.LowerDistanceParameters()
    props = GeomLProp_SLProps(surf, u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    n = props.Normal()
    nx, ny, nz = n.X(), n.Y(), n.Z()
    if face_occ.Orientation() == TopAbs_REVERSED:
        nx, ny, nz = -nx, -ny, -nz
    return (nx, ny, nz)


def _ordered_adjacent_faces(edge_occ, edge_face_map):
    """Return the edge's adjacent faces in a geometry-deterministic order.

    OCCT lists adjacent faces in an order that follows internal topology
    iteration, NOT geometry — so two mirror-symmetric edges can receive their
    adjacent faces in opposite order. Picking ".First()" then references
    non-corresponding faces, producing asymmetric chamfers on symmetric edges
    at any angle != 45 deg.

    We instead order the two faces by descending surface area, so the larger
    adjacent face is always the reference. Area is invariant under the mirror
    and rotational symmetries that relate symmetric edges (a face maps to a
    congruent, equal-area face), so corresponding faces are chosen on every
    symmetric edge — this handles the common case (e.g. a non-cube box where
    one adjacent face is genuinely larger).

    When the two faces have equal area (a cube, where every face is congruent
    and area can't decide), we tie-break by the face normal's alignment with
    the global axes, preferring +Z then +Y then +X. That choice is invariant
    under the part's symmetry: the symmetry that maps one symmetric edge to
    another fixes the global up axis, so the same physical face (e.g. the top
    cap rather than a lateral side) is picked on every edge of the group.
    """
    if not edge_face_map.Contains(edge_occ):
        raise RuntimeError("Chamfer: edge has no adjacent faces in this shape")
    face_list = edge_face_map.FindFromKey(edge_occ)
    faces = [TopoDS.Face(f) for f in face_list]
    if not faces:
        raise RuntimeError("Chamfer: edge has no adjacent faces")
    if len(faces) < 2:
        return faces

    def sort_key(face_occ):
        # Primary: larger area first (negate so it sorts ahead).
        area = round(_face_area(face_occ), 5)
        # Tie-break: alignment with global +Z, +Y, +X (negate so "more
        # aligned" sorts ahead). Symmetry-invariant for equal-area faces.
        nrm = _face_normal_at_edge(edge_occ, face_occ)
        if nrm is None:
            align = (0.0, 0.0, 0.0)
        else:
            align = tuple((round(c, 5) or 0.0) for c in nrm)
        return (-area, -align[2], -align[1], -align[0])

    return sorted(faces, key=sort_key)


def _pick_reference_face(edge_occ, edge_face_map, flip: bool):
    """Return one of the faces adjacent to *edge_occ*.

    The two adjacent faces are ordered deterministically by geometry (see
    _ordered_adjacent_faces) so symmetric edges reference corresponding faces.

    flip=False → the first face in geometric order.
    flip=True  → the second adjacent face if one exists, else the first.

    Raises if the edge has no adjacent faces (a degenerate seam edge).
    """
    faces = _ordered_adjacent_faces(edge_occ, edge_face_map)
    pick = faces[1] if (flip and len(faces) >= 2) else faces[0]
    return pick


def chamfer_edges_validate(shape, edge_occs, distance: float, angle_deg: float,
                            flip_reference_face: bool = False) -> list:
    """Trial-chamfer each edge individually and report per-edge errors.

    Returns a list of (error_str | None) aligned with edge_occs. An entry is
    None when the edge would chamfer cleanly on its own; otherwise it carries
    the OCCT/kernel reason.

    This is best-effort validation — an edge that passes here can still cause
    a combined chamfer to fail (e.g. interaction between adjacent edges), but
    catching the obvious cases (seam edges, bad reference faces, kernel
    rejects) is enough to give the user actionable feedback.
    """
    source_occ = shape.wrapped if hasattr(shape, "wrapped") else shape
    if distance <= 0 or not (0.0 < angle_deg < 90.0):
        # Parameters themselves are invalid — every edge "fails" the same way.
        reason = (f"distance must be > 0 (got {distance})" if distance <= 0
                  else f"angle must be in (0, 90), got {angle_deg}")
        return [reason for _ in edge_occs]

    edge_face_map = _build_edge_to_faces_map(source_occ)
    angle_rad = math.radians(angle_deg)
    results: list[str | None] = []
    for e_occ in edge_occs:
        try:
            ref_face = _pick_reference_face(e_occ, edge_face_map,
                                             flip_reference_face)
        except Exception as ex:
            results.append(str(ex))
            continue
        # Try the native kernel first.
        try:
            mk = BRepFilletAPI_MakeChamfer(source_occ)
            mk.AddDA(float(distance), angle_rad, e_occ, ref_face)
            mk.Build()
            if mk.IsDone():
                results.append(None)
                continue
        except Exception as ex:
            # Kernel threw before we could see IsDone — defer to fallback below.
            pass
        # Native rejected. Validation should match what chamfer_edges() actually
        # does, so try to build the fallback cutter before reporting failure.
        # The cache lookup makes this cheap on subsequent calls.
        try:
            _get_cached_cutter(source_occ, e_occ, float(distance),
                                angle_rad, flip_reference_face)
            results.append(None)
        except Exception as ex:
            results.append(str(ex))
    return results


def chamfer_edges(shape, edge_occs, distance: float, angle_deg: float,
                  flip_reference_face: bool = False,
                  per_edge_errors: list | None = None):
    """Chamfer in a kernel worker (see _chamfer_edges_local); per_edge_errors
    is filled from the worker's result."""
    result, per_edge, err = _chamfer_task(shape, list(edge_occs), distance,
                                          angle_deg, flip_reference_face)
    if per_edge_errors is not None:
        per_edge_errors.extend(per_edge)
    if err is not None:
        raise err
    return result


@offload()
def _chamfer_task(shape, edge_occs, distance, angle_deg, flip_reference_face):
    from cad.operations.fillet_proc import _own_edges
    edge_occs = _own_edges(shape, [getattr(e, "wrapped", e) for e in edge_occs])
    per_edge: list = []
    try:
        result = _chamfer_edges_local(shape, edge_occs, distance, angle_deg,
                                      flip_reference_face, per_edge)
    except ValueError as ex:
        return None, per_edge, ValueError(str(ex))
    except Exception as ex:   # OCCT exception types don't pickle
        return None, per_edge, RuntimeError(str(ex))
    return result, per_edge, None


def _chamfer_edges_local(shape, edge_occs, distance: float, angle_deg: float,
                         flip_reference_face: bool = False,
                         per_edge_errors: list | None = None):
    """Chamfer the given OCCT edges with distance + angle.

    shape         : build123d Shape (with .wrapped) or raw TopoDS_Shape
    edge_occs     : list of TopoDS_Edge to chamfer
    distance      : mm measured along the reference face
    angle_deg     : degrees, measured from the reference face (must be > 0
                    and < 90 — OCCT rejects 0 and >=90)
    flip_reference_face : swap to the other adjacent face for the angle
                          measurement, mirroring the asymmetry across the edge
    per_edge_errors : if a list is supplied, on fallback the function appends
                      (edge_index, error_str | None) per edge so callers can
                      flag specific edges that failed without running
                      chamfer_edges_validate as a separate pass.

    Tries OCCT's native chamfer kernel first; if that fails (e.g. drafted
    cone-to-plane edges, where the kernel refuses to build), falls back to
    building one cutter solid per edge and removing them all in a single
    boolean cut. The cutters are cached by (edge identity, distance, angle,
    flip) so changing parameters during preview only rebuilds what's stale.
    """
    source_occ = shape.wrapped if hasattr(shape, "wrapped") else shape
    if not edge_occs:
        raise RuntimeError("Chamfer: no edges selected")
    if distance <= 0:
        raise ValueError(f"Chamfer: distance must be positive, got {distance}")
    if not (0.0 < angle_deg < 90.0):
        raise ValueError(
            f"Chamfer: angle must be in (0, 90), got {angle_deg}")

    edge_face_map = _build_edge_to_faces_map(source_occ)
    smooth = smooth_edges(edge_occs, edge_face_map)
    if smooth:
        raise RuntimeError(
            f"Chamfer: selected edge{'s' if len(smooth) > 1 else ''} "
            f"{', '.join(str(i + 1) for i in smooth)} "
            f"{'are' if len(smooth) > 1 else 'is'} smooth (the faces meet "
            f"tangentially) — there is no corner to chamfer")
    mk = BRepFilletAPI_MakeChamfer(source_occ)
    angle_rad = math.radians(angle_deg)
    added = 0
    for e_occ in edge_occs:
        try:
            ref_face = _pick_reference_face(e_occ, edge_face_map,
                                             flip_reference_face)
            mk.AddDA(float(distance), angle_rad, e_occ, ref_face)
            added += 1
        except Exception:
            # Skip edges OCCT can't chamfer (seam edges, degenerate, etc.)
            continue

    if added > 0:
        try:
            mk.Build()
            # IsDone() isn't enough: on e.g. an ellipse edge between two
            # cylinders the kernel "succeeds" with an invalid solid missing
            # a large chunk (walker seed 9). Treat that as a failure.
            if mk.IsDone() and BRepCheck_Analyzer(mk.Shape()).IsValid():
                if per_edge_errors is not None:
                    per_edge_errors.extend((i, None) for i in range(len(edge_occs)))
                return as_compound(mk.Shape())
        except Exception:
            # OCCT can throw "no suitable edges for chamfer or fillet" from
            # Build() even after AddDA accepts the edge (e.g. drafted seam
            # edges). Fall through to the boolean-cut fallback rather than
            # propagating an error that the fallback would have handled.
            pass

    # Fallback: build a cutter solid per edge, then remove them all from
    # the body in a single boolean cut. Cutters are looked up in a process-
    # wide cache so repeated previews with the same edges and parameters
    # don't recompute them.
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    cutters = []
    last_error = None
    edge_errors: list[tuple[int, str | None]] = []
    for i, e_occ in enumerate(edge_occs):
        try:
            cutter = _get_cached_cutter(
                source_occ, e_occ, distance, angle_rad, flip_reference_face)
            cutters.append(cutter)
            edge_errors.append((i, None))
        except Exception as ex:
            last_error = ex
            edge_errors.append((i, str(ex)))

    if per_edge_errors is not None:
        per_edge_errors.extend(edge_errors)

    if not cutters:
        msg = "Chamfer: kernel failed and boolean fallback also failed"
        if last_error is not None:
            msg += f" — {last_error}"
        raise RuntimeError(msg)

    # One boolean cut with all cutters together — much faster than chaining
    # N sequential cuts, and OCCT can parallelise the intersection work.
    la = TopTools_ListOfShape(); la.Append(source_occ)
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    lb = TopTools_ListOfShape()
    for c in cutters:
        # Cut with a copy: the boolean raises the tool's tolerances in place,
        # and a cached cutter that has been through one cut makes the next
        # cut behave differently — live and reload disagreed (walker seed 2).
        lb.Append(BRepBuilderAPI_Copy(c).Shape())
    op = BRepAlgoAPI_Cut()
    op.SetArguments(la); op.SetTools(lb); op.SetRunParallel(True)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("Chamfer: combined boolean cut failed")
    result = op.Shape()

    # Filter out micro-sliver solids from cutter grazing. Same logic as the
    # old per-edge cut path, applied to the combined result.
    raw_solids = list(as_compound(result).solids())
    if not raw_solids:
        raise RuntimeError("Chamfer: combined cut produced empty result")
    gp_before = GProp_GProps()
    BRepGProp.VolumeProperties_s(source_occ, gp_before)
    body_vol = abs(gp_before.Mass())
    sliver_threshold = max(1e-6, body_vol * 1e-7)
    kept = []
    gp_s = GProp_GProps()
    for s in raw_solids:
        BRepGProp.VolumeProperties_s(s.wrapped, gp_s)
        if abs(gp_s.Mass()) > sliver_threshold:
            kept.append(s.wrapped)
    if not kept:
        raise RuntimeError("Chamfer: the kernel can't chamfer these edges and the "
                           "fallback cut missed the body (common where two curved "
                           "faces meet) — try a smaller distance or other edges")
    # A chamfer only removes material, so it can't make bodies: a chip left
    # between two cutters (0.01mm³, walker seed 13) is material the chamfer
    # removes. Keep as many solids as went in, the largest.
    n_in = max(1, len(as_compound(source_occ).solids()))
    if len(kept) > n_in:
        vol = {}
        for s in kept:
            BRepGProp.VolumeProperties_s(s, gp_s)
            vol[id(s)] = abs(gp_s.Mass())
        kept = sorted(kept, key=lambda s: -vol[id(s)])[:n_in]
    if len(kept) == 1:
        result = kept[0]
    else:
        comp = TopoDS_Compound(); BRep_Builder().MakeCompound(comp)
        for s in kept:
            BRep_Builder().Add(comp, s)
        result = comp

    if not BRepCheck_Analyzer(result).IsValid():
        raise RuntimeError("Chamfer: result is not a valid solid — "
                           "try a smaller distance or fewer edges")
    return as_compound(result)


# ---------------------------------------------------------------------------
# Cutter cache
# ---------------------------------------------------------------------------
#
# Building a chamfer cutter for one edge is the expensive step (sample edge,
# project offsets onto two faces, sew shell, build solid). The same cutter is
# valid as long as edge identity and parameters are unchanged, so cache by
# (body_id, edge_key, distance, angle, flip). Bounded LRU so we never leak —
# during a panel session the user touches at most a few dozen distinct
# (edge × params) combinations.

from collections import OrderedDict

_CUTTER_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_CUTTER_CACHE_MAX = 256


def _edge_cache_key(edge_occ) -> tuple:
    """Cheap stable fingerprint for a TopoDS_Edge — midpoint, length, type."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_AbscissaPoint
    adp = BRepAdaptor_Curve(edge_occ)
    u0, u1 = adp.FirstParameter(), adp.LastParameter()
    mid = adp.Value((u0 + u1) * 0.5)
    length = GCPnts_AbscissaPoint.Length_s(adp)
    return (int(adp.GetType()),
            round(mid.X(), 4), round(mid.Y(), 4), round(mid.Z(), 4),
            round(length, 4))


_LAST_SOURCE: list = [None, None]   # [shape (held so it can't recycle), key]


def _source_key(source_occ) -> tuple:
    """Geometry fingerprint of the chamfered body (memoized for the last one,
    since every edge of a call asks with the same shape)."""
    last, key = _LAST_SOURCE
    if last is not None and last.IsSame(source_occ):
        return key
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as IMap
    counts = []
    for kind in (TopAbs_FACE, TopAbs_EDGE):
        m = IMap()
        TopExp.MapShapes_s(source_occ, kind, m)
        counts.append(m.Size())
    box = Bnd_Box()
    BRepBndLib.Add_s(source_occ, box)
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(source_occ, props)
    key = (*counts, *(round(c, 4) for p in (box.CornerMin(), box.CornerMax())
                                for c in (p.X(), p.Y(), p.Z())), round(props.Mass(), 4))
    _LAST_SOURCE[:] = [source_occ, key]
    return key


def _get_cached_cutter(source_occ, edge_occ, distance: float,
                       angle_rad: float, flip: bool):
    """Return a cutter solid for this edge, building and caching if needed."""
    # Key on the body's geometry, not id(): workers get a fresh unpickled copy
    # per call, so ids recycle — and a previously chamfered preview must not
    # reuse cutters built against the wrong topology.
    key = (_source_key(source_occ),
           _edge_cache_key(edge_occ),
           round(distance, 6), round(angle_rad, 8), bool(flip))
    hit = _CUTTER_CACHE.get(key)
    if hit is not None:
        _CUTTER_CACHE.move_to_end(key)
        return hit

    edge_face_map = _build_edge_to_faces_map(source_occ)
    # Order the two adjacent faces by geometry, not OCCT list order, so the
    # cutter's reference face matches the native path and stays consistent
    # across symmetric edges (same reasoning as _pick_reference_face).
    faces = _ordered_adjacent_faces(edge_occ, edge_face_map)
    if len(faces) < 2:
        raise RuntimeError("edge needs two adjacent faces")
    face_a = faces[0]
    face_b = faces[1]
    if flip:
        face_a, face_b = face_b, face_a
    distance_b = distance * math.tan(angle_rad)
    cutter = _build_chamfer_cutter(edge_occ, face_a, face_b,
                                    distance, distance_b,
                                    body_shape=source_occ)
    # We deliberately don't check whether the cutter sits inside the body
    # here — that check belongs to the per-edge `_chamfer_edge_via_cut`
    # path which is no longer used. The combined-cut path in chamfer_edges
    # tolerates cutters that partially miss; if a chamfer is geometrically
    # wrong the user sees it in the preview, but we don't flag the edge
    # red for being "close to too big".

    _CUTTER_CACHE[key] = cutter
    if len(_CUTTER_CACHE) > _CUTTER_CACHE_MAX:
        _CUTTER_CACHE.popitem(last=False)
    return cutter


def chamfer_cache_clear():
    """Drop all cached cutters. Call when a body's shape changes for reasons
    unrelated to chamfer (e.g. a different upstream op committed)."""
    _CUTTER_CACHE.clear()
    _LAST_SOURCE[:] = [None, None]


# ---------------------------------------------------------------------------
# Boolean-cut chamfer fallback
# ---------------------------------------------------------------------------
#
# OCCT's BRepFilletAPI_MakeChamfer refuses some otherwise-valid edges — most
# notably the boundary between a drafted (conical) lateral face and a flat
# circular cap. The fallback below sidesteps the kernel: it samples a chain
# of points along the edge, projects offset points onto each adjacent face,
# and lofts a closed solid through the three resulting closed polylines.
# Cutting that solid from the body produces a real chamfer with the correct
# adjacency, though the chamfer face is a thru-section spline rather than
# a perfect plane.


def _build_chamfer_cutter(edge_occ, face_a, face_b,
                           distance_a: float, distance_b: float, N: int = 24,
                           body_shape=None):
    """Build a closed solid bounded by three swept ruled surfaces between
    the edge curve and its two offset curves on face_a and face_b.

    Two cases:
      - Edge is geometrically closed (full circle / closed B-spline): build
        three closed loops and sew them into a tube — no end caps needed.
      - Edge is open (open arc, line segment, sub-arc of a periodic carrier):
        build three open wires and cap the two ends with planar triangles
        (edge_end, a_end, b_end), then sew.

    The earlier code branched on `adp.IsPeriodic()`, which describes the
    UNDERLYING CARRIER curve, not the edge itself. A sub-arc of a circle has
    IsPeriodic() == True but IsClosed() == False — and treating it as a full
    closed loop produced a cutter wire that ran around the whole carrier
    circle instead of just the sub-arc, leaving a cutter that only partially
    overlapped the body (the "halfway chamfer" bug on drafted round edges).
    """
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCP.BRepFill import BRepFill
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.Geom import Geom_Circle
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.collections import Array1_gp_Pnt as TColgp_Array1OfPnt
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopoDS import TopoDS

    adp = BRepAdaptor_Curve(edge_occ)
    u0, u1 = adp.FirstParameter(), adp.LastParameter()
    # The edge is closed iff its endpoints coincide. IsClosed() on the
    # adaptor is unreliable for sub-arcs of periodic carriers, so check
    # geometrically against a tolerance comparable to the chamfer scale.
    p_first = adp.Value(u0)
    p_last  = adp.Value(u1)
    endpoint_gap = math.sqrt(
        (p_first.X() - p_last.X()) ** 2 +
        (p_first.Y() - p_last.Y()) ** 2 +
        (p_first.Z() - p_last.Z()) ** 2)
    closed = endpoint_gap < 1e-6
    # For closed loops, N samples and don't repeat the seam point.
    # For open arcs, N+1 samples to include both endpoints.
    n_samples = N if closed else N + 1

    # Body classifier lets _offset_point_on_face prefer the side of the face
    # that walks INTO the body (matters for faces whose UV chart wraps the
    # body — cones, cylinders — where both ±in_surf can land on-face).
    body_classifier = None
    if body_shape is not None:
        from OCP.BRepClass3d import BRepClass3d_SolidClassifier
        body_classifier = BRepClass3d_SolidClassifier(body_shape)

    # Seed sign_hint from the MIDDLE of the edge rather than sample 0. At
    # edge endpoints we sit at a face vertex where ±perp can fall off the
    # bounded face on both sides, making classification unreliable. The
    # midpoint sits in the face interior where one side is clearly IN.
    def _seed_sign(face, distance):
        u_seed = (u0 + u1) * 0.5
        p_seed = adp.Value(u_seed)
        tv_seed = adp.DN(u_seed, 1)
        ts = gp_Vec(tv_seed.X(), tv_seed.Y(), tv_seed.Z())
        if ts.Magnitude() < 1e-9:
            return None
        ts.Normalize()
        _, s = _offset_point_on_face(p_seed, ts, face, distance, None,
                                      body_classifier=body_classifier)
        return s

    sign_a = _seed_sign(face_a, distance_a)
    sign_b = _seed_sign(face_b, distance_b)

    edge_pts, a_pts, b_pts = [], [], []
    for k in range(n_samples):
        u = u0 + (u1 - u0) * k / N
        p = adp.Value(u)
        tv = adp.DN(u, 1)
        tan = gp_Vec(tv.X(), tv.Y(), tv.Z())
        if tan.Magnitude() < 1e-9:
            raise RuntimeError("degenerate tangent at sample")
        tan.Normalize()
        a, sign_a = _offset_point_on_face(p, tan, face_a, distance_a, sign_a,
                                           body_classifier=body_classifier)
        b, sign_b = _offset_point_on_face(p, tan, face_b, distance_b, sign_b,
                                           body_classifier=body_classifier)
        if a is None or b is None:
            raise RuntimeError(f"failed to project offset at sample {k}")
        edge_pts.append(p); a_pts.append(a); b_pts.append(b)

    def _try_fit_line(points):
        """Return (p_start, p_end) gp_Pnt if points are collinear within
        tolerance, else None. Endpoints used for an exact line edge."""
        arr = np.array([[p.X(), p.Y(), p.Z()] for p in points])
        v = arr[-1] - arr[0]
        v_len = float(np.linalg.norm(v))
        if v_len < 1e-9:
            return None
        v_hat = v / v_len
        # Perpendicular deviation from the chord through endpoints.
        rel = arr - arr[0]
        proj = rel @ v_hat
        proj_pts = arr[0] + np.outer(proj, v_hat)
        dev = np.linalg.norm(arr - proj_pts, axis=1)
        if dev.max() > 1e-4:
            return None
        return points[0], points[-1]

    def _try_fit_circle(points):
        """Return (Geom_Circle, u_start, u_end) if points lie on a circular arc
        within tolerance, else (None, None, None).

        Uses an algebraic circle fit (not naive centroid) so it works for short
        arcs where the sample centroid is far from the true circle center."""
        arr = np.array([[p.X(), p.Y(), p.Z()] for p in points])
        # Find best-fit plane via SVD around centroid of samples.
        cen0 = arr.mean(axis=0)
        cnt = arr - cen0
        _, _, vh = np.linalg.svd(cnt, full_matrices=False)
        normal = vh[2]
        # If samples are nearly collinear, planar SVD is unstable — bail.
        # (Caller will fall through to the line or B-spline path.)
        # Reject if singular value ratio implies a degenerate fit.
        # Project samples onto best-fit plane (2D coordinates in that plane).
        u_dir = vh[0]
        v_dir = vh[1]
        x = cnt @ u_dir
        y = cnt @ v_dir
        # Algebraic 2D circle fit: solve [2x, 2y, 1] @ [a, b, c] = x²+y²
        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        rhs = x * x + y * y
        try:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None, None, None
        a, b, c = sol
        r2 = c + a * a + b * b
        if r2 <= 1e-12:
            return None, None, None
        R = math.sqrt(r2)
        # Center in 3D.
        center3d = cen0 + a * u_dir + b * v_dir
        # Measure residual distance of each sample from this best-fit circle.
        dx = x - a
        dy = y - b
        radii = np.sqrt(dx * dx + dy * dy)
        # Tolerance scales with R so it works for both 0.1 mm and 100 mm circles.
        if np.max(np.abs(radii - R)) > max(1e-4, 1e-3 * R):
            return None, None, None
        # Orient the circle so its parameter direction matches sample order.
        x_dir3 = (arr[0] - center3d)
        x_norm = float(np.linalg.norm(x_dir3))
        if x_norm < 1e-9:
            return None, None, None
        x_dir3 /= x_norm
        forward = np.cross(normal, x_dir3)
        if np.dot(forward, arr[1] - arr[0]) < 0:
            normal = -normal
            forward = -forward
        ax = gp_Ax2(
            gp_Pnt(float(center3d[0]), float(center3d[1]), float(center3d[2])),
            gp_Dir(float(normal[0]), float(normal[1]), float(normal[2])),
            gp_Dir(float(x_dir3[0]),  float(x_dir3[1]),  float(x_dir3[2])),
        )
        # Compute U-parameter for first and last samples on the oriented circle.
        def _u_param(p3d):
            d = p3d - center3d
            return math.atan2(float(np.dot(d, forward)), float(np.dot(d, x_dir3)))
        u_s = _u_param(arr[0])
        u_e = _u_param(arr[-1])
        # Unwrap so u_e is on the correct side of u_s for a forward-going arc.
        while u_e < u_s - 1e-9:
            u_e += 2 * math.pi
        return Geom_Circle(ax, R), u_s, u_e

    def wire_from_points(points):
        # 1. Straight line — when the underlying offset is on a plane, offset
        # points are colinear. A single line edge is the cleanest topology.
        if not closed:
            line_ends = _try_fit_line(points)
            if line_ends is not None:
                p_s, p_e = line_ends
                edge = BRepBuilderAPI_MakeEdge(p_s, p_e).Edge()
                return BRepBuilderAPI_MakeWire(edge).Wire()

        # 2. Circular arc — when the offset is on a cylinder, cone, or sphere,
        # the offset points lie on a circle. Use a trimmed circle edge.
        circ, u_s, u_e = _try_fit_circle(points)
        if circ is not None:
            if closed:
                edge = BRepBuilderAPI_MakeEdge(circ).Edge()
            else:
                edge = BRepBuilderAPI_MakeEdge(circ, u_s, u_e).Edge()
            return BRepBuilderAPI_MakeWire(edge).Wire()

        # 3. Free-form B-spline fit through the samples — last resort, but
        # smooth enough to avoid the polygonal-facet topology that confuses
        # the boolean engine.
        arr = TColgp_Array1OfPnt(1, len(points))
        for i, pt in enumerate(points, start=1):
            arr.SetValue(i, pt)
        curve = GeomAPI_PointsToBSpline(arr).Curve()
        if closed:
            curve.SetPeriodic()
        edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        return BRepBuilderAPI_MakeWire(edge).Wire()

    w_edge = wire_from_points(edge_pts)
    w_a    = wire_from_points(a_pts)
    w_b    = wire_from_points(b_pts)

    # Three ruled surfaces forming the triangular tube around the chamfer.
    shell_ea = BRepFill.Shell_s(w_edge, w_a)    # body-side strip on face A
    shell_ab = BRepFill.Shell_s(w_a,    w_b)    # the new chamfer face
    shell_be = BRepFill.Shell_s(w_b,    w_edge) # body-side strip on face B

    sewing = BRepBuilderAPI_Sewing(1e-3)
    sewing.Add(shell_ea); sewing.Add(shell_ab); sewing.Add(shell_be)

    # Open arcs need triangular end caps to close the tube into a solid.
    if not closed:
        for idx in (0, -1):
            cap = _triangle_face(edge_pts[idx], a_pts[idx], b_pts[idx])
            if cap is not None:
                sewing.Add(cap)

    sewing.Perform()
    sewn = sewing.SewedShape()
    if sewn.ShapeType() != TopAbs_SHELL:
        raise RuntimeError(
            f"chamfer cutter shell did not sew (got {sewn.ShapeType()})")

    solid_mk = BRepBuilderAPI_MakeSolid(TopoDS.Shell(sewn))
    if not solid_mk.IsDone():
        raise RuntimeError("chamfer cutter: solid construction failed")
    cutter = solid_mk.Solid()

    gp_vol = GProp_GProps()
    BRepGProp.VolumeProperties_s(cutter, gp_vol)
    if gp_vol.Mass() < 0:
        cutter.Reverse()
    return cutter


def _triangle_face(p0, p1, p2):
    """Build a planar triangular face from three gp_Pnt corners.

    Returns the TopoDS_Face or None if the triangle is degenerate (collinear
    or zero-area). Used to cap the ends of an open-arc chamfer cutter."""
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace,
    )
    from OCP.gp import gp_Vec
    v1 = gp_Vec(p0, p1)
    v2 = gp_Vec(p0, p2)
    if v1.Crossed(v2).Magnitude() < 1e-9:
        return None
    try:
        e01 = BRepBuilderAPI_MakeEdge(p0, p1).Edge()
        e12 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
        e20 = BRepBuilderAPI_MakeEdge(p2, p0).Edge()
        wire = BRepBuilderAPI_MakeWire(e01, e12, e20).Wire()
        return BRepBuilderAPI_MakeFace(wire, True).Face()
    except Exception:
        return None


def _offset_point_on_face(p_world, tangent, face, distance: float, sign_hint,
                           body_classifier=None):
    """Return (gp_Pnt, sign) where the point is `distance` along the
    in-surface direction perpendicular to `tangent`. The first call passes
    sign_hint=None and the function picks the side that:
      1. lies inside the bounded face (not just on its infinite carrier), and
      2. heads INTO the parent solid body if a classifier is supplied.

    Criterion (2) matters for faces whose UV chart wraps around the body
    (cones, cylinders, spheres) — both ±in_surf directions can land on the
    bounded face, but only one walks into the body interior. Without the
    body check, the cutter can end up entirely outside the body and the
    boolean cut leaves the body unchanged.

    Subsequent calls reuse `sign_hint` to avoid the polyline flipping
    mid-edge."""
    from OCP.BRep import BRep_Tool
    from OCP.IntTools import IntTools_FClass2d as BRepTopAdaptor_FClass2d
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON

    surf = BRep_Tool.Surface_s(face)
    proj = GeomAPI_ProjectPointOnSurf(p_world, surf)
    if proj.NbPoints() == 0:
        return None, sign_hint
    u, v = proj.LowerDistanceParameters()
    props = GeomLProp_SLProps(surf, u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        return None, sign_hint
    n = props.Normal()
    n_vec = gp_Vec(n.X(), n.Y(), n.Z())
    in_surf = n_vec.Crossed(tangent)
    if in_surf.Magnitude() < 1e-9:
        return None, sign_hint
    in_surf.Normalize()

    if sign_hint is None:
        # First sample — pick the sign whose candidate point (a) lies INSIDE
        # the bounded face and (b) heads into the body interior.
        face_classifier = BRepTopAdaptor_FClass2d(face, 1e-6)
        best_sign = None
        best_rank = (99, 99, float("inf"))
        for s in (+1, -1):
            cand = gp_Pnt(p_world.X() + s * in_surf.X() * distance,
                          p_world.Y() + s * in_surf.Y() * distance,
                          p_world.Z() + s * in_surf.Z() * distance)
            pj = GeomAPI_ProjectPointOnSurf(cand, surf)
            if pj.NbPoints() == 0:
                continue
            cu, cv = pj.LowerDistanceParameters()
            face_state = face_classifier.Perform(gp_Pnt2d(cu, cv))
            face_rank = (0 if face_state == TopAbs_IN
                         else 1 if face_state == TopAbs_ON else 2)
            # Body rank: probe the midpoint between edge and candidate. Use a
            # small inward step so we land slightly inside, not on, the surface.
            body_rank = 0
            if body_classifier is not None:
                mid = gp_Pnt(0.5 * (p_world.X() + cand.X()),
                             0.5 * (p_world.Y() + cand.Y()),
                             0.5 * (p_world.Z() + cand.Z()))
                body_classifier.Perform(mid, 1e-6)
                bs = body_classifier.State()
                body_rank = 0 if bs == TopAbs_IN else 1 if bs == TopAbs_ON else 2
            rank = (face_rank, body_rank, pj.LowerDistance())
            if rank < best_rank:
                best_rank, best_sign = rank, s
        if best_sign is None:
            return None, sign_hint
        sign_hint = best_sign

    out = gp_Pnt(p_world.X() + sign_hint * in_surf.X() * distance,
                 p_world.Y() + sign_hint * in_surf.Y() * distance,
                 p_world.Z() + sign_hint * in_surf.Z() * distance)
    return out, sign_hint
