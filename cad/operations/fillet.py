"""
cad/operations/fillet.py

3D edge fillet operations.
Uses OCC BRepFilletAPI_MakeFillet for multi-edge filleting.

Public API
----------
fillet_edges(shape, face_indices, edge_occs, radius)
    Commit-quality fillet.  Accepts either face indices (all edges of those
    faces are filleted) or direct TopoDS_Edge objects, or both.

fillet_preview(shape, face_indices, edge_occs, radius)
    Same but used for live preview — identical result, separate name for clarity.
"""

from build123d import Compound
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.ChFi3d import ChFi3d_FilletShape


def _collect_edges(shape, face_indices, edge_occs):
    """
    Return a deduplicated list of TopoDS_Edge objects from:
      - all edges belonging to each face in face_indices
      - the directly supplied edge_occs list
    """
    seen = set()
    unique = []

    def _add(e_occ):
        eid = id(e_occ)
        if eid not in seen:
            seen.add(eid)
            unique.append(e_occ)

    all_faces = list(shape.faces())
    for fi in face_indices:
        if fi >= len(all_faces):
            raise RuntimeError(f"Fillet: face_idx {fi} out of range")
        for edge in all_faces[fi].edges():
            _add(edge.wrapped)

    for e_occ in (edge_occs or []):
        _add(e_occ)

    return unique


def _add_edge(nf, e_occ, r):
    """Add one edge to a MakeFillet builder at radius spec *r*.

    r may be a scalar (constant) or an (r1, r2) pair (linear taper along the
    edge, OCCT Add(R1, R2, E)).
    """
    if isinstance(r, (tuple, list)):
        r1, r2 = float(r[0]), float(r[1])
        if r1 <= 0 or r2 <= 0:
            raise ValueError(f"Fillet: radii must be positive, got {r}")
        nf.Add(r1, r2, e_occ)
    else:
        if r <= 0:
            raise ValueError(f"Fillet: radius must be positive, got {r}")
        nf.Add(float(r), e_occ)


def _run_fillet(source_occ, unique_edges, radius):
    """Build a fillet on `unique_edges`.

    radius may be:
      - a scalar              → constant radius on every edge
      - an (r1, r2) tuple     → radius varies linearly from edge start to end
                                (variable-radius fillet, OCCT Add(R1, R2, E))
      - a list (one entry per edge, each a scalar or (r1, r2) tuple) → per-edge
                                radii, aligned positionally with unique_edges

    Contract to disambiguate: a *tuple* always means a taper pair; a *list*
    always means per-edge values. So per-edge tapers are list-of-tuples.
    """
    if not unique_edges:
        raise RuntimeError("Fillet: no edges selected")

    per_edge = isinstance(radius, list)
    if per_edge and len(radius) != len(unique_edges):
        raise ValueError(
            f"Fillet: per-edge radius list has {len(radius)} entries for "
            f"{len(unique_edges)} edges")

    nf = BRepFilletAPI_MakeFillet(source_occ, ChFi3d_FilletShape.ChFi3d_Rational)
    for i, e_occ in enumerate(unique_edges):
        try:
            _add_edge(nf, e_occ, radius[i] if per_edge else radius)
        except ValueError:
            raise
        except Exception:
            continue   # skip seam / degenerate edges

    if not nf.IsDone():
        try:
            nf.Build()
        except Exception as ex:
            raise RuntimeError(f"Fillet: build failed — {ex}")
    if not nf.IsDone():
        raise RuntimeError(f"Fillet failed — {_fillet_failure_reason(nf)}")

    return Compound(nf.Shape())


def _fillet_failure_reason(nf) -> str:
    """Human-readable reason a BRepFilletAPI_MakeFillet build didn't complete.

    BRepFilletAPI_MakeFillet has no .Error() in this OCP build; query the real
    status surface instead (contour stripe status + faulty vertices/contours).
    A faulty vertex is the usual cause: the fillet runs into a high-valence
    junction (e.g. a corner from a boolean union of many parts) and the kernel
    can't resolve the rounded surface there — often only a smaller radius fits.
    """
    parts = []
    try:
        nfv = nf.NbFaultyVertices()
        if nfv:
            parts.append(f"{nfv} faulty vertex/vertices (corner too complex "
                         f"for this radius — try a smaller radius)")
    except Exception:
        pass
    try:
        nfc = nf.NbFaultyContours()
        if nfc:
            parts.append(f"{nfc} faulty contour(s)")
    except Exception:
        pass
    try:
        for i in range(1, nf.NbContours() + 1):
            st = nf.StripeStatus(i)
            if "Ok" not in str(st):
                parts.append(f"contour {i}: {st}")
    except Exception:
        pass
    return "; ".join(parts) if parts else "kernel could not build the fillet"


def fillet_edges(shape, face_indices: list, edge_occs: list, radius: float):
    """Commit-quality fillet from face indices and/or direct edge objects.

    OCCT's fillet kernel sometimes rejects valid-but-imperfect geometry with
    "no suitable edges for chamfer or fillet" (common on STEP imports and on
    BSpline faces from drafted/oblique extrudes). When the first attempt fails
    that way, heal the shape (same pipeline as STEP import) and retry once —
    this rescues the healable cases. Geometry that is genuinely hostile to the
    fillet solver still raises, so callers can surface a clear message.
    """
    source_occ = shape.wrapped if hasattr(shape, 'wrapped') else shape
    unique = _collect_edges(shape, face_indices, edge_occs)
    try:
        return _run_fillet(source_occ, unique, radius)
    except RuntimeError:
        healed = _healed_with_edges(shape, face_indices, edge_occs)
        if healed is None:
            raise
        healed_shape, healed_source, healed_edges = healed
        # Retry on the healed shape; if it still fails, propagate so the panel
        # can tell the user this geometry can't be filleted.
        return _run_fillet(healed_source, healed_edges, radius)


def _healed_with_edges(shape, face_indices, edge_occs):
    """Heal `shape` and re-resolve the requested edges against the healed body.

    Returns (healed_shape, healed_source_occ, healed_edge_list) or None if
    healing produced nothing usable. Edges are re-matched by midpoint because
    healing creates new TopoDS sub-shapes that won't be IsSame() as the
    originals."""
    try:
        from cad.importer import _heal
        from build123d import Compound as _Compound
        source_occ = shape.wrapped if hasattr(shape, 'wrapped') else shape
        healed_occ = _heal(source_occ)
        healed_shape = _Compound(healed_occ)
        solids = list(healed_shape.solids())
        if solids:
            healed_shape = solids[0]
        # Original edges we were asked to fillet (pre-heal), as TopoDS_Edge.
        wanted = _collect_edges(shape, face_indices, edge_occs)
        healed_edges = _rematch_edges(healed_shape, wanted)
        if not healed_edges:
            return None
        healed_source = healed_shape.wrapped
        return healed_shape, healed_source, healed_edges
    except Exception:
        return None


def _rematch_edges(healed_shape, wanted_occ_edges):
    """Find edges in `healed_shape` whose midpoints match `wanted_occ_edges`."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    def _mid(e_occ):
        a = BRepAdaptor_Curve(e_occ)
        p = a.Value((a.FirstParameter() + a.LastParameter()) * 0.5)
        return (round(p.X(), 2), round(p.Y(), 2), round(p.Z(), 2))

    wanted_mids = set()
    for e in wanted_occ_edges:
        try:
            wanted_mids.add(_mid(e))
        except Exception:
            continue

    matched = []
    for e in healed_shape.edges():
        try:
            if _mid(e.wrapped) in wanted_mids:
                matched.append(e.wrapped)
        except Exception:
            continue
    return matched


def fillet_edges_subdivided(shape, face_indices: list, edge_occs: list,
                            radius: float, segments: int = 4):
    """Fillet attempt with each selected edge split into *segments* sub-edges.

    The ChFi3d contour walker sometimes stalls or fails on a single long /
    closed / high-curvature edge but converges when the same edge is filleted as
    a chain of shorter arcs. We split each edge by parameter range into co-edges
    of the original curve (same geometry, new TopoDS_Edge bounds), collect them
    against the shape, and fillet the chain.

    Raises like _run_fillet if even the subdivided chain can't build.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    source_occ = shape.wrapped if hasattr(shape, 'wrapped') else shape
    base_edges = _collect_edges(shape, face_indices, edge_occs)
    sub_edges = []
    for e in base_edges:
        try:
            a = BRepAdaptor_Curve(e)
            curve = a.Curve().Curve()          # underlying Geom_Curve handle
            u0, u1 = a.FirstParameter(), a.LastParameter()
            step = (u1 - u0) / segments
            for k in range(segments):
                lo = u0 + k * step
                hi = u0 + (k + 1) * step
                mk = BRepBuilderAPI_MakeEdge(curve, lo, hi)
                if mk.IsDone():
                    sub_edges.append(mk.Edge())
                else:
                    sub_edges.append(e)        # fall back to the whole edge
                    break
        except Exception:
            sub_edges.append(e)
    if not sub_edges:
        raise RuntimeError("Fillet: subdivision produced no edges")
    return _run_fillet(source_occ, sub_edges, radius)


def fillet_preview(shape, face_indices: list, edge_occs: list, radius: float):
    """Live-preview fillet — same as fillet_edges, separate name for clarity."""
    return fillet_edges(shape, face_indices, edge_occs, radius)


# ---------------------------------------------------------------------------
# Back-compat aliases used by existing callers
# ---------------------------------------------------------------------------

def fillet_face(shape, face_indices, radius: float):
    return fillet_edges(shape, face_indices, [], radius)


def fillet_face_preview(shape, face_indices, all_faces, radius: float):
    return fillet_edges(shape, face_indices, [], radius)
