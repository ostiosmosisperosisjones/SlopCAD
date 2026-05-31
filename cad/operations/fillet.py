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


def _run_fillet(source_occ, unique_edges, radius: float):
    if not unique_edges:
        raise RuntimeError("Fillet: no edges selected")
    if radius <= 0:
        raise ValueError(f"Fillet: radius must be positive, got {radius}")

    nf = BRepFilletAPI_MakeFillet(source_occ, ChFi3d_FilletShape.ChFi3d_Rational)
    for e_occ in unique_edges:
        try:
            nf.Add(radius, e_occ)
        except Exception:
            continue   # skip seam / degenerate edges

    if not nf.IsDone():
        try:
            nf.Build()
        except Exception as ex:
            raise RuntimeError(f"Fillet: build failed — {ex}")
    if not nf.IsDone():
        raise RuntimeError(f"Fillet failed — kernel returned: {nf.Error()}")

    return Compound(nf.Shape())


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
