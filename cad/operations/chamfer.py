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
import math
from build123d import Compound
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer
from OCP.TopExp import TopExp
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopoDS import TopoDS


def _build_edge_to_faces_map(shape_occ):
    """Return TopTools_IndexedDataMapOfShapeListOfShape mapping each edge to
    its adjacent faces."""
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape_occ, TopAbs_EDGE, TopAbs_FACE,
                                    edge_face_map)
    return edge_face_map


def _pick_reference_face(edge_occ, edge_face_map, flip: bool):
    """Return one of the faces adjacent to *edge_occ*.

    flip=False → the first face OCCT lists for this edge.
    flip=True  → the second adjacent face if one exists, else the first.

    Raises if the edge has no adjacent faces (a degenerate seam edge).
    """
    if not edge_face_map.Contains(edge_occ):
        raise RuntimeError("Chamfer: edge has no adjacent faces in this shape")
    face_list = edge_face_map.FindFromKey(edge_occ)
    n = face_list.Extent()
    if n == 0:
        raise RuntimeError("Chamfer: edge has no adjacent faces")
    pick = face_list.Last() if (flip and n >= 2) else face_list.First()
    return TopoDS.Face_s(pick)


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
        try:
            mk = BRepFilletAPI_MakeChamfer(source_occ)
            mk.AddDA(float(distance), angle_rad, e_occ, ref_face)
            mk.Build()
            if not mk.IsDone():
                results.append("OCCT chamfer kernel rejected this edge")
                continue
        except Exception as ex:
            results.append(str(ex))
            continue
        results.append(None)
    return results


def chamfer_edges(shape, edge_occs, distance: float, angle_deg: float,
                  flip_reference_face: bool = False):
    """Chamfer the given OCCT edges with distance + angle.

    shape         : build123d Shape (with .wrapped) or raw TopoDS_Shape
    edge_occs     : list of TopoDS_Edge to chamfer
    distance      : mm measured along the reference face
    angle_deg     : degrees, measured from the reference face (must be > 0
                    and < 90 — OCCT rejects 0 and >=90)
    flip_reference_face : swap to the other adjacent face for the angle
                          measurement, mirroring the asymmetry across the edge

    Tries OCCT's native chamfer kernel first; if that fails (e.g. drafted
    cone-to-plane edges, where the kernel refuses to build), falls back to
    a boolean-cut construction that always works at the cost of producing
    a slightly facetted chamfer face.
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
        mk.Build()
        if mk.IsDone():
            return Compound(mk.Shape())
        # Kernel built additions but failed to produce a result — fall through
        # to the boolean-cut fallback below.

    # Fallback: chamfer each edge by boolean cut. Slower and produces a
    # facetted cut face, but works on edges OCCT's chamfer kernel refuses
    # (typically drafted curved-to-planar boundaries).
    result_occ = source_occ
    success = 0
    last_error = None
    for e_occ in edge_occs:
        try:
            result_occ = _chamfer_edge_via_cut(
                result_occ, e_occ, distance, angle_rad, flip_reference_face)
            success += 1
        except Exception as ex:
            last_error = ex
            continue

    if success == 0:
        msg = "Chamfer: kernel failed and boolean fallback also failed"
        if last_error is not None:
            msg += f" — {last_error}"
        raise RuntimeError(msg)

    return Compound(result_occ)


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


def _chamfer_edge_via_cut(shape_occ, edge_occ, distance: float,
                           angle_rad: float, flip_reference_face: bool):
    """Cut a chamfer-shaped tool away from shape_occ for one edge."""
    edge_face_map = _build_edge_to_faces_map(shape_occ)
    if not edge_face_map.Contains(edge_occ):
        raise RuntimeError("edge has no adjacent faces")
    faces = list(edge_face_map.FindFromKey(edge_occ))
    if len(faces) < 2:
        raise RuntimeError("edge needs two adjacent faces")
    face_a = TopoDS.Face_s(faces[0])
    face_b = TopoDS.Face_s(faces[1])
    # `flip_reference_face` swaps which face is the "reference" — so
    # distance is measured along face_b instead of face_a.
    if flip_reference_face:
        face_a, face_b = face_b, face_a

    # AddDA semantics: distance along reference face, angle between the
    # cut plane and the reference. The cut intersects the other face at
    # distance * tan(angle).
    distance_b = distance * math.tan(angle_rad)
    cutter = _build_chamfer_cutter(edge_occ, face_a, face_b, distance, distance_b)

    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.TopTools import TopTools_ListOfShape
    la = TopTools_ListOfShape(); la.Append(shape_occ)
    lb = TopTools_ListOfShape(); lb.Append(cutter)
    op = BRepAlgoAPI_Cut()
    op.SetArguments(la); op.SetTools(lb); op.SetRunParallel(True)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("boolean-cut chamfer: cut failed")
    result = op.Shape()
    if not list(Compound(result).solids()):
        raise RuntimeError("boolean-cut chamfer: empty result")
    return result


def _build_chamfer_cutter(edge_occ, face_a, face_b,
                           distance_a: float, distance_b: float, N: int = 24):
    """Build a closed solid bounded by three swept ruled surfaces between
    the edge curve and its two offset curves on face_a and face_b.

    Each closed curve is reconstructed as either an exact circle (when
    the sample points lie on one — common for rotationally symmetric
    bodies) or a periodic B-spline otherwise. Both forms avoid the
    polygon-facet "buzzsaw" that plain polygonal wires produced; the
    exact-circle path additionally avoids the BSpline's internal knots
    that cause OCCT's boolean cut to fragment the chamfer face.

    All three curves are oriented consistently (same rotational sense
    as the sample sequence) so the resulting cutter has positive volume.

    For non-periodic source edges (open arcs, line segments) a polygonal
    wire is used — they don't suffer the seam issue.
    """
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax2
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_Sewing,
    )
    from OCP.BRepFill import BRepFill
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.Geom import Geom_Circle
    from OCP.GeomAPI import GeomAPI_PointsToBSpline
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopoDS import TopoDS

    adp = BRepAdaptor_Curve(edge_occ)
    u0, u1 = adp.FirstParameter(), adp.LastParameter()
    periodic = adp.IsPeriodic()
    n_samples = N if periodic else N + 1

    edge_pts, a_pts, b_pts = [], [], []
    sign_a = sign_b = None
    for k in range(n_samples):
        u = u0 + (u1 - u0) * k / N
        p = adp.Value(u)
        tv = adp.DN(u, 1)
        tan = gp_Vec(tv.X(), tv.Y(), tv.Z())
        if tan.Magnitude() < 1e-9:
            raise RuntimeError("degenerate tangent at sample")
        tan.Normalize()
        a, sign_a = _offset_point_on_face(p, tan, face_a, distance_a, sign_a)
        b, sign_b = _offset_point_on_face(p, tan, face_b, distance_b, sign_b)
        if a is None or b is None:
            raise RuntimeError(f"failed to project offset at sample {k}")
        edge_pts.append(p); a_pts.append(a); b_pts.append(b)

    def _try_fit_circle(points):
        """Return Geom_Circle if points lie on a circle (rel. std < 1%), else None.
        Circle is oriented so it traverses the points in their given order."""
        arr = np.array([[p.X(), p.Y(), p.Z()] for p in points])
        cen = arr.mean(axis=0)
        cnt = arr - cen
        _, _, vh = np.linalg.svd(cnt, full_matrices=False)
        normal = vh[2]
        proj   = cnt - np.outer(cnt @ normal, normal)
        radii  = np.linalg.norm(proj, axis=1)
        if radii.mean() < 1e-9 or radii.std() / radii.mean() > 0.01:
            return None
        x_norm = np.linalg.norm(proj[0])
        if x_norm < 1e-9:
            return None
        x_dir = proj[0] / x_norm
        # Orient: ensure the circle parameter direction matches the sample
        # direction (forward = normal × x_dir should align with points[1] − points[0]).
        forward = np.cross(normal, x_dir)
        if np.dot(forward, arr[1] - arr[0]) < 0:
            normal = -normal
        ax = gp_Ax2(
            gp_Pnt(float(cen[0]), float(cen[1]), float(cen[2])),
            gp_Dir(float(normal[0]), float(normal[1]), float(normal[2])),
            gp_Dir(float(x_dir[0]),  float(x_dir[1]),  float(x_dir[2])),
        )
        return Geom_Circle(ax, float(radii.mean()))

    def closed_wire(points):
        if not periodic:
            # Open edge — polygonal wire is fine; no seam to worry about.
            poly = BRepBuilderAPI_MakePolygon()
            for pt in points:
                poly.Add(pt)
            return poly.Wire()
        # Periodic edge — try exact circle first (best topology), fall
        # back to periodic B-spline.
        circ = _try_fit_circle(points)
        if circ is not None:
            edge = BRepBuilderAPI_MakeEdge(circ).Edge()
        else:
            arr = TColgp_Array1OfPnt(1, len(points))
            for i, pt in enumerate(points, start=1):
                arr.SetValue(i, pt)
            curve = GeomAPI_PointsToBSpline(arr).Curve()
            curve.SetPeriodic()
            edge = BRepBuilderAPI_MakeEdge(curve).Edge()
        return BRepBuilderAPI_MakeWire(edge).Wire()

    w_edge = closed_wire(edge_pts)
    w_a    = closed_wire(a_pts)
    w_b    = closed_wire(b_pts)

    # Three ruled surfaces forming the triangular tube around the chamfer.
    shell_ea = BRepFill.Shell_s(w_edge, w_a)    # body-side strip on face A
    shell_ab = BRepFill.Shell_s(w_a,    w_b)    # the new chamfer face
    shell_be = BRepFill.Shell_s(w_b,    w_edge) # body-side strip on face B

    sewing = BRepBuilderAPI_Sewing(1e-3)
    sewing.Add(shell_ea); sewing.Add(shell_ab); sewing.Add(shell_be)
    sewing.Perform()
    sewn = sewing.SewedShape()
    if sewn.ShapeType() != TopAbs_SHELL:
        raise RuntimeError(
            f"chamfer cutter shell did not sew (got {sewn.ShapeType()})")

    solid_mk = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(sewn))
    if not solid_mk.IsDone():
        raise RuntimeError("chamfer cutter: solid construction failed")
    cutter = solid_mk.Solid()

    # Validate orientation by volume; flip if inverted.
    gp_vol = GProp_GProps()
    BRepGProp.VolumeProperties_s(cutter, gp_vol)
    if gp_vol.Mass() < 0:
        cutter.Reverse()
    return cutter


def _offset_point_on_face(p_world, tangent, face, distance: float, sign_hint):
    """Return (gp_Pnt, sign) where the point is `distance` along the
    in-surface direction perpendicular to `tangent`. The first call passes
    sign_hint=None and the function picks the side that lands inside the
    bounded face (not just on its infinite carrier surface); subsequent
    calls reuse the same sign to avoid the polyline flipping mid-edge.

    The returned point is NOT snapped back to the face surface — staying
    in the tangent plane keeps consecutive samples colinear, which is what
    the ruled-surface loft needs to produce a clean cutter."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
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
        # First sample — pick the sign whose candidate point lies INSIDE
        # the bounded face (not just on its underlying surface). Falls
        # back to surface-proximity if neither sign is unambiguously IN.
        classifier = BRepTopAdaptor_FClass2d(face, 1e-6)
        best_sign = None
        best_state_rank = 99
        best_d = float("inf")
        for s in (+1, -1):
            cand = gp_Pnt(p_world.X() + s * in_surf.X() * distance,
                          p_world.Y() + s * in_surf.Y() * distance,
                          p_world.Z() + s * in_surf.Z() * distance)
            pj = GeomAPI_ProjectPointOnSurf(cand, surf)
            if pj.NbPoints() == 0:
                continue
            cu, cv = pj.LowerDistanceParameters()
            state = classifier.Perform(gp_Pnt2d(cu, cv))
            # Rank: IN=0 (best), ON=1, anything else=2.
            rank = 0 if state == TopAbs_IN else (1 if state == TopAbs_ON else 2)
            d = pj.LowerDistance()
            if (rank, d) < (best_state_rank, best_d):
                best_state_rank, best_d, best_sign = rank, d, s
        if best_sign is None:
            return None, sign_hint
        sign_hint = best_sign

    out = gp_Pnt(p_world.X() + sign_hint * in_surf.X() * distance,
                 p_world.Y() + sign_hint * in_surf.Y() * distance,
                 p_world.Z() + sign_hint * in_surf.Z() * distance)
    return out, sign_hint
