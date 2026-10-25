import math

from cad.kernel_pool import offload
from cad.solid_ref import as_compound
from build123d import Compound
from OCP.BRepOffset import BRepOffset_MakeOffset, BRepOffset_Skin
from OCP.GeomAbs import GeomAbs_Arc
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape
from OCP.collections import (
    IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher
    as _EdgeFaces)
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepLib import BRepLib_MakeFace
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepOffset import BRepOffset_Analyse
from OCP.BRepTools import BRepTools
from OCP.ChFiDS import ChFiDS_Concave
from OCP.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                         GeomAbs_Sphere, GeomAbs_Torus)
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt, gp_Vec

from cad.debug import log, dump_brep, shape_stats


_ANALYTIC = (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
             GeomAbs_Torus)


def _normal_at(face, p):
    pr = GeomAPI_ProjectPointOnSurf(p, BRep_Tool.Surface_s(face))
    u, v = pr.LowerDistanceParameters()
    n = BRepLProp_SLProps(BRepAdaptor_Surface(face), u, v, 1, 1e-6).Normal()
    if face.Orientation() == TopAbs_REVERSED:
        n.Reverse()
    return n


def _cut_face(body_occ, face_occ, thickness: float):
    """
    The face to offset into a cut tool. The offset skin's side walls are ruled
    along the face normal, so where a neighbour meets the face obliquely the
    cut leaves a thin lip of material between wall and neighbour. If every
    boundary edge is convex (extending goes into air), return the face's
    surface grown past its boundary by just enough to clear that lip;
    otherwise (concave edge, freeform surface, no oblique neighbour) the face
    itself.
    """
    ad = BRepAdaptor_Surface(face_occ)
    if ad.GetType() not in _ANALYTIC:
        return face_occ
    an = BRepOffset_Analyse(body_occ, 0.01)
    anc = _EdgeFaces()
    TopExp.MapShapesAndAncestors_s(body_occ, TopAbs_EDGE, TopAbs_FACE, anc)
    t = abs(thickness)
    margin = 0.0
    exp = TopExp_Explorer(face_occ, TopAbs_EDGE)
    while exp.More():
        ed = TopoDS.Edge(exp.Current()); exp.Next()
        if BRep_Tool.IsClosed_s(ed, face_occ):
            continue                                  # seam
        if any(c.Type() == ChFiDS_Concave for c in an.Type(ed)):
            return face_occ
        if not anc.Contains(ed):
            continue
        c = BRepAdaptor_Curve(ed)
        p = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
        n1 = _normal_at(face_occ, p)
        for nb in anc.FindFromKey(ed):
            nb = TopoDS.Face(nb)
            if nb.IsSame(face_occ):
                continue
            d = min(abs(n1.Dot(_normal_at(nb, p))), 0.9999)  # tangent: ~70·t
            margin = max(margin, t * d / math.sqrt(1 - d * d))  # lip width
    if margin < 1e-6:
        return face_occ
    margin = margin * 1.5 + t

    u0, u1, v0, v1 = BRepTools.UVBounds_s(face_occ)
    P, Du, Dv = gp_Pnt(), gp_Vec(), gp_Vec()
    ad.D1((u0 + u1) / 2, (v0 + v1) / 2, P, Du, Dv)

    def grow(a, b, d, periodic, period):
        if periodic and (b - a) + 2 * d >= period - 1e-9:
            return a, a + period
        return a - d, b + d

    U0, U1 = grow(u0, u1, margin / max(Du.Magnitude(), 1e-9),
                  ad.IsUPeriodic(), ad.UPeriod() if ad.IsUPeriodic() else 0)
    V0, V1 = grow(v0, v1, margin / max(Dv.Magnitude(), 1e-9),
                  ad.IsVPeriodic(), ad.VPeriod() if ad.IsVPeriodic() else 0)
    if ad.GetType() == GeomAbs_Sphere:
        V0, V1 = max(V0, -math.pi / 2), min(V1, math.pi / 2)
    mf = BRepLib_MakeFace(BRep_Tool.Surface_s(face_occ), U0, U1, V0, V1, 1e-7)
    if not mf.IsDone():
        return face_occ
    nf = mf.Face()
    if face_occ.Orientation() == TopAbs_REVERSED:
        nf.Reverse()
    log("Thicken", f"cut tool grown by {margin:.4g} past convex edges")
    return nf


def _cut(body_occ, face_occ, thickness: float, tol: float):
    """Cut with the grown face's slab, falling back to the exact slab."""
    grown = _cut_face(body_occ, face_occ, thickness)
    faces = [grown, face_occ] if grown is not face_occ else [face_occ]
    for i, f in enumerate(faces):
        try:
            tool_occ = _offset_solid(f, thickness, tol)
        except RuntimeError:
            if i + 1 < len(faces):
                continue
            raise
        # see thicken_face for why the copy
        tool_occ = BRepBuilderAPI_Copy(tool_occ).Shape()
        lst_a = TopTools_ListOfShape(); lst_a.Append(body_occ)
        lst_b = TopTools_ListOfShape(); lst_b.Append(tool_occ)
        op = BRepAlgoAPI_Cut()
        op.SetArguments(lst_a)
        op.SetTools(lst_b)
        op.SetRunParallel(True)
        op.Build()
        ok = op.IsDone()
        if ok and i + 1 < len(faces):
            shp = op.Shape()
            ok = (BRepCheck_Analyzer(shp).IsValid()
                  and TopExp_Explorer(shp, TopAbs_SOLID).More())
        if ok or i + 1 == len(faces):
            return op, tool_occ
        log("Thicken", "grown cut tool failed — using exact slab")


def _offset_solid(face_occ, thickness: float, tol: float) -> object:
    # Positive thickness → slab grows outward (away from face normal).
    # Negative thickness → slab grows inward; OCCT accepts negative offset values
    # which flip the offset direction, so the tool solid sits inside the body.
    om = BRepOffset_MakeOffset()
    om.Initialize(face_occ, float(thickness), tol,
                  BRepOffset_Skin, False, False, GeomAbs_Arc, True)
    om.MakeOffsetShape()
    if not om.IsDone():
        raise RuntimeError(f"Thicken failed — offset kernel error: {om.Error()}")
    return om.Shape()


@offload()
def thicken_face_preview(face_occ, thickness: float) -> Compound:
    """Offset slab only — kept for back-compat, not used for body preview."""
    if thickness == 0:
        raise ValueError("thickness must be non-zero")
    return Compound(_offset_solid(face_occ, thickness, 1e-2))


@offload()
def thicken_body_preview(body_occ, face_occ, thickness: float) -> Compound:
    """Full fuse/cut at coarse tolerance — used for live preview."""
    if thickness == 0:
        raise ValueError("thickness must be non-zero")
    if thickness < 0:
        op, _ = _cut(body_occ, face_occ, thickness, 1e-2)
    else:
        tool_occ = _offset_solid(face_occ, thickness, 1e-2)
        tool_occ = BRepBuilderAPI_Copy(tool_occ).Shape()  # see thicken_face for why
        lst_a = TopTools_ListOfShape(); lst_a.Append(body_occ)
        lst_b = TopTools_ListOfShape(); lst_b.Append(tool_occ)
        op = BRepAlgoAPI_Fuse()
        op.SetArguments(lst_a)
        op.SetTools(lst_b)
        op.SetRunParallel(True)
        op.Build()
    if not op.IsDone():
        raise RuntimeError("Thicken preview failed")
    return as_compound(op.Shape())


@offload()
def thicken_face(body_occ, face_occ, thickness: float) -> Compound:
    """
    Commit-quality thicken (thickness > 0) or cut (thickness < 0).
    Positive: offset slab fused into body.
    Negative: offset slab cut from body (removes material inward).
    """
    if thickness == 0:
        raise ValueError("thickness must be non-zero")

    log("Thicken", f"thickness={thickness}")
    log("Thicken", f"input body : {shape_stats(body_occ)}")
    log("Thicken", f"input face : {shape_stats(face_occ)}")
    dump_brep("thicken_input_body", body_occ)
    dump_brep("thicken_input_face", face_occ)

    tool_occ = _offset_solid(face_occ, thickness, 1e-4)
    log("Thicken", f"offset tool: {shape_stats(tool_occ)}")
    dump_brep("thicken_offset_tool", tool_occ)

    # The offset tool reuses TopoDS subshapes from the source face (its inner
    # cylindrical surface IS the original body face). OCCT's BOP silently
    # returns an empty shape when arguments share subshape identity — round-
    # trip the tool through BRepBuilderAPI_Copy to give it a fresh identity.
    tool_occ = BRepBuilderAPI_Copy(tool_occ).Shape()

    lst_a = TopTools_ListOfShape(); lst_a.Append(body_occ)
    lst_b = TopTools_ListOfShape(); lst_b.Append(tool_occ)

    if thickness > 0:
        op = BRepAlgoAPI_Fuse()
        op.SetArguments(lst_a)
        op.SetTools(lst_b)
        op.SetRunParallel(True)
        op.Build()
        if not op.IsDone():
            raise RuntimeError("Thicken fuse failed")
        result = as_compound(op.Shape())
        log("Thicken", f"fuse result: {shape_stats(result)}")
        dump_brep("thicken_fuse_result", result)
        if not list(result.solids()):
            # Some configurations (e.g. cylindrical face with both ends rounded
            # by fillet/chamfer) make OCCT's fuse silently return an empty
            # shape even though IsDone() is true. Fail loudly instead of
            # leaving the body looking deleted.
            raise RuntimeError("Thicken produced empty result")
    else:
        op, tool_occ = _cut(body_occ, face_occ, thickness, 1e-4)
        if not op.IsDone():
            raise RuntimeError("Thicken cut failed")
        result = as_compound(op.Shape())
        log("Thicken", f"cut result : {shape_stats(result)}")
        dump_brep("thicken_cut_result", result)
        if not list(result.solids()):
            raise RuntimeError("Thicken cut produced empty result")

    return result
