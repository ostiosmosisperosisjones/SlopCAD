"""
cad/operations/shell.py

shell_body — hollow a solid into walls of uniform thickness, leaving the
picked faces open (BRepOffsetAPI_MakeThickSolid).  Runs in a kernel worker:
a thick-solid build on a filleted body can take seconds.
"""

from __future__ import annotations

from cad.kernel_pool import offload


@offload()
def shell_body(body_occ, face_occs: list, thickness: float,
               outside: bool = False):
    """Shell the solid(s) of *body_occ* that own *face_occs* (removed, so the
    shell is open there); other solids of the body pass through untouched.
    No faces = a closed hollow.  thickness > 0; *outside* grows the walls
    out from the current skin instead of in.  Returns a Compound."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid
    from OCP.BRepOffset import BRepOffset_Skin
    from OCP.GeomAbs import GeomAbs_Arc, GeomAbs_Intersection
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.collections import List_TopoDS_Shape
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    from cad.solid_ref import as_compound

    if thickness <= 0:
        raise ValueError("Shell thickness must be positive")
    body = getattr(body_occ, "wrapped", body_occ)
    faces = [getattr(f, "wrapped", f) for f in face_occs]

    solids = []
    exp = TopExp_Explorer(body, TopAbs_SOLID)
    while exp.More():
        solids.append(TopoDS.Solid(exp.Current()))
        exp.Next()
    if not solids:
        raise RuntimeError("Shell needs a solid body")

    def owns(solid, face):
        e = TopExp_Explorer(solid, TopAbs_FACE)
        while e.More():
            if e.Current().IsSame(face):
                return True
            e.Next()
        return False

    targets = [s for s in solids if any(owns(s, f) for f in faces)] \
        if faces else solids
    if faces and len(targets) != 1:
        raise RuntimeError("Shell faces must all be on one solid")

    offset = thickness if outside else -thickness
    out = []
    for s in solids:
        if not any(s.IsSame(t) for t in targets):
            out.append(s)
            continue
        if not faces:
            out.append(_closed_hollow(s, thickness, outside))
            continue
        removed = List_TopoDS_Shape()
        for f in faces:
            removed.Append(f)
        shape = None
        expect = (_area(s) - sum(_area(f) for f in faces)) * thickness
        for join in (GeomAbs_Arc, GeomAbs_Intersection):
            mk = BRepOffsetAPI_MakeThickSolid()
            try:
                mk.MakeThickSolidByJoin(s, removed, offset, 1e-4,
                                        BRepOffset_Skin, False, False, join)
                mk.Build()
            except Exception:
                continue
            if not mk.IsDone():
                continue
            cand = mk.Shape()
            if not BRepCheck_Analyzer(cand).IsValid():
                # Arc joins on filleted bodies come back with one face out
                # of tolerance; ShapeFix settles it.
                from OCP.ShapeFix import ShapeFix_Shape
                fx = ShapeFix_Shape(cand)
                fx.Perform()
                cand = fx.Shape()
                if not BRepCheck_Analyzer(cand).IsValid():
                    continue
            if _plausible(_volume(cand), expect, outside, _volume(s)):
                shape = cand
                break
        if shape is None:
            raise RuntimeError(
                "Shell failed — OCCT can't grow these walls outward "
                "(try Inside)" if outside else
                f"Shell failed — {thickness:g} mm walls may be thicker than "
                f"the part allows (try thinner, or fewer open faces)")
        out.append(shape)

    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    for s in out:
        b.Add(comp, s)
    return as_compound(comp)


def _closed_hollow(solid, thickness: float, outside: bool):
    """No open faces: MakeThickSolid returns only the cavity, so cut an
    offset copy instead — inward: solid minus its shrunk self; outward:
    the grown self minus the solid."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
    from OCP.BRepOffset import BRepOffset_Skin
    from OCP.GeomAbs import GeomAbs_Arc, GeomAbs_Intersection
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.BRepCheck import BRepCheck_Analyzer

    mk = BRepOffsetAPI_MakeOffsetShape()
    mk.PerformByJoin(solid, thickness if outside else -thickness, 1e-4,
                     BRepOffset_Skin, False, False,
                     GeomAbs_Arc if outside else GeomAbs_Intersection)
    if not mk.IsDone():
        raise RuntimeError(f"Shell failed — can't offset the body by {thickness:g} mm")
    off = mk.Shape()
    if off.ShapeType() != TopAbs_SOLID:
        e = TopExp_Explorer(off, TopAbs_SHELL)
        if not e.More():
            raise RuntimeError("Shell failed — offset made no closed skin")
        off = BRepBuilderAPI_MakeSolid(TopoDS.Shell(e.Current())).Solid()
    a, b = (off, solid) if outside else (solid, off)
    cut = BRepAlgoAPI_Cut(a, b)
    cut.Build()
    if not cut.IsDone() or not BRepCheck_Analyzer(cut.Shape()).IsValid():
        raise RuntimeError(f"Shell failed — {thickness:g} mm walls may be "
                           f"thicker than the part allows")
    e = TopExp_Explorer(cut.Shape(), TopAbs_SOLID)
    return TopoDS.Solid(e.Current()) if e.More() else cut.Shape()


def _area(shape) -> float:
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, p)
    return p.Mass()


def _volume(shape) -> float:
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return p.Mass()


def _plausible(vol: float, expect: float, outside: bool,
               before: float) -> bool:
    """A wall's volume is about its kept skin area x thickness: less when
    thick inward walls crowd the inside, more when outward walls round
    corners.  MakeThickSolid can report success yet hand back the body
    unchanged, or a scrap when the walls don't fit — both fall outside."""
    if expect <= 0 or vol <= 0 or abs(vol - before) < 1e-6 * before:
        return False
    r = vol / expect
    return 0.8 <= r <= 3.0 if outside else 0.3 <= r <= 1.2
