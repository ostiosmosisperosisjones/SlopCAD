"""
cad/body_check.py

Health check for an op's result solid: the kind of damage OCCT's own
BRepCheck validity pass lets through, but that makes every later boolean
unreliable (a cut keeps phantom pieces, a union leaves inside-out shells).

  - inside-out solid   — negative volume
  - non-manifold body  — open edges (the shell has a hole) or an edge shared
                         by more than two faces
  - self-intersection  — faces/edges of the body cross each other

Both checks run in kernel workers.  The self-intersection pass can take
minutes on a large BSpline body, so the viewport schedules everything in the
background with a timeout (viewer/body_health.py); describe() turns the
findings into the history entry's warning text.
"""

from __future__ import annotations

from cad.kernel_pool import offload


def fingerprint(shape) -> tuple | None:
    """Cheap identity for a result shape, stable across replays that rebuild
    the same geometry — so a check isn't repeated for an unchanged result."""
    if shape is None:
        return None
    try:
        bb = shape.bounding_box()
        vol = sum(s.volume for s in shape.solids())
        return (len(shape.faces()), len(shape.edges()), round(vol, 3),
                tuple(round(v, 3) for v in (bb.min.X, bb.min.Y, bb.min.Z,
                                            bb.max.X, bb.max.Y, bb.max.Z)))
    except Exception:
        return None


def _non_manifold(occ) -> str:
    """Count open edges (one face) and over-shared edges (3+ faces)."""
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.collections import (
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher
        as TopTools_IndexedDataMapOfShapeListOfShape)
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS
    emap = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(occ, TopAbs_EDGE, TopAbs_FACE, emap)
    open_n = over_n = 0
    for i in range(1, emap.Extent() + 1):
        edge = TopoDS.Edge(emap.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        # A seam edge lists its one face twice; count distinct faces.
        faces = []
        for f in emap.FindFromIndex(i):
            if not any(f.IsSame(g) for g in faces):
                faces.append(f)
        seam = len(faces) == 1 and emap.FindFromIndex(i).Size() >= 2
        if len(faces) == 1 and not seam:
            open_n += 1
        elif len(faces) > 2:
            over_n += 1
    parts = []
    if open_n:
        parts.append(f"{open_n} open edge{'s' * (open_n != 1)}")
    if over_n:
        parts.append(f"{over_n} edge{'s' * (over_n != 1)} shared by 3+ faces")
    return ", ".join(parts)


@offload()
def quick_problems(shape) -> list[str]:
    """Inside-out and non-manifold checks — milliseconds even on big bodies."""
    solids = shape.solids()
    if not solids:
        return []
    problems = []
    if any(s.volume < 0 for s in solids):
        problems.append("inside-out solid (negative volume)")
    nm = _non_manifold(shape.wrapped)
    if nm:
        problems.append(f"non-manifold body ({nm})")
    return problems


@offload()
def self_intersections(shape) -> int:
    """Number of self-intersections.  Slow on large BSpline bodies (a minute
    or more) — callers pass a timeout."""
    from OCP.BOPAlgo import BOPAlgo_ArgumentAnalyzer
    aa = BOPAlgo_ArgumentAnalyzer()
    aa.SetShape1(shape.wrapped)
    aa.SelfInterMode = True
    aa.Perform()
    return sum(1 for r in aa.GetCheckResult()
               if r.GetCheckStatus().name == "BOPAlgo_SelfIntersect")


def describe(problems: list[str]) -> str:
    if not problems:
        return ""
    return ("Result is " + "; ".join(problems)
            + " — later booleans on this body may give wrong results")
