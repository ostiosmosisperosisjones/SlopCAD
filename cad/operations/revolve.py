"""
cad/operations/revolve.py
"""

from cad.kernel_pool import offload
from cad.solid_ref import as_compound
import numpy as np
from build123d import revolve, Plane, Face, Compound, Axis, Vector, Location
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCP.gp import gp_Ax1, gp_Pnt, gp_Dir


@offload()
def revolve_face_direct(body_shape, face: Face, axis_point: np.ndarray,
                         axis_dir: np.ndarray, angle_deg: float, side: int = 1):
    """
    Revolve a face object around an axis by angle_deg degrees.

    Used for sketch-profile revolve where the face is built from a SketchEntry.

    Positive angle → counter-clockwise when viewed from axis_dir tip.

    If body_shape is None (first operation on a new body), the revolved solid
    is returned as-is with no boolean.

    Returns the resulting build123d Compound/Solid, or raises on failure.
    """
    if angle_deg == 0:
        return body_shape

    Plane(face)  # raises if not planar

    revolved = _do_revolve_solid(face, axis_point, axis_dir, angle_deg, side)

    if body_shape is None:
        return as_compound(revolved)

    return _do_revolve_boolean(body_shape, revolved, angle_deg)


def split_face_at_axis(face, axis_point, axis_dir) -> tuple[list, list] | None:
    """The profile cut along the revolve axis, as (faces on the +m side,
    faces on the -m side) with m = axis_dir x face normal, when the axis lies
    in the face's plane and runs through its interior; else None.  OCCT
    can't revolve a face that crosses its own axis ("BRep_API: command not
    done"); one side of it can be."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeHalfSpace
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pln
    f = face if isinstance(face, Face) else Face(face)
    pl = Plane(f)
    n = np.array([pl.z_dir.X, pl.z_dir.Y, pl.z_dir.Z], dtype=float)
    o = np.array([pl.origin.X, pl.origin.Y, pl.origin.Z], dtype=float)
    d = np.asarray(axis_dir, dtype=float); d = d / np.linalg.norm(d)
    p = np.asarray(axis_point, dtype=float)
    size = max(1.0, f.bounding_box().diagonal)
    if abs(float(d @ n)) > 1e-9 or abs(float((p - o) @ n)) > 1e-7 * size:
        return None                                   # axis not in the profile plane
    m = np.cross(d, n); m /= np.linalg.norm(m)
    # Signed distances of the profile's sample points from the axis line, in
    # its plane: both signs (beyond tolerance) means the axis crosses it.
    side = []
    ex = TopExp_Explorer(f.wrapped, TopAbs_EDGE)
    while ex.More():
        c = BRepAdaptor_Curve(TopoDS.Edge(ex.Current()))
        for k in range(17):
            t = c.FirstParameter() + (c.LastParameter() - c.FirstParameter()) * k / 16
            q = c.Value(t)
            side.append(float((np.array([q.X(), q.Y(), q.Z()]) - p) @ m))
        ex.Next()
    tol = 1e-7 * size
    if not side or min(side) > -tol or max(side) < tol:
        return None                                   # one side only (or touching)
    cut_plane = BRepBuilderAPI_MakeFace(
        gp_Pln(gp_Pnt(*p.tolist()), gp_Dir(*m.tolist()))).Face()
    out = []
    for sgn in (1.0, -1.0):
        ref = p + sgn * m * size
        hs = BRepPrimAPI_MakeHalfSpace(cut_plane, gp_Pnt(*ref.tolist())).Solid()
        com = BRepAlgoAPI_Common(f.wrapped, hs)
        if not com.IsDone():
            raise RuntimeError("Revolve: couldn't split the profile at the axis")
        out.append([x for x in Compound(com.Shape()).faces()
                    if x.area > 1e-12 * size ** 2])
    return (out[0], out[1]) if out[0] and out[1] else None


def profile_side(face, axis_point, axis_dir, side: int = 1):
    """The faces revolve uses: *face* itself, or — when the axis runs
    through it — only its part on *side* (+1: axis_dir x normal, -1: the
    other side)."""
    halves = split_face_at_axis(face, axis_point, axis_dir)
    if halves is None:
        return [face if isinstance(face, Face) else Face(face)]
    return halves[0] if side >= 0 else halves[1]


def _do_revolve_solid(face: Face, axis_point: np.ndarray,
                       axis_dir: np.ndarray, angle_deg: float, side: int = 1):
    """Produce the raw revolved solid (no boolean).  A profile the axis runs
    through is revolved on one *side* of the axis only (user's call: a disc
    revolved about its diameter whole makes a ball at any angle)."""
    d = axis_dir / np.linalg.norm(axis_dir)
    ax = Axis(
        origin=Vector(float(axis_point[0]), float(axis_point[1]), float(axis_point[2])),
        direction=Vector(float(d[0]), float(d[1]), float(d[2])),
    )
    pieces = profile_side(face, axis_point, d, side)
    if len(pieces) == 1:
        return revolve(pieces[0], axis=ax, revolution_arc=angle_deg)
    # A non-convex profile can leave several pieces on one side: disjoint, so
    # their revolutions only touch at most.
    solids = [revolve(x, axis=ax, revolution_arc=angle_deg) for x in pieces]
    shape = solids[0].wrapped
    for x in solids[1:]:
        fu = BRepAlgoAPI_Fuse(shape, x.wrapped)
        if not fu.IsDone():
            raise RuntimeError("Revolve: couldn't join the profile's pieces")
        shape = fu.Shape()
    sol = Compound(shape).solids()
    return sol[0] if len(sol) == 1 else Compound(children=list(sol))


def _do_revolve_boolean(shape, revolved_solid, angle_deg: float):
    """Boolean fuse or cut of revolved solid against existing shape."""
    if angle_deg > 0:
        op = BRepAlgoAPI_Fuse()
        op.SetArguments(_to_list(shape.wrapped))
        op.SetTools(_to_list(revolved_solid.wrapped))
        op.SetRunParallel(True)
        op.Build()
        if not op.IsDone():
            raise RuntimeError("Boolean fuse (revolve) failed")
        return as_compound(op.Shape())
    else:
        op = BRepAlgoAPI_Cut()
        op.SetArguments(_to_list(shape.wrapped))
        op.SetTools(_to_list(revolved_solid.wrapped))
        op.SetRunParallel(True)
        op.Build()
        if not op.IsDone():
            raise RuntimeError("Boolean cut (revolve) failed")
        result = as_compound(op.Shape())
        if not list(result.solids()):
            raise RuntimeError("Revolve cut produced empty result")
        return result


def _to_list(occ_shape):
    from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape
    lst = TopTools_ListOfShape()
    lst.Append(occ_shape)
    return lst
