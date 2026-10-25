"""
cad/operations/revolve.py
"""

from cad.solid_ref import as_compound
import numpy as np
from build123d import revolve, Plane, Face, Compound, Axis, Vector, Location
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCP.gp import gp_Ax1, gp_Pnt, gp_Dir


def revolve_face_direct(body_shape, face: Face, axis_point: np.ndarray,
                         axis_dir: np.ndarray, angle_deg: float):
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

    revolved = _do_revolve_solid(face, axis_point, axis_dir, angle_deg)

    if body_shape is None:
        return as_compound(revolved)

    return _do_revolve_boolean(body_shape, revolved, angle_deg)


def _do_revolve_solid(face: Face, axis_point: np.ndarray,
                       axis_dir: np.ndarray, angle_deg: float):
    """Produce the raw revolved solid (no boolean)."""
    d = axis_dir / np.linalg.norm(axis_dir)
    ax = Axis(
        origin=Vector(float(axis_point[0]), float(axis_point[1]), float(axis_point[2])),
        direction=Vector(float(d[0]), float(d[1]), float(d[2])),
    )
    return revolve(face, axis=ax, revolution_arc=angle_deg)


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
    from OCP.TopTools import TopTools_ListOfShape
    lst = TopTools_ListOfShape()
    lst.Append(occ_shape)
    return lst
