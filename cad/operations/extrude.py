"""
cad/operations/extrude.py
"""

import numpy as np
from build123d import extrude, Plane, Face, Compound
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCP.BOPAlgo import BOPAlgo_GlueEnum


def extrude_face(shape, face_idx: int, distance: float,
                 direction: np.ndarray | None = None,
                 start_offset: float = 0.0,
                 end_offset: float = 0.0,
                 draft_angle: float = 0.0):
    """
    Extrude a planar face of *shape* by *distance* along its normal
    (or along *direction* if given).

    Positive distance  → extrude outward, fuse  (adds material).
    Negative distance  → extrude inward,  cut   (removes material).

    Returns the resulting build123d Compound/Solid, or raises on failure.
    """
    faces = list(shape.faces())
    if face_idx < 0 or face_idx >= len(faces):
        raise IndexError(
            f"face_idx {face_idx} out of range (shape has {len(faces)} faces)"
        )

    face: Face = faces[face_idx]
    Plane(face)  # raises if not planar

    if distance == 0:
        return shape

    return _do_extrude_boolean(shape, face, distance, direction=direction,
                               start_offset=start_offset, end_offset=end_offset,
                               draft_angle=draft_angle)


def extrude_face_direct(body_shape, face: Face, distance: float,
                        direction: np.ndarray | None = None,
                        start_offset: float = 0.0,
                        end_offset: float = 0.0,
                        draft_angle: float = 0.0):
    """
    Extrude a face object directly (not looked up by index).

    Used for sketch-profile extrusion where the face is built from a
    SketchEntry wire rather than being an existing face of body_shape.

    Positive distance  → fuse (adds material).
    Negative distance  → cut  (removes material).

    If body_shape is None (first operation on a new body, e.g. from a
    world-plane sketch), the extrusion is returned as-is with no boolean.

    Returns the resulting build123d Compound/Solid, or raises on failure.
    """
    if distance == 0:
        return body_shape

    Plane(face)  # raises if not planar

    if body_shape is None:
        extruded = _do_extrude_solid(face, distance, direction,
                                     start_offset=start_offset, end_offset=end_offset,
                                     draft_angle=draft_angle)
        return Compound(extruded.wrapped)

    return _do_extrude_boolean(body_shape, face, distance, glue=False,
                               direction=direction,
                               start_offset=start_offset, end_offset=end_offset,
                               draft_angle=draft_angle)


def _do_extrude_solid(face: Face, distance: float,
                      direction: np.ndarray | None = None,
                      start_offset: float = 0.0,
                      end_offset: float = 0.0,
                      draft_angle: float = 0.0):
    """Produce the raw extruded solid (no boolean).

    start_offset: translate the face along the extrude direction by this amount
                  before extruding, so the solid starts offset from the face.
    end_offset:   shorten the extrude distance by this amount.
    draft_angle:  taper angle in degrees (positive = walls flare outward).
    """
    from build123d import extrude as b3d_extrude, Vector
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Vec

    sign = -1.0 if distance < 0 else 1.0

    # Resolve the unit direction vector
    if direction is not None:
        d = direction / np.linalg.norm(direction)
    else:
        from build123d import Plane
        pl = Plane(face)
        n = pl.z_dir
        d = np.array([n.X, n.Y, n.Z], dtype=float)
        # For negative distance (cut), the extrude goes inward (opposite normal)
        d = d * sign

    effective_dist = abs(distance) - end_offset

    if start_offset != 0.0:
        # Translate face along d by start_offset
        tx = gp_Trsf()
        tx.SetTranslation(gp_Vec(
            float(d[0] * start_offset),
            float(d[1] * start_offset),
            float(d[2] * start_offset),
        ))
        moved = BRepBuilderAPI_Transform(face.wrapped, tx, True).Shape()
        from build123d import Face as B3dFace
        face = B3dFace(moved)

    vec = Vector(float(d[0] * effective_dist),
                 float(d[1] * effective_dist),
                 float(d[2] * effective_dist))
    return b3d_extrude(face, amount=effective_dist, dir=vec, taper=float(draft_angle))


def _do_extrude_boolean(shape, face: Face, distance: float,
                        glue: bool = True,
                        direction: np.ndarray | None = None,
                        start_offset: float = 0.0,
                        end_offset: float = 0.0,
                        draft_angle: float = 0.0):
    """
    Shared boolean logic for both extrude entry points.

    glue=True  — safe only when the tool's base face is the exact same OCCT
                 face object as an existing face on shape (i.e. extrude_face).
                 Tells OCCT shapes share a face → skips full intersection scan.
    glue=False — required for sketch-profile extrudes where the face is
                 reconstructed geometry, not a face already on shape.
    """
    if distance > 0:
        extruded = _do_extrude_solid(face, distance, direction,
                                     start_offset=start_offset, end_offset=end_offset,
                                     draft_angle=draft_angle)
        op = BRepAlgoAPI_Fuse()
        op.SetArguments(_to_list(shape.wrapped))
        op.SetTools(_to_list(extruded.wrapped))
        if glue and start_offset == 0.0 and draft_angle == 0.0:
            # Glue is only valid when the tool base face coincides with a shape face.
            # With a start_offset the tool is translated away, so glue must be off.
            op.SetGlue(BOPAlgo_GlueEnum.BOPAlgo_GlueShift)
        op.SetRunParallel(True)
        op.Build()
        if not op.IsDone():
            raise RuntimeError("Boolean fuse failed")
        return Compound(op.Shape())

    # Cut path — add a small epsilon so the tool face never lands exactly
    # coincident with an existing face (OCCT drops geometry when that happens).
    cut_dist = distance - 0.01  # more negative = 0.01mm deeper
    extruded = _do_extrude_solid(face, cut_dist, direction,
                                 start_offset=start_offset, end_offset=end_offset,
                                 draft_angle=draft_angle)
    op = BRepAlgoAPI_Cut()
    op.SetArguments(_to_list(shape.wrapped))
    op.SetTools(_to_list(extruded.wrapped))
    op.SetRunParallel(True)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("Boolean cut failed")
    result = Compound(op.Shape())
    if not list(result.solids()):
        # Tool went the wrong direction — retry flipped.
        if direction is not None:
            flip_dir = -direction
        else:
            _pl = Plane(face)
            _n  = np.array([_pl.z_dir.X, _pl.z_dir.Y, _pl.z_dir.Z])
            flip_dir = _n
        extruded_flip = _do_extrude_solid(face, abs(cut_dist), flip_dir,
                                          start_offset=start_offset, end_offset=end_offset,
                                          draft_angle=draft_angle)
        op2 = BRepAlgoAPI_Cut()
        op2.SetArguments(_to_list(shape.wrapped))
        op2.SetTools(_to_list(extruded_flip.wrapped))
        op2.SetRunParallel(True)
        op2.Build()
        if op2.IsDone():
            result2 = Compound(op2.Shape())
            if list(result2.solids()):
                print("[Cut] Flipped direction succeeded.")
                return result2
        raise RuntimeError("Cut produced empty result in both directions")
    return result


def _to_list(occ_shape):
    from OCP.TopTools import TopTools_ListOfShape
    lst = TopTools_ListOfShape()
    lst.Append(occ_shape)
    return lst
