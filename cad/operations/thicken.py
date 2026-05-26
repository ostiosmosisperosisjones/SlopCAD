from build123d import Compound
from OCP.BRepOffset import BRepOffset_MakeOffset, BRepOffset_Skin
from OCP.GeomAbs import GeomAbs_Arc
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.TopTools import TopTools_ListOfShape

from cad.debug import log, dump_brep, shape_stats


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


def thicken_face_preview(face_occ, thickness: float) -> Compound:
    """Offset slab only — kept for back-compat, not used for body preview."""
    if thickness == 0:
        raise ValueError("thickness must be non-zero")
    return Compound(_offset_solid(face_occ, thickness, 1e-2))


def thicken_body_preview(body_occ, face_occ, thickness: float) -> Compound:
    """Full fuse/cut at coarse tolerance — used for live preview."""
    if thickness == 0:
        raise ValueError("thickness must be non-zero")
    tool_occ = _offset_solid(face_occ, thickness, 1e-2)
    tool_occ = BRepBuilderAPI_Copy(tool_occ).Shape()  # see thicken_face for why
    lst_a = TopTools_ListOfShape(); lst_a.Append(body_occ)
    lst_b = TopTools_ListOfShape(); lst_b.Append(tool_occ)
    if thickness > 0:
        op = BRepAlgoAPI_Fuse()
    else:
        op = BRepAlgoAPI_Cut()
    op.SetArguments(lst_a)
    op.SetTools(lst_b)
    op.SetRunParallel(True)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("Thicken preview failed")
    return Compound(op.Shape())


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
        result = Compound(op.Shape())
        log("Thicken", f"fuse result: {shape_stats(result)}")
        dump_brep("thicken_fuse_result", result)
        if not list(result.solids()):
            # Some configurations (e.g. cylindrical face with both ends rounded
            # by fillet/chamfer) make OCCT's fuse silently return an empty
            # shape even though IsDone() is true. Fail loudly instead of
            # leaving the body looking deleted.
            raise RuntimeError("Thicken produced empty result")
    else:
        op = BRepAlgoAPI_Cut()
        op.SetArguments(lst_a)
        op.SetTools(lst_b)
        op.SetRunParallel(True)
        op.Build()
        if not op.IsDone():
            raise RuntimeError("Thicken cut failed")
        result = Compound(op.Shape())
        log("Thicken", f"cut result : {shape_stats(result)}")
        dump_brep("thicken_cut_result", result)
        if not list(result.solids()):
            raise RuntimeError("Thicken cut produced empty result")

    return result
