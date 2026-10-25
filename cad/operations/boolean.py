"""
cad/operations/boolean.py

bool_op — the one BRepAlgoAPI boolean the heavy ops share, run in a kernel
worker (cad.kernel_pool) so a long cut doesn't hold the GIL on the UI's
thread.
"""

from __future__ import annotations

from cad.kernel_pool import offload


@offload()
def bool_op(kind: str, args: list, tools: list, fuzzy: float = 0.0):
    """kind: "fuse" | "cut" | "common".  args/tools: shapes (build123d or
    TopoDS).  Returns the result as a Compound, or None if OCCT reports the
    operation not done (callers raise their own message)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common
    from OCP.collections import List_TopoDS_Shape
    from cad.solid_ref import as_compound

    op = {"fuse": BRepAlgoAPI_Fuse, "cut": BRepAlgoAPI_Cut,
          "common": BRepAlgoAPI_Common}[kind]()
    lst_a, lst_b = List_TopoDS_Shape(), List_TopoDS_Shape()
    for s in args:
        lst_a.Append(getattr(s, "wrapped", s))
    for s in tools:
        lst_b.Append(getattr(s, "wrapped", s))
    op.SetArguments(lst_a)
    op.SetTools(lst_b)
    op.SetRunParallel(True)
    if fuzzy:
        op.SetFuzzyValue(fuzzy)
    op.Build()
    if not op.IsDone():
        return None
    return as_compound(op.Shape())
