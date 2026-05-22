"""
cad/op_types.py

Re-exports all operation types and the dispatch table.
Implementation lives in:
  cad/op_base.py             — Op base class, _push_result()
  cad/op_extrude.py          — FaceExtrudeOp, CrossBodyCutOp, SketchExtrudeOp, SketchOp, ImportOp
  cad/op_fillet.py           — FaceFilletOp
  cad/op_revolve_thicken.py  — ThickenOp, FaceRevolveOp, SketchRevolveOp
"""

from cad.op_base import Op, _push_result
from cad.op_extrude import (
    FaceExtrudeOp,
    CrossBodyCutOp,
    SketchExtrudeOp,
    SketchOp,
    ImportOp,
)
from cad.op_fillet import FaceFilletOp
from cad.op_revolve_thicken import (
    ThickenOp,
    FaceRevolveOp,
    SketchRevolveOp,
    CrossBodyRevolveCutOp,
)

from typing import Any


def _extrude_or_cut_from_params(operation: str, params: dict) -> Op:
    """Route extrude/cut params to the right op type, applying sign from operation."""
    sign = -1 if operation == "cut" else 1
    if "cut_body_id" in params:
        return CrossBodyCutOp._from_params(params, sign)
    if "from_sketch_id" in params:
        return SketchExtrudeOp._from_params(params, sign)
    return FaceExtrudeOp._from_params(params, sign)


def _revolve_from_params(operation: str, params: dict) -> Op:
    """Route revolve / revolve_cut params to the right op type."""
    if operation == "revolve_cut":
        return CrossBodyRevolveCutOp._from_params(params)
    if "source_body_id" in params and "cut_body_id" not in params:
        return FaceRevolveOp._from_params(params)
    return SketchRevolveOp._from_params(params)


_FROM_PARAMS: dict[str, Any] = {
    "extrude":     _extrude_or_cut_from_params,
    "cut":         _extrude_or_cut_from_params,
    "sketch":      lambda op, p: SketchOp._from_params(p),
    "import":      lambda op, p: ImportOp._from_params(p),
    "thicken":     lambda op, p: ThickenOp._from_params(p),
    "fillet":      lambda op, p: FaceFilletOp._from_params(p),
    "revolve":     _revolve_from_params,
    "revolve_cut": _revolve_from_params,
}

__all__ = [
    "Op", "_push_result",
    "FaceExtrudeOp", "CrossBodyCutOp", "SketchExtrudeOp", "SketchOp", "ImportOp",
    "FaceFilletOp",
    "ThickenOp", "FaceRevolveOp", "SketchRevolveOp", "CrossBodyRevolveCutOp",
    "_FROM_PARAMS", "_extrude_or_cut_from_params",
]
