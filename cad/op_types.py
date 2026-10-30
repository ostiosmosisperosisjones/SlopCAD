"""
cad/op_types.py

Re-exports all operation types and the dispatch table.
Implementation lives in:
  cad/op_base.py             — Op base class, _push_result()
  cad/op_extrude.py          — FaceExtrudeOp, CrossBodyCutOp, SketchExtrudeOp, SketchOp, ImportOp
  cad/op_fillet.py           — FaceFilletOp
  cad/op_revolve_thicken.py  — ThickenOp, FaceRevolveOp, SketchRevolveOp
  cad/op_shell.py            — ShellOp
  cad/op_sweep.py            — SketchSweepOp
  cad/op_coil.py             — CoilOp
  cad/op_transform.py        — TransformOp
  cad/op_helix.py            — HelixOp
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
from cad.op_offset_plane import OffsetPlaneOp
from cad.op_loft import SketchLoftOp
from cad.op_chamfer import ChamferOp
from cad.op_boolean import BooleanOp
from cad.op_shell import ShellOp
from cad.op_sweep import SketchSweepOp
from cad.op_coil import CoilOp
from cad.op_transform import TransformOp
from cad.op_helix import HelixOp

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
    "extrude":      _extrude_or_cut_from_params,
    "cut":          _extrude_or_cut_from_params,
    "sketch":       lambda op, p: SketchOp._from_params(p),
    "import":       lambda op, p: ImportOp._from_params(p),
    "thicken":      lambda op, p: ThickenOp._from_params(p),
    "fillet":       lambda op, p: FaceFilletOp._from_params(p),
    "revolve":      _revolve_from_params,
    "revolve_cut":  _revolve_from_params,
    "offset_plane": lambda op, p: OffsetPlaneOp._from_params(p),
    "loft":         lambda op, p: SketchLoftOp._from_params(p),
    "loft_cut":     lambda op, p: SketchLoftOp._from_params(p),
    "chamfer":      lambda op, p: ChamferOp._from_params(p),
    "union":        lambda op, p: BooleanOp._from_params(p),
    "subtract":     lambda op, p: BooleanOp._from_params(p),
    "intersect":    lambda op, p: BooleanOp._from_params(p),
    "shell":        lambda op, p: ShellOp._from_params(p),
    "sweep":        lambda op, p: SketchSweepOp._from_params(p),
    "sweep_cut":    lambda op, p: SketchSweepOp._from_params(p),
    "coil":         lambda op, p: CoilOp._from_params(p),
    "coil_cut":     lambda op, p: CoilOp._from_params(p),
    "transform":    lambda op, p: TransformOp._from_params(p),
    "helix":        lambda op, p: HelixOp._from_params(p),
}

__all__ = [
    "Op", "_push_result",
    "FaceExtrudeOp", "CrossBodyCutOp", "SketchExtrudeOp", "SketchOp", "ImportOp",
    "FaceFilletOp",
    "ThickenOp", "FaceRevolveOp", "SketchRevolveOp", "CrossBodyRevolveCutOp",
    "OffsetPlaneOp",
    "SketchLoftOp",
    "ChamferOp",
    "BooleanOp",
    "ShellOp",
    "SketchSweepOp",
    "CoilOp",
    "TransformOp",
    "HelixOp",
    "_FROM_PARAMS", "_extrude_or_cut_from_params",
]
