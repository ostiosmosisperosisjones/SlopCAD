"""
cad/op_transform.py

TransformOp — move a body: translate by (dx, dy, dz), or rotate by an angle
about a world axis or one of the body's own straight / circular edges.
"Copy" leaves the source in place and makes the moved shape a new body.

The transform is cheap (BRepBuilderAPI_Transform, no geometry rebuilt), so
it runs inline.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from cad.op_base import Op, _push_result

if TYPE_CHECKING:
    from cad.history import History


_WORLD_AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def edge_axis(occ_edge):
    """(point, unit direction) of a straight edge's line or a circular
    edge's axis; None for any other curve."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Line, GeomAbs_Circle
    c = BRepAdaptor_Curve(occ_edge)
    if c.GetType() == GeomAbs_Line:
        ax = c.Line().Position()
    elif c.GetType() == GeomAbs_Circle:
        ax = c.Circle().Axis()
    else:
        return None
    p, d = ax.Location(), ax.Direction()
    return (p.X(), p.Y(), p.Z()), (d.X(), d.Y(), d.Z())


def transform_shape(shape, mode: str, dx=0.0, dy=0.0, dz=0.0,
                    angle_deg=0.0, axis_point=(0, 0, 0), axis_dir=(0, 0, 1)):
    """*shape* (build123d) moved; same topology, so face / edge order
    survives."""
    import math
    from OCP.gp import gp_Trsf, gp_Vec, gp_Ax1, gp_Pnt, gp_Dir
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from cad.solid_ref import as_compound
    t = gp_Trsf()
    if mode == "rotate":
        t.SetRotation(gp_Ax1(gp_Pnt(*axis_point), gp_Dir(*axis_dir)),
                      math.radians(angle_deg))
    else:
        t.SetTranslation(gp_Vec(dx, dy, dz))
    mk = BRepBuilderAPI_Transform(shape.wrapped, t, True)
    if not mk.IsDone():
        raise RuntimeError("Transform failed")
    return as_compound(type(shape)(mk.Shape()))


@dataclass
class TransformOp(Op):
    """
    source_body_id : body moved (copy=True → the source stays put)
    mode           : "translate" | "rotate"
    dx, dy, dz     : translation, mm
    angle_deg      : rotation, degrees (right-hand about the axis)
    axis           : "X" | "Y" | "Z" (world, through the origin) | "edge"
    axis_edge_ref  : EdgeRef of the body edge (pre-transform) when axis="edge"
    flip_axis      : reverse the edge axis direction
    copy           : make a moved copy as a new body
    """
    source_body_id: str
    mode:           str   = "translate"
    dx:             float = 0.0
    dy:             float = 0.0
    dz:             float = 0.0
    angle_deg:      float = 0.0
    axis:           str   = "Z"
    axis_edge_ref:  Any   = None
    flip_axis:      bool  = False
    copy:           bool  = False

    def creates_body_from_nothing(self, history: "History", entry_index: int) -> bool:
        return self.copy

    def _axis(self, history, shape):
        if self.axis in _WORLD_AXES:
            p, d = (0.0, 0.0, 0.0), _WORLD_AXES[self.axis]
        else:
            occ = (self.axis_edge_ref.resolve(history, shape)
                   if self.axis_edge_ref is not None else None)
            ax = edge_axis(occ) if occ is not None else None
            if ax is None:
                raise RuntimeError("TransformOp: rotation axis edge not found "
                                   "(or no longer straight / circular)")
            p, d = ax
        if self.flip_axis:
            d = tuple(-v for v in d)
        return p, d

    def apply(self, history, shape):
        p, d = (self._axis(history, shape) if self.mode == "rotate"
                else ((0, 0, 0), (0, 0, 1)))
        return transform_shape(shape, self.mode, self.dx, self.dy, self.dz,
                               self.angle_deg, p, d)

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        if self.copy or shape is None:
            shape = history._shape_for_body_at(self.source_body_id, entry_index)
            if shape is None:
                raise RuntimeError(
                    f"TransformOp: no shape for body '{self.source_body_id}'")
        return self.apply(history, shape)

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        compute, finalize = self._split_commit(viewport, extra_params)
        try:
            shape_after = compute()
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            shape_after = None
            viewport._pending_op_error = str(ex)
        else:
            viewport._pending_op_error = None
        try:
            finalize(shape_after)
        finally:
            viewport._pending_op_error = None
        return shape_after

    def _split_commit(self, viewport: Any, extra_params: dict | None = None):
        shape_before = viewport.workspace.current_shape(self.source_body_id)
        if shape_before is None:
            raise RuntimeError(f"[Transform] No shape for body {self.source_body_id}")
        op_params = self.to_params()
        preserved = None
        if extra_params:
            op_params.update(extra_params)
            preserved = (op_params.pop("_preserved_body_ids", None) or [None])[0]
        history = viewport.history
        count = len(list(shape_before.solids()))

        def compute():
            return self.apply(history, shape_before)

        def finalize(shape_after):
            if not self.copy:
                _push_result(viewport, "transform", op_params, self.source_body_id,
                             None, shape_before, shape_after, count)
                return
            from cad.units import format_op_label as _lbl
            ws = viewport.workspace
            src = ws.bodies.get(self.source_body_id)
            existing = ws.bodies.get(preserved) if preserved else None
            name = existing.name if existing else (
                f"{src.name} copy" if src else ws.next_part_name())
            body = ws.add_body(name, shape_after, body_id=preserved)
            op_params["child_body_ids"] = [body.id]
            entry = viewport.history.push(
                label=_lbl("transform", op_params), operation="transform",
                params=op_params, body_id=body.id, face_ref=None,
                shape_before=None, shape_after=shape_after)
            if shape_after is None:
                entry.error = True
                entry.error_msg = getattr(viewport, '_pending_op_error', None) \
                    or "transform failed"
            body.created_at_entry_id = entry.entry_id
            viewport._rebuild_bodies({body.id})
            viewport.history_changed.emit()

        return compute, finalize

    def to_params(self) -> dict:
        p: dict = {"source_body_id": self.source_body_id, "mode": self.mode}
        if self.mode == "rotate":
            p.update(angle_deg=self.angle_deg, axis=self.axis)
            if self.axis_edge_ref is not None:
                p["axis_edge_ref"] = self.axis_edge_ref.to_dict()
            if self.flip_axis:
                p["flip_axis"] = True
        else:
            p.update(dx=self.dx, dy=self.dy, dz=self.dz)
        if self.copy:
            p["copy"] = True
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "TransformOp":
        from cad.edge_ref import EdgeRef
        ref = params.get("axis_edge_ref")
        return cls(source_body_id=params["source_body_id"],
                   mode=params.get("mode", "translate"),
                   dx=float(params.get("dx", 0.0)),
                   dy=float(params.get("dy", 0.0)),
                   dz=float(params.get("dz", 0.0)),
                   angle_deg=float(params.get("angle_deg", 0.0)),
                   axis=params.get("axis", "Z"),
                   axis_edge_ref=EdgeRef.from_dict(ref) if ref else None,
                   flip_axis=bool(params.get("flip_axis", False)),
                   copy=bool(params.get("copy", False)))
