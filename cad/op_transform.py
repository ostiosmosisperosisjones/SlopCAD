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


def make_trsf(mode: str, dx=0.0, dy=0.0, dz=0.0,
               angle_deg=0.0, axis_point=(0, 0, 0), axis_dir=(0, 0, 1)):
    import math
    from OCP.gp import gp_Trsf, gp_Vec, gp_Ax1, gp_Pnt, gp_Dir
    t = gp_Trsf()
    if mode == "rotate":
        t.SetRotation(gp_Ax1(gp_Pnt(*map(float, axis_point)), gp_Dir(*map(float, axis_dir))),
                      math.radians(angle_deg))
    else:
        t.SetTranslation(gp_Vec(float(dx), float(dy), float(dz)))
    return t


def transform_shape(shape, mode: str, dx=0.0, dy=0.0, dz=0.0,
                    angle_deg=0.0, axis_point=(0, 0, 0), axis_dir=(0, 0, 1)):
    """*shape* (build123d) moved; same topology, so face / edge order
    survives."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from cad.solid_ref import as_compound
    t = make_trsf(mode, dx, dy, dz, angle_deg, axis_point, axis_dir)
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
    axis_sense     : +1/-1, the edge axis's sense (before flip_axis) against
                     the direction the edge's provenance predicts, fixed at
                     the first apply (the geometry it was picked on).  An
                     edge's own direction is OCC's choice and can reverse
                     on a rebuild; this keeps the rotation turning the way
                     it was picked (cad.op_extrude.oriented_tangent).
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
    axis_sense:     int | None = None

    def creates_body_from_nothing(self, history: "History", entry_index: int) -> bool:
        return self.copy

    def _axis(self, history, shape, entry_index=None):
        if self.axis in _WORLD_AXES:
            p, d = (0.0, 0.0, 0.0), _WORLD_AXES[self.axis]
        else:
            if entry_index is None and history is not None:
                entry_index = len(history._entries)
            at = (self.source_body_id, entry_index) if history is not None else None
            occ = (self.axis_edge_ref.resolve(history, shape, at=at)
                   if self.axis_edge_ref is not None else None)
            ax = edge_axis(occ) if occ is not None else None
            if ax is None:
                raise RuntimeError("TransformOp: rotation axis edge not found "
                                   "(or no longer straight / circular)")
            p, d = ax
            d = self._keep_sense(history, d, at, entry_index)
        if self.flip_axis:
            d = tuple(-v for v in d)
        return p, d

    def _keep_sense(self, history, d, at, entry_index):
        import numpy as np
        tn = getattr(getattr(self.axis_edge_ref, "origin", None), "tangent_now", None)
        if tn is None or history is None:
            return d
        try:
            pred = tn(history, at)
        except Exception:
            pred = None
        if pred is None:
            return d
        d = np.asarray(d, dtype=float)
        c = float(d @ pred)
        if self.axis_sense is None:
            self.axis_sense = 1 if c >= 0 else -1
            e = (history._entries[entry_index]
                 if entry_index is not None and entry_index < len(history._entries) else None)
            if e is not None and e.operation == "transform":
                e.params["axis_sense"] = self.axis_sense
        elif c * self.axis_sense < 0:
            d = -d
        return tuple(float(v) for v in d)

    def apply(self, history, shape, entry_index=None):
        p, d = (self._axis(history, shape, entry_index) if self.mode == "rotate"
                else ((0, 0, 0), (0, 0, 1)))
        self._axis_used = (p, d)
        return transform_shape(shape, self.mode, self.dx, self.dy, self.dz,
                               self.angle_deg, p, d)

    def trsf(self, history, entry_index):
        """The gp_Trsf this entry applies (cad.body_motion), or None when
        its edge axis can't be found.  The axis is the one the last apply()
        used: replay runs the transform before anything downstream asks."""
        if self.mode != "rotate":
            return make_trsf("translate", self.dx, self.dy, self.dz)
        ax = getattr(self, "_axis_used", None)
        if ax is None:
            shape = history._shape_for_body_at(self.source_body_id, entry_index)
            try:
                ax = self._axis(history, shape, entry_index) if shape is not None else None
            except Exception:
                ax = None
            if ax is None:
                return None
        return make_trsf("rotate", angle_deg=self.angle_deg,
                         axis_point=ax[0], axis_dir=ax[1])

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        if self.copy or shape is None:
            shape = history._shape_for_body_at(self.source_body_id, entry_index)
            if shape is None:
                raise RuntimeError(
                    f"TransformOp: no shape for body '{self.source_body_id}'")
        return self.apply(history, shape, entry_index)

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
            if self.axis_sense is not None:      # fixed by compute's apply
                op_params["axis_sense"] = int(self.axis_sense)
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
            if self.axis_sense is not None:
                p["axis_sense"] = int(self.axis_sense)
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
                   copy=bool(params.get("copy", False)),
                   axis_sense=params.get("axis_sense"))
