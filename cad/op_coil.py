"""
cad/op_coil.py

CoilOp — a helical coil around a sketch circle: the circle's centre and its
sketch's normal are the axis, its diameter the coil's, so editing the
sketch moves / resizes the coil.  New body, or fused into / cut from
another; body handling is SketchLoftOp's (no profiles — the section is
generated).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cad.op_loft import SketchLoftOp

if TYPE_CHECKING:
    from cad.history import History


@dataclass
class CoilOp(SketchLoftOp):
    """
    circle_sketch_id : sketch entry holding the circle
    circle_uid       : that circle's entity uid
    height           : along the axis (mm; flip → the other side)
    turns            : revolutions over the height
    section          : "circle" | "square"
    section_size     : section diameter / side (mm)
    left_hand        : left-handed winding
    flip             : grow against the sketch normal
    """
    _OP           = "coil"
    _MIN_PROFILES = 0

    circle_sketch_id: str   = ""
    circle_uid:       str   = ""
    height:           float = 20.0
    turns:            float = 5.0
    section:          str   = "circle"
    section_size:     float = 1.0
    left_hand:        bool  = False
    flip:             bool  = False

    @property
    def from_sketch_ids(self) -> list[str]:
        return [self.circle_sketch_id]

    def _task(self, faces: list, target=None, mode=None):
        from cad.operations.coil import coil_task
        c, n, x, r = self._axis
        return coil_task(c, n, x, r, self.height, self.turns, self.section,
                         self.section_size, self.left_hand, target, mode)

    def _resolve_profile_faces(self, history: "History", entry_index: int) -> list:
        # No profile faces: resolve the axis instead (every execute / commit
        # path calls this before _task).
        self._axis = self._resolve_axis(history, entry_index)
        return []

    def _resolve_axis(self, history: "History", entry_index: int):
        """(centre, normal, x_dir, radius) in world space."""
        import numpy as np
        from cad.history import _replay_sketch_entry
        from cad.sketch import ArcEntity
        idx = history.id_to_index(self.circle_sketch_id)
        if idx is None:
            raise RuntimeError("CoilOp: circle sketch not found")
        if idx >= entry_index:
            raise RuntimeError("CoilOp: circle sketch is after this op — "
                               "invalid reorder")
        rec = history._entries[idx]
        se = rec.params.get("sketch_entry")
        if se is None or rec.error:
            raise RuntimeError("CoilOp: circle sketch is in an error state")
        if se.plane_source is not None:
            ok, err = _replay_sketch_entry(se, history, before_index=entry_index)
            if not ok:
                raise RuntimeError(f"CoilOp: sketch reprojection failed: {err}")
        ent = next((e for e in se.entities
                    if getattr(e, "uid", None) == self.circle_uid), None)
        if not isinstance(ent, ArcEntity):
            raise RuntimeError("CoilOp: the coil's circle is gone from its sketch")
        o, xa, ya = (np.asarray(v, float) for v in
                     (se.plane_origin, se.plane_x_axis, se.plane_y_axis))
        n = np.asarray(se.plane_normal, float) * (-1.0 if self.flip else 1.0)
        c = o + ent.center[0] * xa + ent.center[1] * ya
        return tuple(c.tolist()), tuple(n.tolist()), tuple(xa.tolist()), ent.radius

    def to_params(self) -> dict:
        p = super().to_params()
        p.pop("from_profiles", None)
        p.update(circle_sketch_id=self.circle_sketch_id,
                 circle_uid=self.circle_uid, height=self.height,
                 turns=self.turns, section=self.section,
                 section_size=self.section_size)
        if self.left_hand:
            p["left_hand"] = True
        if self.flip:
            p["flip"] = True
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "CoilOp":
        return cls(
            merge_body_id    = params.get("merge_body_id"),
            cut_body_id      = params.get("cut_body_id"),
            force_new_body   = bool(params.get("force_new_body", False)),
            merged_from      = params.get("merged_from"),
            circle_sketch_id = params.get("circle_sketch_id", ""),
            circle_uid       = params.get("circle_uid", ""),
            height           = float(params.get("height", 20.0)),
            turns            = float(params.get("turns", 5.0)),
            section          = params.get("section", "circle"),
            section_size     = float(params.get("section_size", 1.0)),
            left_hand        = bool(params.get("left_hand", False)),
            flip             = bool(params.get("flip", False)),
        )
