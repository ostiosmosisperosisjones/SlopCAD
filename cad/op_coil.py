"""
cad/op_coil.py

CoilOp — a helical coil around a sketch circle or a cylindrical body face.
A circle's centre and its sketch's normal are the axis, its diameter the
coil's, so editing the sketch moves / resizes the coil; a cylinder face
gives its own axis and radius (re-found by AnyFaceRef on replay), so the
coil follows the shaft or bore it wraps.  New body, or fused into / cut from
another; body handling is SketchLoftOp's (no profiles — the section is
generated).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from cad.op_loft import SketchLoftOp

if TYPE_CHECKING:
    from cad.history import History


@dataclass
class CoilOp(SketchLoftOp):
    """
    circle_sketch_id : sketch entry holding the circle, or
    circle_uid       : that circle's entity uid
    body_id, face_ref: a cylindrical body face (AnyFaceRef) — axis + radius
                       from the cylinder, starting at the face's end (the
                       other end with flip); used instead of the circle
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
    body_id:          str | None = None
    face_ref:         Any   = None
    height:           float = 20.0
    turns:            float = 5.0
    section:          str   = "circle"
    section_size:     float = 1.0
    left_hand:        bool  = False
    flip:             bool  = False

    @property
    def from_sketch_ids(self) -> list[str]:
        return [self.circle_sketch_id] if self.circle_sketch_id else []

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
        if self.face_ref is not None:
            return self._face_axis(history, entry_index)
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

    def _face_axis(self, history: "History", entry_index: int):
        from cad.op_helix import cylinder_axis
        shape = history._shape_for_body_at(self.body_id, entry_index)
        if shape is None:
            raise RuntimeError("CoilOp: the face's body has no shape here")
        _i, face = self.face_ref.find_in(shape)
        if face is None:
            raise RuntimeError("CoilOp: the cylindrical face is gone")
        ax = cylinder_axis(face.wrapped)
        if ax is None:
            raise RuntimeError("CoilOp: the face is no longer cylindrical")
        base, d, x, r, length = ax
        if self.flip:
            base, d = base + d * length, -d
        return tuple(base.tolist()), tuple(d.tolist()), tuple(x.tolist()), r

    def to_params(self) -> dict:
        from cad.profile_source import face_ref_to_dict
        p = super().to_params()
        p.pop("from_profiles", None)
        if self.face_ref is not None:
            p.update(body_id=self.body_id, face_ref=face_ref_to_dict(self.face_ref))
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
        from cad.profile_source import face_ref_from_dict
        return cls(
            body_id          = params.get("body_id"),
            face_ref         = face_ref_from_dict(params.get("face_ref")),
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
