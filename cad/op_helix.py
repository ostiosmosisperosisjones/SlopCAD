"""
cad/op_helix.py

HelixOp — a helix curve: construction geometry (no body changes), the path
a sweep takes for threads and springs.  Wraps a cylindrical face (its axis
and radius; the face's full length unless a height is given) or a sketch
circle (its centre / sketch normal and radius).  Like an offset plane it
re-resolves on replay, so it follows edits to the face or circle.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from cad.op_base import Op

if TYPE_CHECKING:
    from cad.history import History


def cylinder_axis(occ_face):
    """(base point, unit axis, x_dir, radius, length) of a cylindrical face:
    the base is where the face starts along the axis.  None for any other
    surface."""
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    s = BRepAdaptor_Surface(occ_face, True)
    if s.GetType() != GeomAbs_Cylinder:
        return None
    cyl = s.Cylinder()
    ax = cyl.Position()
    p, d, x = ax.Location(), ax.Direction(), ax.XDirection()
    p, d, x = (np.array([v.X(), v.Y(), v.Z()]) for v in (p, d, x))
    v0, v1 = s.FirstVParameter(), s.LastVParameter()
    return p + d * v0, d, x, cyl.Radius(), v1 - v0


@dataclass
class HelixOp(Op):
    """
    body_id, face_ref : the cylindrical face wrapped (AnyFaceRef), or
    circle_sketch_id, circle_uid : a sketch circle (axis = sketch normal)
    height     : along the axis, mm; 0 = the face's full length
    turns      : revolutions over the height (used when pitch is 0)
    pitch      : rise per turn, mm; > 0 → it drives and turns follow
    left_hand  : left-handed winding
    flip       : start from the face's other end / run against the normal
    """
    DATUM = True              # history: no shape change, re-resolve only

    body_id:          str | None = None
    face_ref:         Any        = None
    circle_sketch_id: str        = ""
    circle_uid:       str        = ""
    height:           float      = 0.0
    turns:            float      = 5.0
    pitch:            float      = 0.0
    left_hand:        bool       = False
    flip:             bool       = False

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def axis(self, history: "History", entry_index: int):
        """(start centre, unit axis, x_dir, radius, height) in world space."""
        import numpy as np
        if self.face_ref is not None:
            shape = history._shape_for_body_at(self.body_id, entry_index)
            if shape is None:
                raise RuntimeError("Helix: the face's body has no shape here")
            _i, face = self.face_ref.find_in(shape)
            if face is None:
                raise RuntimeError("Helix: the cylindrical face is gone")
            ax = cylinder_axis(face.wrapped)
            if ax is None:
                raise RuntimeError("Helix: the face is no longer cylindrical")
            base, d, x, r, length = ax
            h = self.height if self.height > 0 else length
            if self.flip:
                base, d = base + d * length, -d
            return base, d, x, r, h
        c, n, x, r = self._circle_axis(history, entry_index)
        if self.height <= 0:
            raise RuntimeError("Helix: a circle helix needs a height")
        return c, n * (-1.0 if self.flip else 1.0), x, r, self.height

    def _circle_axis(self, history, entry_index):
        import numpy as np
        from cad.history import _replay_sketch_entry
        from cad.sketch import ArcEntity
        idx = history.id_to_index(self.circle_sketch_id)
        if idx is None or idx >= entry_index:
            raise RuntimeError("Helix: circle sketch not found before it")
        rec = history._entries[idx]
        se = rec.params.get("sketch_entry")
        if se is None or rec.error:
            raise RuntimeError("Helix: circle sketch is in an error state")
        if se.plane_source is not None:
            ok, err = _replay_sketch_entry(se, history, before_index=entry_index)
            if not ok:
                raise RuntimeError(f"Helix: sketch reprojection failed: {err}")
        ent = next((e for e in se.entities
                    if getattr(e, "uid", None) == self.circle_uid), None)
        if not isinstance(ent, ArcEntity):
            raise RuntimeError("Helix: its circle is gone from the sketch")
        o, xa, ya, n = (np.asarray(v, float) for v in (
            se.plane_origin, se.plane_x_axis, se.plane_y_axis, se.plane_normal))
        return o + ent.center[0] * xa + ent.center[1] * ya, n, xa, ent.radius

    def pitch_for(self, height: float) -> float:
        if self.pitch > 0:
            return self.pitch
        if self.turns <= 0:
            raise RuntimeError("Helix needs a positive number of turns")
        return height / self.turns

    def edge(self, history: "History", entry_index: int):
        """(build123d helix Edge, unit axis, turns)."""
        from cad.operations.coil import helix_edge
        c, d, x, r, h = self.axis(history, entry_index)
        pitch = self.pitch_for(h)
        return (helix_edge(tuple(c), tuple(d), tuple(x), r, h, pitch,
                           self.left_hand), tuple(d), h / pitch)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        self.edge(history, entry_index)      # surface errors; no shape change
        return shape

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        from cad.units import format_op_label as _lbl
        params = self.to_params()
        if extra_params:
            params.update(extra_params)
        h = viewport.history
        entry = h.push(label=_lbl("helix", params), operation="helix",
                       params=params, body_id=None, face_ref=None,
                       shape_before=None, shape_after=None)
        try:
            self.edge(h, h.cursor)
        except Exception as ex:
            entry.error, entry.error_msg = True, str(ex)
        if h.is_mid_history:
            viewport._post_push_cascade(None)
        viewport.history_changed.emit()
        viewport.update()
        return None

    def to_params(self) -> dict:
        p: dict = {"turns": self.turns}
        if self.face_ref is not None:
            r = self.face_ref
            p.update(body_id=self.body_id, face_ref={
                "centroid": list(r.centroid), "area": r.area,
                "category": r.category,
                "normal": list(r.normal) if r.normal is not None else None})
        else:
            p.update(circle_sketch_id=self.circle_sketch_id,
                     circle_uid=self.circle_uid)
        if self.height > 0:
            p["height"] = self.height
        if self.pitch > 0:
            p["pitch"] = self.pitch
        if self.left_hand:
            p["left_hand"] = True
        if self.flip:
            p["flip"] = True
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "HelixOp":
        from cad.face_ref import AnyFaceRef
        fr = params.get("face_ref")
        ref = None if fr is None else AnyFaceRef(
            centroid=tuple(fr["centroid"]), area=float(fr["area"]),
            category=fr.get("category", ""),
            normal=tuple(fr["normal"]) if fr.get("normal") else None)
        return cls(body_id=params.get("body_id"), face_ref=ref,
                   circle_sketch_id=params.get("circle_sketch_id", ""),
                   circle_uid=params.get("circle_uid", ""),
                   height=float(params.get("height", 0.0)),
                   turns=float(params.get("turns", 5.0)),
                   pitch=float(params.get("pitch", 0.0)),
                   left_hand=bool(params.get("left_hand", False)),
                   flip=bool(params.get("flip", False)))
