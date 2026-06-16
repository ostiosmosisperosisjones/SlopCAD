"""
cad/plane_ref.py

Abstract sketch plane sources.  Each source resolves to a live build123d Plane
given the history state at a specific replay position.

Classes
-------
SketchPlaneSource  — abstract base
FacePlaneSource    — plane derived from a body face  (current default)
WorldPlaneSource   — one of the three canonical world planes (XY / XZ / YZ)
OffsetPlaneSource  — plane offset along the normal of any other source
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class SketchPlaneSource(ABC):
    """
    Abstract reference to a sketch plane.

    resolve(history, before_index) → build123d.Plane
      history      : History instance — used for shape lookups
      before_index : only entries with index < before_index are considered,
                     matching the replay cursor position
    """

    @abstractmethod
    def resolve(self, history, before_index: int):
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialisation hint for future save/load."""
        ...


# ---------------------------------------------------------------------------
# FacePlaneSource
# ---------------------------------------------------------------------------

class FacePlaneSource(SketchPlaneSource):
    """
    Plane derived from a named body face, located via FaceRef each time
    it is resolved so it follows the face as upstream operations change.
    """

    def __init__(self, body_id: str, face_ref):
        self.body_id  = body_id
        self.face_ref = face_ref          # cad.face_ref.FaceRef

    def resolve(self, history, before_index: int):
        from build123d import Plane
        shape = history._shape_for_body_at(self.body_id, before_index)
        if shape is None:
            raise RuntimeError(
                f"FacePlaneSource: no shape for body '{self.body_id}' "
                f"before history index {before_index}")
        face_idx, face = self.face_ref.find_in(shape)
        if face is None:
            detail = ""
            if self.face_ref is not None:
                try:
                    detail = " — " + self.face_ref.explain_no_match(shape)
                except Exception:
                    pass
            raise RuntimeError(
                f"FacePlaneSource: could not relocate face in body "
                f"'{self.body_id}'{detail}")
        return Plane(face)   # face is a build123d Face object

    def to_dict(self) -> dict:
        return {"type": "face", "body_id": self.body_id}


# ---------------------------------------------------------------------------
# WorldPlaneSource
# ---------------------------------------------------------------------------

class WorldPlaneSource(SketchPlaneSource):
    """
    One of the three canonical world planes.
    axis must be 'XY', 'XZ', or 'YZ'.
    Body-agnostic — resolve() ignores history entirely.
    """

    _VALID = ("XY", "XZ", "YZ")

    def __init__(self, axis: str):
        if axis not in self._VALID:
            raise ValueError(f"WorldPlaneSource: axis must be one of {self._VALID}, "
                             f"got {axis!r}")
        self.axis = axis

    def resolve(self, history, before_index: int):
        from build123d import Plane
        return {"XY": Plane.XY, "XZ": Plane.XZ, "YZ": Plane.YZ}[self.axis]

    def to_dict(self) -> dict:
        return {"type": "world", "axis": self.axis}


# ---------------------------------------------------------------------------
# OffsetPlaneSource
# ---------------------------------------------------------------------------

class OffsetPlaneSource(SketchPlaneSource):
    """
    A plane offset from any other SketchPlaneSource by a fixed distance
    along its normal.  Composable: offset of offset works fine.
    """

    def __init__(self, parent: SketchPlaneSource, distance: float):
        self.parent   = parent
        self.distance = float(distance)

    def resolve(self, history, before_index: int):
        import numpy as np
        from build123d import Plane

        base = self.parent.resolve(history, before_index)
        n = np.array([base.z_dir.X, base.z_dir.Y, base.z_dir.Z])
        o = np.array([base.origin.X, base.origin.Y, base.origin.Z])
        new_o = o + n * self.distance
        return Plane(
            origin = tuple(new_o.tolist()),
            x_dir  = (base.x_dir.X, base.x_dir.Y, base.x_dir.Z),
            z_dir  = (base.z_dir.X, base.z_dir.Y, base.z_dir.Z),
        )

    def to_dict(self) -> dict:
        return {
            "type":     "offset",
            "distance": self.distance,
            "parent":   self.parent.to_dict(),
        }
