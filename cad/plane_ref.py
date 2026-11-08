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
from cad import resolve_ledger


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

    anchor : (u, v) of the plane origin, in the face plane's axes, from the
             body origin's projection onto the face (the world origin moved
             by the TransformOps on the body before the resolve position).
             Baked once — when the face is picked (its centroid then), or on
             the first resolve of a file saved before anchors — and fixed
             after.  The frame used to be Plane(face), whose origin is the
             face CENTROID: any edit to the face's outline (a hole on it
             growing, a pocket from another loop) dragged every sketch on the
             face across it.  The anchor follows the face along its normal
             and with the body's motion, never with its outline.
    """

    def __init__(self, body_id: str, face_ref, origin=None, anchor=None):
        self.body_id  = body_id
        self.face_ref = face_ref          # cad.face_ref.FaceRef
        self.origin   = origin            # cad.face_origin.FaceOrigin | ChamferFaceOrigin | None
        self.anchor   = None if anchor is None else tuple(map(float, anchor))

    def _body_origin(self, history, before_index):
        """The world origin carried by the body's TransformOps before
        *before_index* (cad.body_motion)."""
        import numpy as np
        from cad import body_motion as BM
        if history is None:
            return np.zeros(3)
        _, T = BM.motion(history, self.body_id, before_index, -1)
        return BM.pt(T, (0.0, 0.0, 0.0))

    def plane_for(self, history, face, before_index):
        """The sketch plane on *face* (a build123d Face of this body as of
        before_index): Plane(face)'s axes, origin at the anchor (baked here
        from the face's centroid if it has none yet)."""
        import numpy as np
        from build123d import Plane
        pl = Plane(face)
        v = lambda d: np.array([d.X, d.Y, d.Z], dtype=float)
        c, x, y, n = v(pl.origin), v(pl.x_dir), v(pl.y_dir), v(pl.z_dir)
        b = self._body_origin(history, before_index)
        base = b + float((c - b) @ n) * n
        if self.anchor is None:
            # A file saved before anchors: its points are relative to the
            # plane origin it was saved with (seed_origin, set by the sketch's
            # replay) — not necessarily today's centroid.
            seed = getattr(self, "seed_origin", None)
            d = (np.asarray(seed, dtype=float) if seed is not None else c) - base
            self.anchor = (float(d @ x), float(d @ y))
            self.seed_origin = None
        o = base + self.anchor[0] * x + self.anchor[1] * y
        return Plane(origin=tuple(o.tolist()), x_dir=tuple(x.tolist()), z_dir=tuple(n.tolist()))

    @classmethod
    def for_face(cls, history, body_id: str, face, before_index: int):
        """Source for a build123d Face of body_id as of before_index, with
        provenance when an extruded sketch line made the face.  None if the
        face isn't planar."""
        from cad.face_ref import FaceRef
        face_ref = FaceRef.from_b3d_face(face)
        if face_ref is None:
            return None
        origin = None
        if history is not None:
            from cad.face_origin import any_origin_for_face
            try:
                origin = any_origin_for_face(history, body_id, face.wrapped, before_index)
            except Exception as ex:
                print(f"[Sketch] face provenance skipped: {ex}")
        src = cls(body_id, face_ref, origin)
        if history is not None:
            try:
                src.plane_for(history, face, before_index)     # bakes the anchor
            except Exception as ex:
                print(f"[Sketch] face plane anchor skipped: {ex}")
        return src

    def resolve(self, history, before_index: int):
        from build123d import Plane
        shape = history._shape_for_body_at(self.body_id, before_index)
        if shape is None:
            raise RuntimeError(
                f"FacePlaneSource: no shape for body '{self.body_id}' "
                f"before history index {before_index}")
        face = None
        if self.origin is not None:
            # Provenance first: follows a wall that rotated or resized,
            # which the geometric FaceRef can't.
            _, face = self.origin.find_in(history, shape, (self.body_id, before_index))
            if face is not None:
                resolve_ledger.note(resolve_ledger.EXACT, "FacePlaneSource")
        if face is None:
            face_idx, face = self.face_ref.find_in(shape)
            if face is not None and self.origin is None:
                self._adopt_origin(history, face, before_index)
        if face is None and self.origin is None and getattr(self, "last_plane", None):
            # The FaceRef is older than the sketch's last plane (an upstream
            # fix can even flip the face's recorded normal): the one face
            # still exactly on that plane is it.
            on = [f for f in shape.faces() if self._on_last_plane(f)]
            if len(on) == 1:
                face = on[0]
                resolve_ledger.note(resolve_ledger.FUZZY, "FacePlaneSource",
                                    "face recognised by the sketch's saved plane")
                self._adopt_origin(history, face, before_index)
        if face is None and self.face_ref is not None:
            face_idx, face = self.face_ref.find_coplanar_in(shape)
        if face is None:
            # Saved before revolve caps got an origin (or the origin names an
            # entry that is gone): recognise the cap by its rotation
            # invariants and keep the origin from now on.
            from cad.face_origin import revolve_origin_from_ref
            origin = revolve_origin_from_ref(history, self.body_id,
                                             self.face_ref, before_index)
            if origin is not None:
                _, face = origin.find_in(history, shape, (self.body_id, before_index))
                if face is not None:
                    self.origin = origin
                    resolve_ledger.note(resolve_ledger.FUZZY, "FacePlaneSource",
                                        "revolve cap recognised by shape")
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
        return self.plane_for(history, face, before_index)

    def _on_last_plane(self, face, tol: float = 1e-4) -> bool:
        """face is where the sketch last stood: its centroid is the saved
        plane origin and its normal the saved plane normal (either sense).
        A FaceRef can be older than that (taken before an upstream edit and
        matched as "moved" ever since); the saved plane is the last place
        the sketch was actually resolved to."""
        lp = getattr(self, "last_plane", None)
        if lp is None:
            return False
        import numpy as np
        from cad.face_ref import _occ_face_normal, _occ_face_props
        n = _occ_face_normal(face.wrapped)
        if n is None:
            return False
        c, _ = _occ_face_props(face.wrapped)
        o, ln = (np.asarray(v, dtype=float) for v in lp)
        return (abs(abs(float(n @ ln)) - 1.0) < 1e-9
                and float(np.linalg.norm(np.asarray(c, dtype=float) - o)) <= tol)

    def _adopt_origin(self, history, face, before_index):
        """A sketch saved before its face had an origin kind (a revolve's
        flat side, an imported face, …): while the face is still exactly the
        one it was picked on, bake the origin now, so later edits follow it
        instead of drifting the FaceRef match out of range.  Never from a
        moved match: that would freeze a guess in."""
        try:
            if not (self.face_ref.is_exactly(face) or self._on_last_plane(face)):
                return
            from cad.face_origin import any_origin_for_face
            origin = any_origin_for_face(history, self.body_id, face.wrapped, before_index)
        except Exception as ex:
            print(f"[Sketch] face provenance upgrade skipped: {ex}")
            return
        if origin is not None:
            self.origin = origin
            print(f"[Sketch] face plane got provenance: {type(origin).__name__}")

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
