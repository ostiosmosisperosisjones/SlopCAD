"""
cad/profile_source.py

ProfileSource — where an op's profile face(s) come from.  One type covers
both inputs a profile-consuming op can take:

  SketchProfile    closed loop(s) of a sketch entry, re-built and re-matched
                   (by region boundary, then centroid) at the consuming op's
                   history position
  BodyFaceProfile  one face of a body, re-found by FaceRef (planar) or
                   AnyFaceRef (curved) on that body's shape at the op's
                   history position

Both resolve to plain build123d Faces, so an op never needs to know which
kind it was handed.  Extrude, cut, revolve, revolve-cut, loft, sweep (and
helix / coil through their own axis inputs) all go through here.

Serialized form (JSON-safe, stored in entry.params["profiles"]):

  {"kind": "sketch", "sketch_id": "<entry id>", "face_indices": [..] | null,
   "face_centroids": [[u, v], ..] | null, "face_boundaries": [[uid, ..], ..] | null}
  {"kind": "face", "body_id": "<body id>", "face_idx": 3,
   "ref": {"type": "plane", ...FaceRef...} | {"type": "any", ...AnyFaceRef...} | null}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cad.history import History


class ProfileSource:
    """Base class.  Subclasses: SketchProfile, BodyFaceProfile."""

    kind = ""

    def resolve(self, history: "History", entry_index: int, op_name: str = "Op",
                shape=None) -> list:
        """The profile's build123d Faces as of *entry_index* (replay path)."""
        raise NotImplementedError

    def resolve_live(self, viewport: Any, op_name: str = "Op") -> list:
        """The profile's Faces in the viewport's current state (commit path).
        Bakes whatever the profile needs to re-find them on replay."""
        raise NotImplementedError

    def bake(self, history: "History", entry_index: int) -> None:
        """Fingerprint the profile from the state before *entry_index* so
        replay can re-find it (replay-side commits: loft, sweep).  No-op once
        baked."""

    def owner_body_id(self, history: "History") -> str | None:
        """The body this profile lives on (a sketch's parent body, a face's
        body), or None for a sketch on a world / datum plane."""
        raise NotImplementedError

    def key(self) -> tuple:
        """Hashable identity for UI lists: ("sketch", sid, i) / ("face", bid, i)."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def copy(self) -> "ProfileSource":
        return profile_from_dict(self.to_dict())


# ---------------------------------------------------------------------------
# SketchProfile
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class SketchProfile(ProfileSource):
    """
    sketch_id       : entry_id of the sketch entry
    face_indices    : which loops (indices into the sketch's build_faces());
                      None → every closed loop
    face_centroids  : UV centroid per selected loop, baked at commit — with
                      face_boundaries, re-matches loops across sketch edits
    face_boundaries : sorted entity uids bounding each selected loop
    """
    kind = "sketch"

    sketch_id:       str
    face_indices:    list | None = None
    face_centroids:  list | None = None
    face_boundaries: list | None = None

    # -- lookup ------------------------------------------------------------

    def sketch_record(self, history: "History", entry_index: int | None = None,
                      op_name: str = "Op"):
        """(record, SketchEntry) of the sketch; raises with op_name context."""
        idx = history.id_to_index(self.sketch_id)
        if idx is None:
            raise RuntimeError(f"{op_name}: sketch entry '{self.sketch_id}' not found")
        if entry_index is not None and idx >= entry_index:
            raise RuntimeError(
                f"{op_name}: sketch entry is after this entry — invalid reorder")
        rec = history._entries[idx]
        se = rec.params.get("sketch_entry")
        if se is None:
            raise RuntimeError(f"{op_name}: no sketch_entry at id '{self.sketch_id}'")
        return rec, se

    def sketch_entry(self, history: "History"):
        """The SketchEntry, or None — no reprojection, no errors."""
        idx = history.id_to_index(self.sketch_id)
        if idx is None:
            return None
        return history._entries[idx].params.get("sketch_entry")

    def owner_body_id(self, history: "History") -> str | None:
        se = self.sketch_entry(history)
        return se.body_id if se is not None else None

    def plane_normal(self, history: "History"):
        se = self.sketch_entry(history)
        return None if se is None else se.plane_normal

    # -- resolve -----------------------------------------------------------

    def resolve(self, history: "History", entry_index: int, op_name: str = "Op",
                shape=None) -> list:
        rec, se = self.sketch_record(history, entry_index, op_name)
        if rec.error:
            raise RuntimeError(f"{op_name}: sketch entry is in an error state")
        if se.plane_source is not None:
            from cad.history import reproject_consumed_sketch
            # Reproject at the *consuming* op's position so the profile
            # follows intervening ops on the sketch's parent face; falls back
            # to the authored plane if one of them destroyed the anchor face.
            ok, err = reproject_consumed_sketch(se, history, entry_index, rec)
            if not ok:
                raise RuntimeError(f"{op_name}: sketch reprojection failed: {err}")
        all_faces, all_regions = se.build_faces()
        if not all_faces:
            raise RuntimeError(f"{op_name}: sketch has no closed loops")
        if self.face_centroids:
            from cad.op_base import _match_sketch_faces, _lost_region_warning
            lost: list = []
            faces, matched = _match_sketch_faces(
                se, all_faces, all_regions, self.face_centroids,
                self.face_boundaries, op_name, lost)
            _lost_region_warning(history, entry_index, lost)
            if not self.face_boundaries:
                self.face_boundaries = matched   # backfill (see History.replay_from)
            return faces
        if self.face_indices is not None:
            faces = [all_faces[i] for i in self.face_indices
                     if 0 <= i < len(all_faces)]
            if not faces:
                raise RuntimeError(f"{op_name}: stored sketch face indices out of range")
            from cad import resolve_ledger
            resolve_ledger.note(resolve_ledger.FUZZY, "sketch region", "by index")
            return faces
        return list(all_faces)

    def resolve_live(self, viewport: Any, op_name: str = "Op") -> list:
        """Faces from the viewport's already-built sketch faces; bakes
        face_indices (all loops when None) + centroids + boundaries."""
        import numpy as np
        sketch_idx = viewport.history.id_to_index(self.sketch_id)
        if sketch_idx is None:
            raise RuntimeError(f"{op_name}: sketch entry '{self.sketch_id}' not found")
        se = viewport.history.entries[sketch_idx].params.get("sketch_entry")
        if se is None:
            raise RuntimeError(f"{op_name}: no sketch entry")
        all_faces = viewport._sketch_faces.get(sketch_idx, [])
        if not all_faces:
            raise RuntimeError(f"{op_name}: sketch has no closed loops")
        idxs = (list(range(len(all_faces))) if self.face_indices is None
                else [i for i in self.face_indices if 0 <= i < len(all_faces)])
        if not idxs:
            raise RuntimeError(f"{op_name}: no valid sketch loops selected")
        self.face_indices = idxs
        uvs = [all_faces[i][1] for i in idxs]
        self.face_centroids = (None if any(u is None for u in uvs) else
                               [np.asarray(u).mean(axis=0).tolist() for u in uvs])
        bounds = se.build_face_boundaries()
        self.face_boundaries = [sorted(bounds[i]) for i in idxs if i < len(bounds)]
        return [all_faces[i][0] for i in idxs]

    def bake(self, history: "History", entry_index: int) -> None:
        """Bake centroids + boundaries for the stored loops from the sketch
        as it is now (replay-side commit, e.g. loft).  face_indices=None
        stays None (it's the panel key of a one-loop sketch).  No-op once
        baked."""
        import numpy as np
        if self.face_centroids:
            return
        se = self.sketch_entry(history)
        if se is None:
            return
        all_faces, all_regions = se.build_faces()
        idxs = (list(range(len(all_faces))) if self.face_indices is None
                else [i for i in self.face_indices if 0 <= i < len(all_faces)])
        if not idxs or len(all_regions) != len(all_faces):
            return
        self.face_centroids = [np.asarray(all_regions[i][0]).mean(axis=0).tolist()
                               for i in idxs]
        bounds = se.build_face_boundaries()
        self.face_boundaries = [sorted(bounds[i]) for i in idxs if i < len(bounds)]

    # -- identity / serialization ------------------------------------------

    def single_index(self) -> int | None:
        """The one loop index of a one-loop profile (loft/sweep key), else None."""
        if self.face_indices and len(self.face_indices) == 1:
            return self.face_indices[0]
        return None

    def key(self) -> tuple:
        return ("sketch", self.sketch_id, self.single_index())

    def to_dict(self) -> dict:
        d: dict = {"kind": "sketch", "sketch_id": self.sketch_id}
        if self.face_indices is not None:
            d["face_indices"] = list(self.face_indices)
        if self.face_centroids is not None:
            d["face_centroids"] = [list(c) for c in self.face_centroids]
        if self.face_boundaries is not None:
            d["face_boundaries"] = [list(b) for b in self.face_boundaries]
        return d


# ---------------------------------------------------------------------------
# BodyFaceProfile
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class BodyFaceProfile(ProfileSource):
    """
    body_id  : body the face belongs to
    face_idx : face index at commit time (fallback when there is no ref)
    face_ref : FaceRef (planar) or AnyFaceRef (curved), baked at commit
    origin   : the face's provenance (cad.face_origin, planar faces), baked
               at commit.  Tried first, so the face is followed through
               parameter edits rather than matched by centroid and area.
    """
    kind = "face"

    body_id:  str
    face_idx: int
    face_ref: Any = None
    origin:   Any = None

    def owner_body_id(self, history: "History") -> str | None:
        return self.body_id

    def locate(self, shape, op_name: str = "Op", history=None, at=None):
        """(index, Face) of this face on *shape*; raises when it's gone.
        With *history* (and *at* = (body_id, before_index) of *shape*) the
        provenance is tried first."""
        if self.origin is not None and history is not None:
            idx, face = self.origin.find_in(history, shape, at)
            if face is not None:
                from cad import resolve_ledger
                resolve_ledger.note(resolve_ledger.EXACT, "body face profile")
                return idx, face
        if self.face_ref is not None:
            idx, face = self.face_ref.find_in(shape)
            if face is None:
                detail = ""
                explain = getattr(self.face_ref, "explain_no_match", None)
                if explain is not None:
                    try:
                        detail = " — " + explain(shape)
                    except Exception:
                        pass
                raise RuntimeError(
                    f"{op_name}: could not re-locate face on body "
                    f"'{self.body_id}'{detail}")
            return idx, face
        all_f = list(shape.faces())
        if not 0 <= self.face_idx < len(all_f):
            raise RuntimeError(
                f"{op_name}: face_idx {self.face_idx} out of range on body "
                f"'{self.body_id}'")
        from cad import resolve_ledger
        resolve_ledger.note(resolve_ledger.FUZZY, "body face profile", "by index")
        return self.face_idx, all_f[self.face_idx]

    def resolve(self, history: "History", entry_index: int, op_name: str = "Op",
                shape=None) -> list:
        if shape is None:
            shape = history._shape_for_body_at(self.body_id, entry_index)
        if shape is None:
            raise RuntimeError(f"{op_name}: no shape for body '{self.body_id}'")
        return [self.locate(shape, op_name, history,
                            (self.body_id, entry_index))[1]]

    def resolve_live(self, viewport: Any, op_name: str = "Op") -> list:
        shape = viewport.workspace.current_shape(self.body_id)
        if shape is None:
            raise RuntimeError(f"{op_name}: body '{self.body_id}' has no shape")
        all_f = list(shape.faces())
        if not 0 <= self.face_idx < len(all_f):
            raise RuntimeError(
                f"{op_name}: face_idx {self.face_idx} out of range on body "
                f"'{self.body_id}'")
        face = all_f[self.face_idx]
        self.face_ref = make_face_ref(face)
        h = viewport.history
        self.origin = _face_origin(h, self.body_id, face, h.cursor + 1)
        return [face]

    def bake(self, history: "History", entry_index: int) -> None:
        """Commit-time: ref + provenance of face_idx on the body as of
        *entry_index* (face_idx is the face just picked)."""
        if self.face_ref is not None and self.origin is not None:
            return
        shape = history._shape_for_body_at(self.body_id, entry_index)
        if shape is None:
            return
        all_f = list(shape.faces())
        if 0 <= self.face_idx < len(all_f):
            face = all_f[self.face_idx]
            if self.face_ref is None:
                self.face_ref = make_face_ref(face)
            if self.origin is None:
                self.origin = _face_origin(history, self.body_id, face, entry_index)

    def is_planar(self) -> bool:
        from cad.face_ref import FaceRef
        return isinstance(self.face_ref, FaceRef)

    def key(self) -> tuple:
        return ("face", self.body_id, self.face_idx)

    def to_dict(self) -> dict:
        d = {"kind": "face", "body_id": self.body_id,
             "face_idx": int(self.face_idx),
             "ref": face_ref_to_dict(self.face_ref)}
        if self.origin is not None:
            d["origin"] = self.origin.to_dict()
        return d


def _face_origin(history, body_id, face, before_index):
    """Provenance of a planar profile face, or None."""
    from cad.face_ref import FaceRef
    if FaceRef.from_occ_face(face.wrapped) is None:
        return None
    from cad.face_origin import any_origin_for_face
    try:
        return any_origin_for_face(history, body_id, face.wrapped, before_index)
    except Exception as ex:
        print(f"[Profile] face has no provenance: {ex}")
        return None


# ---------------------------------------------------------------------------
# Face refs
# ---------------------------------------------------------------------------

def make_face_ref(face):
    """FaceRef for a planar build123d face, else AnyFaceRef."""
    from cad.face_ref import FaceRef, AnyFaceRef
    return (FaceRef.from_occ_face(face.wrapped)
            or AnyFaceRef.from_occ_face(face.wrapped))


def face_ref_to_dict(ref) -> dict | None:
    from cad.face_ref import FaceRef, AnyFaceRef
    if isinstance(ref, FaceRef):
        return {"type": "plane", "normal": list(ref.normal), "area": ref.area,
                "centroid_perp": list(ref.centroid_perp),
                "centroid_along": ref.centroid_along}
    if isinstance(ref, AnyFaceRef):
        d = {"type": "any", "normal": None}
        d.update(ref.to_dict())
        return d
    return None


def face_ref_from_dict(d: dict | None):
    from cad.face_ref import FaceRef, AnyFaceRef
    if not d:
        return None
    if d.get("type") == "any" or "centroid" in d:
        return AnyFaceRef.from_dict(d)
    return FaceRef(normal=tuple(d["normal"]), area=float(d["area"]),
                   centroid_perp=tuple(d["centroid_perp"]),
                   centroid_along=float(d["centroid_along"]))


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def profile_from_dict(d: dict) -> ProfileSource:
    if d.get("kind") == "face":
        from cad.face_origin import face_origin_from_dict
        return BodyFaceProfile(body_id=d["body_id"], face_idx=int(d.get("face_idx", 0)),
                               face_ref=face_ref_from_dict(d.get("ref")),
                               origin=face_origin_from_dict(d.get("origin")))
    if d.get("kind") == "sketch":
        fi = d.get("face_indices")
        return SketchProfile(
            sketch_id=d["sketch_id"],
            face_indices=list(fi) if fi is not None else None,
            face_centroids=d.get("face_centroids"),
            face_boundaries=d.get("face_boundaries"))
    raise ValueError(f"unknown profile kind {d.get('kind')!r}")


def as_profile(p) -> ProfileSource:
    """Normalize anything a caller (panel, test, legacy save) hands an op:

      ProfileSource                 → itself
      {"kind": ...} dict            → profile_from_dict
      ("sketch", sid, i|None)       → SketchProfile of loop i (None = all)
      ("face", body_id, face_idx)   → BodyFaceProfile (ref baked at commit)
      (sid, i|None) / [sid, i]      → legacy loft key: SketchProfile
      "sid"                         → legacy loft id: whole sketch
    """
    if isinstance(p, ProfileSource):
        return p
    if isinstance(p, dict):
        return profile_from_dict(p)
    if isinstance(p, str):
        return SketchProfile(sketch_id=p)
    p = tuple(p)
    if len(p) == 3 and p[0] == "face":
        return BodyFaceProfile(body_id=p[1], face_idx=int(p[2]))
    if len(p) == 3 and p[0] == "sketch":
        return SketchProfile(sketch_id=p[1],
                             face_indices=None if p[2] is None else [int(p[2])])
    sid = p[0]
    fi = p[1] if len(p) > 1 else None
    return SketchProfile(sketch_id=sid, face_indices=None if fi is None else [int(fi)])


def profiles_to_dicts(profiles) -> list[dict]:
    return [as_profile(p).to_dict() for p in profiles]


def profiles_from_dicts(raw) -> list[ProfileSource]:
    return [as_profile(p) for p in (raw or [])]


def faces_of(profiles, history: "History", entry_index: int,
             op_name: str = "Op") -> list:
    """All profiles' faces, in order (replay path)."""
    faces: list = []
    for p in profiles:
        faces.extend(p.resolve(history, entry_index, op_name))
    return faces


def faces_live(profiles, viewport: Any, op_name: str = "Op") -> list:
    """All profiles' faces, in order (commit path; bakes refs)."""
    faces: list = []
    for p in profiles:
        faces.extend(p.resolve_live(viewport, op_name))
    return faces


def persist_backfilled_boundaries(op, params: dict) -> None:
    """After a replay, write region boundaries that sketch matching
    backfilled (ops saved before boundaries existed matched by centroid) into
    the entry's params, so later parameter changes match by identity."""
    for key in ("profiles", "from_profiles"):
        profs, raw = getattr(op, key, None), params.get(key)
        if not profs or not isinstance(raw, list) or len(raw) != len(profs):
            continue
        for prof, d in zip(profs, raw):
            if (isinstance(prof, SketchProfile) and isinstance(d, dict)
                    and prof.face_boundaries and not d.get("face_boundaries")):
                d["face_boundaries"] = [list(b) for b in prof.face_boundaries]
        return
    # Pre-profile params: one sketch, boundaries at the top level.
    fb = getattr(op, "face_boundaries", None)
    if fb is None:
        sp = sketch_profiles(getattr(op, "profiles", None) or [])
        fb = sp[0].face_boundaries if len(sp) == 1 else None
    if fb and "face_boundaries" not in params:
        params["face_boundaries"] = fb


def sketch_profiles(profiles) -> list[SketchProfile]:
    return [p for p in profiles if isinstance(p, SketchProfile)]


def first_sketch_id(profiles) -> str | None:
    sp = sketch_profiles(profiles)
    return sp[0].sketch_id if sp else None
