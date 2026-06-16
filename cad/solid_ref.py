"""
cad/solid_ref.py

Stable solid references — the body-level analogue of FaceRef / EdgeRef.

When an operation splits one input into several disjoint solids (a force_new_body
extrude/revolve, a loft that yields multiple lumps), each resulting solid becomes
its own body.  The association between a child body and its physical solid must
survive save → load → replay.

History stored that association positionally: child_body_ids[i] ↔
list(result.solids())[i].  But OCCT's solid iteration order is not guaranteed to
match between the commit-time shape and a replay-rebuilt shape (nor across edits
or OCCT versions), so the index mapping silently scrambles which body gets which
solid on reload.

SolidRef fixes that by fingerprinting each solid geometrically — volume, center
of mass, and bounding-box extents — so replay can assign each rebuilt solid to
the body it belongs to regardless of iteration order.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib


def _occ_solid_props(occ_solid):
    """Return (volume, center_of_mass_xyz, bbox_extents_xyz) for a raw OCC solid."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(occ_solid, props)
    com = props.CentreOfMass()
    vol = props.Mass()

    box = Bnd_Box()
    BRepBndLib.Add_s(occ_solid, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    extents = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

    return (float(vol),
            np.array([com.X(), com.Y(), com.Z()]),
            extents)


@dataclass
class SolidRef:
    """
    Geometry-based solid identifier — survives the solid-ordering churn that
    plain indices do not.

    volume   : solid volume (mm³)
    com      : center of mass (world mm)
    extents  : bounding-box extents (dx, dy, dz)
    """
    volume:  float
    com:     tuple
    extents: tuple

    @classmethod
    def from_occ_solid(cls, occ_solid) -> "SolidRef":
        vol, com, extents = _occ_solid_props(occ_solid)
        return cls(
            volume  = round(vol, 6),
            com     = tuple(np.round(com, 6).tolist()),
            extents = tuple(np.round(extents, 6).tolist()),
        )

    @classmethod
    def from_b3d_solid(cls, solid) -> "SolidRef":
        return cls.from_occ_solid(solid.wrapped)

    def _score(self, occ_solid) -> float:
        """Lower-is-better distance to a candidate solid: center-of-mass
        distance dominates (it's what distinguishes sibling segments), with
        volume and extent mismatch as gentle tiebreakers so two coincident
        centroids still resolve.  Relative terms keep large and small parts on
        the same footing."""
        vol, com, extents = _occ_solid_props(occ_solid)
        com_dist = float(np.linalg.norm(com - np.array(self.com)))
        vol_rel  = abs(vol - self.volume) / max(self.volume, 1.0)
        ext_rel  = float(np.linalg.norm(extents - np.array(self.extents))
                         ) / max(float(np.linalg.norm(self.extents)), 1.0)
        return com_dist + 10.0 * vol_rel + ext_rel

    # Above this score nothing is trusted as a match — keeps a body from
    # adopting a wholly unrelated solid when its real segment is gone.
    _SCORE_CUTOFF = 5.0

    def find_in(self, shape) -> tuple[int, object] | tuple[None, None]:
        """Find the best-matching solid in a build123d shape.

        Returns (solid_index, b3d_solid) or (None, None) if nothing scores
        within the cutoff.
        """
        best_idx, best_solid, best_score = None, None, float("inf")
        for idx, solid in enumerate(shape.solids()):
            score = self._score(solid.wrapped)
            if score < best_score:
                best_score, best_idx, best_solid = score, idx, solid
        if best_idx is None or best_score > self._SCORE_CUTOFF:
            return None, None
        return best_idx, best_solid


def assign_solids_to_children(result, child_body_ids, child_solid_refs,
                              workspace, history=None, entry_index=None):
    """Distribute the solids of *result* among *child_body_ids* and set each
    child body's source_shape.

    Mapping is by SolidRef when refs are present (robust to ordering); falls
    back to positional index when a ref is missing or its match is ambiguous,
    so legacy entries (saved before refs existed) behave no worse than before.

    When child_solid_refs is absent but *history*/*entry_index* are supplied,
    refs are first recovered from downstream anchors (see
    recover_child_solid_refs) so legacy files map correctly instead of relying
    on the fragile positional order that originally broke them.

    Returns the list of build123d solids in child-body order — element i is the
    solid assigned to child_body_ids[i] (or None if none could be assigned).
    """
    from build123d import Compound

    solids = list(result.solids())
    n = len(child_body_ids)
    assigned = [None] * n
    used = set()

    refs = list(child_solid_refs) if child_solid_refs else []
    if not refs and history is not None and entry_index is not None:
        refs = recover_child_solid_refs(result, child_body_ids,
                                        history, entry_index)
    for i in range(n):
        ref = refs[i] if i < len(refs) else None
        chosen = None
        if ref is not None:
            idx, _ = ref.find_in(result)
            if idx is not None and idx not in used:
                chosen = idx
        if chosen is None:
            # Positional fallback: first unused solid at this index, then any.
            if i < len(solids) and i not in used:
                chosen = i
            else:
                chosen = next((j for j in range(len(solids)) if j not in used),
                              None)
        if chosen is not None:
            used.add(chosen)
            assigned[i] = Compound(solids[chosen].wrapped)

    if workspace is not None:
        for i, bid in enumerate(child_body_ids):
            body = workspace.bodies.get(bid)
            if body is not None:
                body.source_shape = assigned[i]

    return assigned


def _solid_contains_point(occ_solid, pt, pad: float = 0.5) -> bool:
    """True if *pt* lies within the (padded) bounding box of a raw OCC solid.

    A cheap, robust containment test for recovery: a downstream anchor (a face
    centroid or edge midpoint that lives ON a child body) falls inside that
    body's segment.  Padding absorbs the small drift between the anchor's
    recorded position and the freshly rebuilt segment.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box = Bnd_Box()
    BRepBndLib.Add_s(occ_solid, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    x, y, z = pt
    return (xmin - pad <= x <= xmax + pad and
            ymin - pad <= y <= ymax + pad and
            zmin - pad <= z <= zmax + pad)


def _downstream_anchor_for_body(history, entry_index, body_id):
    """Find a world-space point known to lie on *body_id*, by scanning history
    entries after *entry_index* for the first geometric reference anchored to
    that body.  Returns an (x, y, z) tuple or None.

    Recognized anchors (all carry a body_id + a world position):
      - a sketch whose plane is a FacePlaneSource on this body  (face_ref)
      - an entry tagged to this body that carries a FaceRef     (entry.face_ref)
      - a ThickenOp whose source is this body                   (face_refs)
    This is how legacy files (saved before child_solid_refs existed) recover
    which physical segment each split child actually is.
    """
    entries = history._entries
    for i in range(entry_index + 1, len(entries)):
        e = entries[i]

        # 1. Sketch on a face of this body.
        se = e.params.get("sketch_entry")
        if se is not None:
            ps = getattr(se, "plane_source", None)
            fr = getattr(ps, "face_ref", None)
            if (getattr(ps, "body_id", None) == body_id and fr is not None):
                # FaceRef stores centroid as centroid_perp + centroid_along·normal
                n = np.array(fr.normal)
                return tuple((np.array(fr.centroid_perp)
                              + fr.centroid_along * n).tolist())

        # 2. Entry tagged to this body carrying a FaceRef.
        if e.body_id == body_id and getattr(e, "face_ref", None) is not None:
            fr = e.face_ref
            n = np.array(fr.normal)
            return tuple((np.array(fr.centroid_perp)
                          + fr.centroid_along * n).tolist())

        # 3. Thicken whose source is this body (AnyFaceRef centroids).
        if e.params.get("source_body_id") == body_id:
            refs = e.params.get("face_refs")
            if refs:
                c = refs[0].get("centroid") if isinstance(refs[0], dict) else None
                if c is not None:
                    return tuple(c)
    return None


def recover_child_solid_refs(result, child_body_ids, history, entry_index):
    """Best-effort reconstruction of child_solid_refs for a legacy split entry.

    For each child body, find a downstream anchor point known to lie on it and
    match it to the segment whose bbox contains that point; build a SolidRef
    from that segment.  Children with no anchor are left as None (the caller's
    positional fallback handles them, by elimination against already-claimed
    segments).

    Returns a list aligned with child_body_ids: SolidRef | None per child.
    """
    solids = list(result.solids())
    refs = [None] * len(child_body_ids)
    claimed = set()
    for i, bid in enumerate(child_body_ids):
        anchor = _downstream_anchor_for_body(history, entry_index, bid)
        if anchor is None:
            continue
        for j, s in enumerate(solids):
            if j in claimed:
                continue
            if _solid_contains_point(s.wrapped, anchor):
                refs[i] = SolidRef.from_occ_solid(s.wrapped)
                claimed.add(j)
                break
    return refs


def solid_refs_to_dicts(solids) -> list:
    """Build a list of serializable SolidRef dicts, one per solid (in order)."""
    out = []
    for s in solids:
        r = SolidRef.from_occ_solid(s.wrapped)
        out.append({"volume": r.volume, "com": list(r.com),
                    "extents": list(r.extents)})
    return out


def solid_refs_from_dicts(dicts) -> list:
    """Inverse of solid_refs_to_dicts()."""
    if not dicts:
        return []
    return [SolidRef(volume=float(d["volume"]),
                     com=tuple(d["com"]),
                     extents=tuple(d["extents"]))
            for d in dicts]
