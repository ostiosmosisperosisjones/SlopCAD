"""
cad/face_ref.py

Stable face references that survive boolean operations.

A FaceRef fingerprints a face by:
  - normal vector        (orientation, invariant through most ops)
  - area                 (size, invariant unless face is split)
  - centroid_perp        (centroid projected onto the plane perpendicular
                          to the normal — invariant through extrude/cut)
  - centroid_along       (centroid projected onto the normal — shifts
                          predictably: += distance for extrude/cut)

To find a face after an operation, we match on normal + area +
centroid_perp. That triple uniquely identifies a face in all cases
we've verified, and is robust to topology renumbering.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _occ_face_props(occ_face):
    """Return (centroid_xyz, area) for a raw OCC face."""
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(occ_face, props)
    c = props.CentreOfMass()
    return np.array([c.X(), c.Y(), c.Z()]), props.Mass()


def _occ_face_normal(occ_face):
    """
    Return unit normal for a planar OCC face, or None if not planar.
    Normal direction follows the face orientation in the shape.
    """
    try:
        surf = BRepAdaptor_Surface(occ_face)
        if surf.GetType() != GeomAbs_Plane:
            return None
        d = surf.Plane().Axis().Direction()
        n = np.array([d.X(), d.Y(), d.Z()])
        return n / np.linalg.norm(n)
    except Exception:
        return None


def _occ_face_anchor_and_normal(occ_face):
    """Return (anchor_xyz, unit_normal) sampled on the face surface.

    First tries to project the area centroid onto the surface (good anchor for
    trimmed/holed planar faces). Falls back to the UV-midpoint of the face's
    bounds when projection is ambiguous (closed cylinders/spheres where the
    centroid is equidistant from infinitely many surface points). Honors face
    orientation. Returns (None, None) if sampling fails.
    """
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepTools import BRepTools
        from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
        from OCP.gp import gp_Pnt, gp_Vec
        from OCP.TopAbs import TopAbs_REVERSED

        surface = BRep_Tool.Surface_s(occ_face)
        if surface is None:
            return None, None

        u = v = None
        try:
            c, _ = _occ_face_props(occ_face)
            proj = GeomAPI_ProjectPointOnSurf(
                gp_Pnt(float(c[0]), float(c[1]), float(c[2])), surface)
            if proj.NbPoints() >= 1:
                u, v = proj.LowerDistanceParameters()
        except Exception:
            pass

        if u is None:
            umin, umax, vmin, vmax = BRepTools.UVBounds_s(occ_face)
            u = 0.5 * (umin + umax)
            v = 0.5 * (vmin + vmax)

        p   = gp_Pnt()
        d1u = gp_Vec()
        d1v = gp_Vec()
        surface.D1(u, v, p, d1u, d1v)
        anchor = np.array([p.X(), p.Y(), p.Z()], dtype=float)
        n = np.array([d1u.Y() * d1v.Z() - d1u.Z() * d1v.Y(),
                      d1u.Z() * d1v.X() - d1u.X() * d1v.Z(),
                      d1u.X() * d1v.Y() - d1u.Y() * d1v.X()], dtype=float)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            return None, None
        n /= ln
        if occ_face.Orientation() == TopAbs_REVERSED:
            n = -n
        return anchor, n
    except Exception:
        return None, None


def _occ_face_normal_at_centroid(occ_face):
    """Back-compat: return just the normal from _occ_face_anchor_and_normal."""
    _, n = _occ_face_anchor_and_normal(occ_face)
    return n


# ---------------------------------------------------------------------------
# FaceRef
# ---------------------------------------------------------------------------

@dataclass
class FaceRef:
    """
    Geometry-based face identifier — survives boolean topology changes.

    All values are in world (un-normalised) coordinates, matching
    build123d / OCCT space.
    """
    normal:          tuple   # (nx, ny, nz) unit vector
    area:            float
    centroid_perp:   tuple   # centroid projected perpendicular to normal
    centroid_along:  float   # centroid projected along normal

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    @classmethod
    def from_occ_face(cls, occ_face) -> "FaceRef | None":
        """Build a FaceRef from a raw OCC TopoDS_Face. Returns None if not planar."""
        normal = _occ_face_normal(occ_face)
        if normal is None:
            return None
        centroid, area = _occ_face_props(occ_face)
        along = float(np.dot(centroid, normal))
        perp  = centroid - along * normal
        return cls(
            normal         = tuple(np.round(normal, 8)),
            area           = round(float(area), 6),
            centroid_perp  = tuple(np.round(perp, 6)),
            centroid_along = round(along, 6),
        )

    @classmethod
    def from_b3d_face(cls, face) -> "FaceRef | None":
        """Build from a build123d Face object."""
        return cls.from_occ_face(face.wrapped)

    # ----------------------------------------------------------------
    # Matching
    # ----------------------------------------------------------------

    def find_in(self, shape,
                normal_tol:  float = 0.001,
                perp_tol:    float = 0.1,
                area_frac:   float = 0.5,
                ) -> tuple[int, object] | tuple[None, None]:
        """
        Find the best matching face in a build123d shape.

        Filters on: same normal direction (parallel) and perp-centroid within
        perp_tol mm.  Area is *not* a hard filter — intervening cuts/fillets
        can shrink a face's area substantially while it remains the same
        physical face.  Instead area is used as a fallback tiebreaker only
        when along_dist is identical.

        Ranking among the surviving candidates:
          1. same-direction normal beats opposite
          2. closest centroid_along to ref (face didn't move far)
          3. closest perp distance
          4. closest area (final tiebreaker)

        area_frac caps how much area drift is tolerated — a candidate whose
        area differs from the ref by more than area_frac * ref.area is
        rejected entirely.  Default 0.5 (50%) is generous enough to survive
        normal cut/fillet operations but rejects matches to wholly different
        faces.

        Returns (face_index, b3d_face) or (None, None) if no match.
        """
        ref_normal = np.array(self.normal)
        ref_perp   = np.array(self.centroid_perp)
        # Absolute area cap so e.g. a 1mm² face can't accept a 100mm² match.
        # Use 0.5mm² as a floor so tiny faces still admit small drift.
        area_cap   = max(self.area * area_frac, 0.5)

        best_idx        = None
        best_face       = None
        best_perp_dist  = float("inf")
        best_same_dir   = False
        best_along_dist = float("inf")
        best_area_dist  = float("inf")

        for idx, face in enumerate(shape.faces()):
            occ = face.wrapped
            n = _occ_face_normal(occ)
            if n is None:
                continue

            # Normal must be parallel (same or opposite direction)
            dot = float(np.dot(n, ref_normal))
            if abs(abs(dot) - 1.0) > normal_tol:
                continue
            same_dir = dot > 0.0

            centroid, area = _occ_face_props(occ)

            # Area drift cap — generous but not unbounded
            if abs(area - self.area) > area_cap:
                continue

            # Perpendicular centroid component must match
            along = float(np.dot(centroid, n))
            perp  = centroid - along * n
            perp_dist = float(np.linalg.norm(perp - ref_perp))
            if perp_dist > perp_tol:
                continue

            along_dist = abs(along - self.centroid_along)
            area_dist  = abs(area - self.area)

            better = False
            if best_idx is None:
                better = True
            elif same_dir and not best_same_dir:
                better = True
            elif same_dir == best_same_dir:
                if along_dist < best_along_dist - 1e-6:
                    better = True
                elif abs(along_dist - best_along_dist) < 1e-6:
                    if perp_dist < best_perp_dist - 1e-6:
                        better = True
                    elif abs(perp_dist - best_perp_dist) < 1e-6 and area_dist < best_area_dist:
                        better = True

            if better:
                best_idx        = idx
                best_face       = face
                best_perp_dist  = perp_dist
                best_same_dir   = same_dir
                best_along_dist = along_dist
                best_area_dist  = area_dist

        return best_idx, best_face

    def explain_no_match(self, shape,
                         normal_tol: float = 0.001,
                         perp_tol:   float = 0.1,
                         area_frac:  float = 0.5) -> str:
        """Human-readable reason find_in() returned nothing — what was sought
        and the closest candidate, with the filter that rejected it.

        Used to turn an opaque "could not relocate face" into an actionable
        line.  Pure diagnostics: never raises, never affects matching.
        """
        ref_normal = np.array(self.normal)
        ref_perp   = np.array(self.centroid_perp)
        area_cap   = max(self.area * area_frac, 0.5)

        want = (f"sought face normal={tuple(round(c, 3) for c in self.normal)} "
                f"area={self.area:.3f} perp={tuple(round(c, 3) for c in self.centroid_perp)}")

        # Rank candidates by how far each got through the filter chain (higher
        # stage = more relevant), then by perp distance — so the reported face
        # is the one that nearly matched, not just any perp-close side face.
        best = None  # (stage, perp_dist, reason)
        for idx, face in enumerate(shape.faces()):
            n = _occ_face_normal(face.wrapped)
            if n is None:
                continue
            dot = float(np.dot(n, ref_normal))
            centroid, area = _occ_face_props(face.wrapped)
            along = float(np.dot(centroid, n))
            perp  = centroid - along * n
            perp_dist = float(np.linalg.norm(perp - ref_perp))

            if abs(abs(dot) - 1.0) > normal_tol:
                stage, reason = 0, f"normal off by {abs(abs(dot) - 1.0):.4f} (tol {normal_tol})"
            elif abs(area - self.area) > area_cap:
                stage, reason = 1, (f"area {area:.3f} vs {self.area:.3f}, "
                                    f"drift {abs(area - self.area):.3f} > cap {area_cap:.3f}")
            elif perp_dist > perp_tol:
                stage, reason = 2, f"perp-centroid off by {perp_dist:.3f}mm (tol {perp_tol})"
            else:
                stage, reason = 3, "passed all filters (unexpected)"
            key = (stage, -perp_dist)  # prefer higher stage, then smaller perp
            if best is None or key > (best[0], -best[1]):
                best = (stage, perp_dist, f"face {idx}: {reason}")

        if best is None:
            return f"{want}; shape has no planar faces to match against"
        return f"{want}; nearest candidate {best[2]}"

    # ----------------------------------------------------------------
    # Prediction helpers
    # ----------------------------------------------------------------

    def predict_after_extrude(self, distance: float) -> "FaceRef":
        """
        Return the FaceRef we expect for the new top/bottom face
        after extruding this face by *distance*.

        The new face has the same normal and area, centroid_along
        shifts by distance, centroid_perp is unchanged.
        """
        return FaceRef(
            normal         = self.normal,
            area           = self.area,
            centroid_perp  = self.centroid_perp,
            centroid_along = round(self.centroid_along + distance, 6),
        )


# ---------------------------------------------------------------------------
# AnyFaceRef  —  centroid+area fingerprint that works for non-planar faces
# ---------------------------------------------------------------------------

@dataclass
class AnyFaceRef:
    """
    Geometry-based face identifier for any face type (planar or curved).

    Matches by centroid position and area — sufficient to re-locate a
    face after operations that preserve face identity (thicken, offset).

    category (optional) — 'plane' or 'curved', captured at commit time and
    used as a topology-aware fallback when strict centroid/area matching
    fails after upstream edits (e.g. draft turning a cylinder into a cone
    keeps its centroid roughly fixed but changes area by hundreds of mm²).
    """
    centroid: tuple   # (x, y, z) world coords
    area:     float
    category: str = ""   # 'plane' | 'curved' | '' (legacy: unknown)

    @classmethod
    def from_occ_face(cls, occ_face) -> "AnyFaceRef":
        centroid, area = _occ_face_props(occ_face)
        return cls(
            centroid = tuple(np.round(centroid, 6)),
            area     = round(float(area), 6),
            category = _surface_category(occ_face),
        )

    def find_in(self, shape,
                area_tol:     float = 0.5,
                centroid_tol: float = 0.5,
                ) -> tuple[int, object] | tuple[None, None]:
        """
        Find best matching face by centroid proximity and area.

        Tries strict matching first (cheap, exact for replays with no
        topology change); falls back to a tolerant category-aware match
        when strict matching misses — survives upstream edits like draft
        that move centroid by mm and area by hundreds of mm².

        Returns (face_index, b3d_face) or (None, None).
        """
        ref_c = np.array(self.centroid)
        best_idx  = None
        best_face = None
        best_dist = float("inf")

        for idx, face in enumerate(shape.faces()):
            centroid, area = _occ_face_props(face.wrapped)
            if abs(area - self.area) > area_tol:
                continue
            dist = float(np.linalg.norm(centroid - ref_c))
            if dist > centroid_tol:
                continue
            if dist < best_dist:
                best_dist  = dist
                best_idx   = idx
                best_face  = face

        if best_idx is not None:
            return best_idx, best_face

        # Strict match missed — fall back to category-aware scoring if we
        # have a category recorded. Without one, behave as before (None).
        if self.category:
            return _find_face_by_category_score(shape, self)

        return None, None


# ---------------------------------------------------------------------------
# Category-aware fallback used by AnyFaceRef.find_in()
# ---------------------------------------------------------------------------

# Faces with a score above this aren't trusted as a match — keeps false
# positives out when nothing in the shape really corresponds to the ref.
_FACE_SCORE_CUTOFF = 200.0


def _surface_category(occ_face) -> str:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    return 'plane' if BRepAdaptor_Surface(occ_face).GetType() == GeomAbs_Plane else 'curved'


def _find_face_by_category_score(shape, ref: "AnyFaceRef"):
    """Lower-is-better scoring: centroid distance + 50 × relative area diff,
    restricted to faces in the same surface category. Returns (idx, face) or
    (None, None) if the best score exceeds the cutoff."""
    ref_c = np.array(ref.centroid)
    best_idx, best_face, best_score = None, None, float('inf')
    for idx, face in enumerate(shape.faces()):
        if _surface_category(face.wrapped) != ref.category:
            continue
        centroid, area = _occ_face_props(face.wrapped)
        cdist = float(np.linalg.norm(centroid - ref_c))
        adiff = abs(area - ref.area) / max(ref.area, 1.0)
        score = cdist + 50.0 * adiff
        if score < best_score:
            best_score, best_idx, best_face = score, idx, face
    if best_score > _FACE_SCORE_CUTOFF:
        return None, None
    return best_idx, best_face
