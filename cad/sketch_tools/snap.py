"""
cad/sketch_tools/snap.py

SnapEngine — resolves a raw cursor (u,v) to a snapped point.

For ReferenceEntity instances with original OCCT edges, snap queries the
true curve geometry rather than the tessellated point array:
  - Circles / arcs  → exact center, exact endpoint, exact mid-parameter pt
  - Nearest-on-curve → GeomAPI_ProjectPointOnCurve (exact)
  - No false midpoints on tessellation vertices

For LineEntity and ReferenceEntity without occ_edges, uses the polyline.

The sketch plane is passed into resolve() so world↔UV conversion is exact.

Snap priority: forced → endpoint → midpoint → center → nearest → grid → free
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


class SnapType(Enum):
    FREE         = auto()
    GRID         = auto()
    ENDPOINT     = auto()
    MIDPOINT     = auto()
    CENTER       = auto()
    NEAREST      = auto()
    TANGENT      = auto()
    INTERSECTION = auto()   # nearest crossing of any two entities
    ANGLE        = auto()   # angle-locked to 15° increments from anchor


@dataclass
class SnapResult:
    point      : np.ndarray   # resolved (u, v) in sketch mm
    type       : SnapType
    cursor_raw : np.ndarray   # original unsnapped cursor
    entity_idx : int | None = None   # index into sketch.entities of the snapped-to entity

    @property
    def snapped(self) -> bool:
        return self.type != SnapType.FREE


# ---------------------------------------------------------------------------
# SnapEngine
# ---------------------------------------------------------------------------

class SnapEngine:

    ANGLE_STEP_DEG = 15.0   # degrees between sticky angles

    def __init__(self, snap_radius_mm: float = 2.0, grid_size: float = 1.0):
        self.snap_radius_mm  = snap_radius_mm
        self.grid_size       = grid_size
        self.forced_type: SnapType | None = None   # legacy one-shot
        self.declared_type: SnapType | None = None  # sticky until click
        self.anchor_pt: np.ndarray | None = None    # set by active tool; used by tangent/angle snap
        self.angle_snap_active: bool = False         # True while Shift held in a modal tool
        self.enabled: set[SnapType] = {
            SnapType.ENDPOINT,
            SnapType.MIDPOINT,
            SnapType.CENTER,
            SnapType.NEAREST,
        }
        # Extra fixed UV snap points (e.g. world origin on a world-plane sketch)
        self.extra_points: list[np.ndarray] = []

    def declare(self, snap_type: SnapType):
        """Set a sticky snap declaration — active until consume_declared()."""
        self.declared_type = snap_type

    def consume_declared(self):
        """Clear the sticky declaration after a click."""
        self.declared_type = None

    def set_angle_snap(self, active: bool):
        self.angle_snap_active = active

    def set_grid_snap(self, active: bool):
        """Legacy — kept for compatibility; maps to angle snap."""
        self.set_angle_snap(active)

    # ------------------------------------------------------------------
    # Angle-lock helper
    # ------------------------------------------------------------------

    def _apply_angle_lock(self, cursor: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Project cursor onto the nearest 15° ray from anchor_pt.
        Returns (locked_cursor, locked_angle_deg).
        Only called when angle_snap_active and anchor_pt is set.
        """
        delta = cursor - self.anchor_pt
        dist  = float(np.linalg.norm(delta))
        if dist < 1e-9:
            return cursor.copy(), 0.0
        raw_deg  = float(np.degrees(np.arctan2(delta[1], delta[0])))
        step     = self.ANGLE_STEP_DEG
        snapped  = round(raw_deg / step) * step
        rad      = np.radians(snapped)
        locked   = self.anchor_pt + dist * np.array([np.cos(rad), np.sin(rad)])
        return locked, snapped % 360

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def resolve(self, cursor_pt2d: np.ndarray,
                sketch_entities: list,
                plane,                        # SketchPlane
                camera_distance: float = 100.0) -> SnapResult:
        """
        cursor_pt2d     : (2,) raw cursor in sketch UV mm
        sketch_entities : list of LineEntity / ReferenceEntity
        plane           : SketchPlane — needed for world↔UV conversion
        """
        cursor = np.array(cursor_pt2d, dtype=np.float64)
        radius = self.snap_radius_mm

        # Angle lock: when Shift is held and a tool anchor is set, constrain
        # the cursor to 15° rays from the anchor before running other snaps.
        angle_locked = False
        locked_angle = 0.0
        if self.angle_snap_active and self.anchor_pt is not None:
            cursor, locked_angle = self._apply_angle_lock(cursor)
            angle_locked = True

        # declared_type is sticky (set by key press, cleared by consume_declared)
        # forced_type is one-shot legacy
        forced = self.declared_type or self.forced_type
        self.forced_type = None

        if forced is not None:
            search_r = max(radius * 8, 20.0)
            p, eidx = self._try_snap(forced, cursor, sketch_entities, search_r, plane)
            if p is not None:
                return SnapResult(point=p, type=forced, cursor_raw=cursor,
                                  entity_idx=eidx)
            return SnapResult(point=cursor.copy(), type=SnapType.FREE,
                              cursor_raw=cursor)

        # When angle-locked, run endpoint/midpoint/center/nearest snaps against
        # the locked cursor (so snapping to geometry on the constrained ray works),
        # but skip GRID (irrelevant when angle-locked).
        snap_order = [SnapType.ENDPOINT, SnapType.MIDPOINT,
                      SnapType.CENTER, SnapType.NEAREST, SnapType.GRID]
        for snap_type in snap_order:
            if angle_locked and snap_type == SnapType.GRID:
                continue
            if snap_type not in self.enabled:
                continue
            p, eidx = self._try_snap(snap_type, cursor, sketch_entities, radius, plane)
            if p is not None:
                return SnapResult(point=p, type=snap_type, cursor_raw=cursor,
                                  entity_idx=eidx)

        if angle_locked:
            return SnapResult(point=cursor.copy(), type=SnapType.ANGLE,
                              cursor_raw=cursor_pt2d)

        return SnapResult(point=cursor.copy(), type=SnapType.FREE,
                          cursor_raw=cursor)

    def _try_snap(self, snap_type, cursor, entities, radius, plane):
        """Returns (point, entity_idx) or (None, None)."""
        if snap_type == SnapType.ENDPOINT:
            return self._snap_endpoint(cursor, entities, radius, plane)
        if snap_type == SnapType.MIDPOINT:
            return self._snap_midpoint(cursor, entities, radius, plane)
        if snap_type == SnapType.CENTER:
            return self._snap_center(cursor, entities, radius, plane)
        if snap_type == SnapType.NEAREST:
            return self._snap_nearest(cursor, entities, radius, plane)
        if snap_type == SnapType.TANGENT:
            return self._snap_tangent(cursor, entities, radius, plane)
        if snap_type == SnapType.INTERSECTION:
            return self._snap_intersection(cursor, entities, radius, plane)
        if snap_type == SnapType.GRID:
            return self._snap_grid(cursor, radius)
        return None, None

    # ------------------------------------------------------------------
    # World ↔ UV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _world_to_uv(world_pt: np.ndarray, plane) -> np.ndarray:
        return plane.project_point(world_pt)

    @staticmethod
    def _uv_to_world(uv: np.ndarray, plane) -> np.ndarray:
        return plane.to_3d(float(uv[0]), float(uv[1]))

    # ------------------------------------------------------------------
    # OCC curve helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _occ_adaptor(occ_edge):
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        return BRepAdaptor_Curve(occ_edge)

    def _ref_to_proxy_entities(self, ref, plane) -> list:
        """
        Convert a ReferenceEntity's occ_edges into UV-space ArcEntity /
        LineEntity proxies so intersection and tangent snap can reuse the
        same math as for drawn entities.
        """
        from cad.sketch import ArcEntity, LineEntity
        from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line
        import math
        proxies = []
        # Fallback for refs with no OCC edge (e.g. included sketch lines)
        if not ref.occ_edges:
            pts = ref.points
            if len(pts) >= 2:
                proxies.append(LineEntity(pts[0], pts[-1]))
            return proxies
        try:
            for occ_edge in ref.occ_edges:
                a = self._occ_adaptor(occ_edge)
                gt = a.GetType()
                if gt == GeomAbs_Circle:
                    circ = a.Circle()
                    loc  = circ.Location()
                    c_world = np.array([loc.X(), loc.Y(), loc.Z()], dtype=np.float64)
                    c_uv    = self._world_to_uv(c_world, plane)
                    # radius in UV space — use distance from center to start point
                    p0_3d    = a.Value(a.FirstParameter())
                    p0_world = np.array([p0_3d.X(), p0_3d.Y(), p0_3d.Z()], dtype=np.float64)
                    r_uv = float(np.linalg.norm(p0_world - c_world))
                    u0, u1 = a.FirstParameter(), a.LastParameter()
                    # Map OCC circle parameter (measured from x_axis in 3D) to
                    # UV-plane angle using start/end points
                    def _angle(t):
                        p3 = a.Value(t)
                        w  = np.array([p3.X(), p3.Y(), p3.Z()], dtype=np.float64)
                        uv = self._world_to_uv(w, plane)
                        d  = uv - c_uv
                        return math.atan2(float(d[1]), float(d[0]))
                    sa = _angle(u0)
                    ea = _angle(u1)
                    if abs(u1 - u0 - 2 * math.pi) < 1e-9:
                        ea = sa + 2 * math.pi
                    elif ea <= sa:
                        ea += 2 * math.pi
                    proxies.append(ArcEntity(
                        center=(float(c_uv[0]), float(c_uv[1])),
                        radius=r_uv,
                        start_angle=sa,
                        end_angle=ea,
                    ))
                elif gt == GeomAbs_Line:
                    p0_3d = a.Value(a.FirstParameter())
                    p1_3d = a.Value(a.LastParameter())
                    uv0 = self._world_to_uv(
                        np.array([p0_3d.X(), p0_3d.Y(), p0_3d.Z()], dtype=np.float64), plane)
                    uv1 = self._world_to_uv(
                        np.array([p1_3d.X(), p1_3d.Y(), p1_3d.Z()], dtype=np.float64), plane)
                    proxies.append(LineEntity(uv0, uv1))
        except Exception:
            pass
        return proxies

    def _curve_endpoints_uv(self, ref, plane) -> list[np.ndarray]:
        """True start/end UV points from OCC curve parameters."""
        results = []
        try:
            for occ_edge in (ref.occ_edges or []):
                a = self._occ_adaptor(occ_edge)
                for t in (a.FirstParameter(), a.LastParameter()):
                    p = a.Value(t)
                    w = np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)
                    results.append(self._world_to_uv(w, plane))
        except Exception:
            pass
        return results

    def _curve_midpoint_uv(self, ref, plane) -> list[np.ndarray]:
        """True mid-parameter UV points — NOT tessellation midpoints."""
        results = []
        try:
            for occ_edge in (ref.occ_edges or []):
                a = self._occ_adaptor(occ_edge)
                t = (a.FirstParameter() + a.LastParameter()) * 0.5
                p = a.Value(t)
                w = np.array([p.X(), p.Y(), p.Z()], dtype=np.float64)
                results.append(self._world_to_uv(w, plane))
        except Exception:
            pass
        return results

    def _curve_centers_uv(self, ref, plane) -> list[np.ndarray]:
        """Center UV for circle/arc/ellipse OCC edges."""
        results = []
        try:
            from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Ellipse
            for occ_edge in (ref.occ_edges or []):
                a = self._occ_adaptor(occ_edge)
                gt = a.GetType()
                if gt == GeomAbs_Circle:
                    loc = a.Circle().Location()
                elif gt == GeomAbs_Ellipse:
                    loc = a.Ellipse().Location()
                else:
                    continue
                w = np.array([loc.X(), loc.Y(), loc.Z()], dtype=np.float64)
                results.append(self._world_to_uv(w, plane))
        except Exception:
            pass
        return results

    def _curve_nearest_uv(self, cursor_uv: np.ndarray, ref,
                           plane) -> tuple[np.ndarray | None, float]:
        """
        Nearest point on OCC curve(s) using GeomAPI_ProjectPointOnCurve.
        Returns (uv, distance_in_uv_space) or (None, inf).
        """
        best_d  = np.inf
        best_uv = None
        try:
            from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
            from OCP.gp import gp_Pnt
            # Convert cursor UV to world 3D for the projection query
            cursor_world = self._uv_to_world(cursor_uv, plane)
            query = gp_Pnt(float(cursor_world[0]),
                           float(cursor_world[1]),
                           float(cursor_world[2]))

            for occ_edge in (ref.occ_edges or []):
                a = self._occ_adaptor(occ_edge)
                geom_curve = a.Curve().Curve()
                u0, u1 = a.FirstParameter(), a.LastParameter()
                proj = GeomAPI_ProjectPointOnCurve(query, geom_curve, u0, u1)
                if proj.NbPoints() > 0:
                    nearest = proj.NearestPoint()
                    w = np.array([nearest.X(), nearest.Y(), nearest.Z()],
                                 dtype=np.float64)
                    uv = self._world_to_uv(w, plane)
                    d  = float(np.linalg.norm(cursor_uv - uv))
                    if d < best_d:
                        best_d  = d
                        best_uv = uv
        except Exception:
            pass
        return best_uv, best_d

    # ------------------------------------------------------------------
    # Snap implementations
    # ------------------------------------------------------------------

    def _snap_endpoint(self, cursor, entities, radius, plane):
        from cad.sketch import (LineEntity, ReferenceEntity, ArcEntity,
                                PointEntity, SplineEntity)
        best_d = radius
        best_p = None
        best_i = None

        # Extra fixed points (world origin etc.) — no entity index
        for ep in self.extra_points:
            d = float(np.linalg.norm(cursor - ep))
            if d < best_d:
                best_d = d
                best_p = np.array(ep, dtype=np.float64)
                best_i = None

        for ei, ent in enumerate(entities):
            candidates = []
            if isinstance(ent, PointEntity):
                candidates = [ent.pos]
            elif isinstance(ent, LineEntity):
                candidates = [ent.p0, ent.p1]
            elif isinstance(ent, ArcEntity):
                candidates = [ent.p0, ent.p1]
            elif isinstance(ent, SplineEntity):
                # Every fit point is a snappable node; chaining usually targets
                # the open endpoints.
                candidates = list(ent.points)
            elif isinstance(ent, ReferenceEntity):
                if ent.occ_edges:
                    if _all_edges_closed(ent.occ_edges):
                        continue
                    candidates = self._curve_endpoints_uv(ent, plane)
                elif len(ent.points) >= 2:
                    p0, p1 = ent.points[0], ent.points[-1]
                    if np.linalg.norm(p1 - p0) > 1e-6:
                        candidates = [p0, p1]

            for p in candidates:
                d = float(np.linalg.norm(cursor - p))
                if d < best_d:
                    best_d = d
                    best_p = np.array(p, dtype=np.float64)
                    best_i = ei
        return best_p, best_i

    def _snap_midpoint(self, cursor, entities, radius, plane):
        from cad.sketch import LineEntity, ReferenceEntity, ArcEntity
        best_d = radius
        best_p = None
        best_i = None

        for ei, ent in enumerate(entities):
            mids = []
            if isinstance(ent, LineEntity):
                mids = [(ent.p0 + ent.p1) * 0.5]
            elif isinstance(ent, ArcEntity):
                mids = [ent.mid_point]
            elif isinstance(ent, ReferenceEntity):
                if ent.occ_edges:
                    mids = self._curve_midpoint_uv(ent, plane)
                else:
                    pts = ent.points
                    mids = [(pts[i] + pts[i+1]) * 0.5
                            for i in range(len(pts) - 1)]

            for m in mids:
                d = float(np.linalg.norm(cursor - m))
                if d < best_d:
                    best_d = d
                    best_p = np.array(m, dtype=np.float64)
                    best_i = ei
        return best_p, best_i

    def _snap_center(self, cursor, entities, radius, plane):
        from cad.sketch import ReferenceEntity, ArcEntity
        best_d = radius
        best_p = None
        best_i = None

        import math as _math
        for ei, ent in enumerate(entities):
            centers = []
            if isinstance(ent, ArcEntity):
                # Full circles: center snap only; arcs: center + endpoints
                centers = [ent.center]
            elif isinstance(ent, ReferenceEntity):
                if ent.occ_edges:
                    centers = self._curve_centers_uv(ent, plane)
                else:
                    if len(ent.points) >= 3:
                        p0, p1 = ent.points[0], ent.points[-1]
                        if np.linalg.norm(p1 - p0) < radius * 4:
                            centers = [np.mean(
                                [p.astype(np.float64) for p in ent.points],
                                axis=0)]

            for c in centers:
                d = float(np.linalg.norm(cursor - c))
                if d < best_d:
                    best_d = d
                    best_p = np.array(c, dtype=np.float64)
                    best_i = ei
        return best_p, best_i

    def _snap_nearest(self, cursor, entities, radius, plane):
        from cad.sketch import LineEntity, ReferenceEntity, ArcEntity
        best_d = radius
        best_p = None
        best_i = None

        for ei, ent in enumerate(entities):
            if isinstance(ent, LineEntity):
                p = _nearest_on_segment(cursor, ent.p0, ent.p1)
                d = float(np.linalg.norm(cursor - p))
                if d < best_d:
                    best_d = d
                    best_p = p
                    best_i = ei

            elif isinstance(ent, ArcEntity):
                p, d = _nearest_on_arc(cursor, ent)
                if d < best_d:
                    best_d = d
                    best_p = p
                    best_i = ei

            elif isinstance(ent, ReferenceEntity):
                if ent.occ_edges:
                    uv, d = self._curve_nearest_uv(cursor, ent, plane)
                    if uv is not None and d < best_d:
                        best_d = d
                        best_p = uv
                        best_i = ei
                else:
                    pts = ent.points
                    for i in range(len(pts) - 1):
                        p = _nearest_on_segment(cursor, pts[i], pts[i+1])
                        d = float(np.linalg.norm(cursor - p))
                        if d < best_d:
                            best_d = d
                            best_p = p
                            best_i = ei
        return best_p, best_i

    def _snap_intersection(self, cursor, entities, radius, plane):
        """Snap to the nearest intersection between any two entities."""
        from cad.sketch import LineEntity, ArcEntity, ReferenceEntity
        from cad.sketch_tools.trim import (
            _line_line_intersect, _line_arc_intersect, _arc_arc_intersect)
        best_d = radius
        best_p = None

        drawn = [e for e in entities if isinstance(e, (LineEntity, ArcEntity))]
        for e in entities:
            if isinstance(e, ReferenceEntity):
                drawn.extend(self._ref_to_proxy_entities(e, plane))
        for i, a in enumerate(drawn):
            for b in drawn[i+1:]:
                pts = []
                if isinstance(a, LineEntity) and isinstance(b, LineEntity):
                    for t, u in _line_line_intersect(a.p0, a.p1, b.p0, b.p1):
                        pts.append(a.p0 + t * (a.p1 - a.p0))
                elif isinstance(a, LineEntity) and isinstance(b, ArcEntity):
                    for t, _ in _line_arc_intersect(a.p0, a.p1, b):
                        pts.append(a.p0 + t * (a.p1 - a.p0))
                elif isinstance(a, ArcEntity) and isinstance(b, LineEntity):
                    for t, _ in _line_arc_intersect(b.p0, b.p1, a):
                        pts.append(b.p0 + t * (b.p1 - b.p0))
                elif isinstance(a, ArcEntity) and isinstance(b, ArcEntity):
                    for t, u in _arc_arc_intersect(a, b):
                        angle = a.start_angle + t * (a.end_angle - a.start_angle)
                        import math
                        pts.append(a.center + a.radius * np.array(
                            [math.cos(angle), math.sin(angle)]))
                for pt in pts:
                    d = float(np.linalg.norm(cursor - pt))
                    if d < best_d:
                        best_d = d; best_p = np.array(pt, dtype=np.float64)
        # Intersection point is between two entities — no single entity_idx
        return best_p, None

    def _snap_tangent(self, cursor, entities, radius, plane):
        """
        Find the tangent point on an arc/circle for a line from anchor_pt.

        When an anchor exists (line tool has a start point), compute the two
        tangent lines from anchor → circle and pick the one whose tangent point
        is closest to the cursor — this is stable because the anchor doesn't move.

        When there's no anchor yet, fall back to nearest point on arc to cursor.
        """
        from cad.sketch import ArcEntity
        import math
        best_d = np.inf
        best_p = None
        best_i = None
        sticky_r = self.snap_radius_mm * 3.0   # ~42px sticky zone around tangent points

        # Without an anchor, track the arc edge freely (nearest point).
        if self.anchor_pt is None:
            return self._snap_nearest(cursor, entities, radius, plane)
        ref_pt = self.anchor_pt

        from cad.sketch import ReferenceEntity
        arc_entities = []
        for ei, ent in enumerate(entities):
            if isinstance(ent, ArcEntity):
                arc_entities.append((ei, ent))
            elif isinstance(ent, ReferenceEntity) and ent.occ_edges:
                for proxy in self._ref_to_proxy_entities(ent, plane):
                    if isinstance(proxy, ArcEntity):
                        arc_entities.append((ei, proxy))

        for ei, ent in arc_entities:
            cx, cy = float(ent.center[0]), float(ent.center[1])
            r      = ent.radius

            # Compute tangent points from ref_pt
            dx   = float(ref_pt[0]) - cx
            dy   = float(ref_pt[1]) - cy
            dist = math.hypot(dx, dy)

            if dist < r + 1e-9:
                # Anchor inside circle — fall back to nearest on arc to cursor
                cdx = float(cursor[0]) - cx
                cdy = float(cursor[1]) - cy
                cd  = math.hypot(cdx, cdy)
                if cd < 1e-10:
                    continue
                p = ent.center + np.array([cdx, cdy]) / cd * r
                d = float(np.linalg.norm(cursor - p))
                if d < best_d:
                    best_d = d; best_p = p; best_i = ei
                continue

            angle_to_center = math.atan2(dy, dx)
            alpha = math.asin(min(1.0, r / dist))

            candidates = []
            for sign in (-1, 1):
                tang_angle = angle_to_center + sign * (math.pi / 2 - alpha)
                tp = np.array([
                    cx + r * math.cos(tang_angle),
                    cy + r * math.sin(tang_angle),
                ], dtype=np.float64)
                # Only include if on the arc's angular span
                a    = math.atan2(float(tp[1] - cy), float(tp[0] - cx))
                span = ent.end_angle - ent.start_angle
                ta   = (a - ent.start_angle) % (2 * math.pi)
                if ta <= span + 1e-6:
                    candidates.append(tp)

            if not candidates:
                continue

            if len(candidates) == 1:
                tp = candidates[0]
            else:
                # Pick whichever tangent point is on the same side of the
                # ref→center axis as the cursor.  This is stable: the cursor
                # must cross the ref→center line to flip sides, which is a
                # large deliberate movement.
                center_pt = np.array([cx, cy])
                axis = center_pt - ref_pt          # ref → center
                # Perpendicular to axis (left-hand normal)
                perp = np.array([-axis[1], axis[0]])
                cursor_side = float(np.dot(cursor - ref_pt, perp))
                tp0_side    = float(np.dot(candidates[0] - ref_pt, perp))
                # Same sign → same side
                if cursor_side * tp0_side >= 0:
                    tp = candidates[0]
                else:
                    tp = candidates[1]

            d = float(np.linalg.norm(cursor - tp))
            if d < sticky_r:
                # Within sticky zone — snap to tangent point, don't slide
                best_p = tp; best_d = 0.0; best_i = ei   # win unconditionally
                break
            else:
                # Outside sticky zone — slide freely along the arc edge
                nearest, nd = _nearest_on_arc(cursor, ent)
                if nd < best_d:
                    best_d = nd; best_p = nearest; best_i = ei

        return best_p, best_i

    def _snap_grid(self, cursor, radius):
        if self.grid_size <= 0:
            return None, None
        gs = self.grid_size
        snapped = np.array([
            round(cursor[0] / gs) * gs,
            round(cursor[1] / gs) * gs,
        ], dtype=np.float64)
        if np.linalg.norm(cursor - snapped) < radius:
            return snapped, None
        return None, None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _nearest_on_segment(p: np.ndarray, a: np.ndarray,
                         b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return np.array(a, dtype=np.float64)
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / ab_len_sq))
    return a + t * ab


def _nearest_on_arc(cursor: np.ndarray,
                    arc) -> tuple[np.ndarray, float]:
    """
    Nearest point on an ArcEntity to cursor (UV space).
    Projects cursor onto the circle, then clamps to the arc's angular range.
    """
    delta = cursor - arc.center
    dist_to_center = float(np.linalg.norm(delta))
    if dist_to_center < 1e-10:
        p = arc.p0
        return p, float(np.linalg.norm(cursor - p))

    # Project cursor onto circle
    angle = np.arctan2(float(delta[1]), float(delta[0]))

    # Clamp angle to arc range [start_angle, end_angle]
    span = arc.end_angle - arc.start_angle
    t = (angle - arc.start_angle) % (2 * np.pi)
    if t > span:
        # Outside arc — pick the closer endpoint angle
        t = span if (t - span) < (2 * np.pi - t) else 0.0
    clamped_angle = arc.start_angle + t

    p = arc.center + arc.radius * np.array(
        [np.cos(clamped_angle), np.sin(clamped_angle)], dtype=np.float64)
    return p, float(np.linalg.norm(cursor - p))


def _all_edges_closed(occ_edges: list) -> bool:
    """
    Return True if all OCC edges in the list are closed curves
    (i.e. the curve's first and last parameter meet at the same point).
    Used to suppress endpoint snap on circles so center snap fires instead.
    """
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        for edge in occ_edges:
            a = BRepAdaptor_Curve(edge)
            p0 = a.Value(a.FirstParameter())
            p1 = a.Value(a.LastParameter())
            dist = ((p0.X()-p1.X())**2 +
                    (p0.Y()-p1.Y())**2 +
                    (p0.Z()-p1.Z())**2) ** 0.5
            if dist > 1e-4:
                return False
        return True
    except Exception:
        return False
