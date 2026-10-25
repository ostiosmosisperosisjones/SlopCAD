"""
cad/sketch_tools/fillet.py

FilletTool — click a corner between any two entities to insert a tangent arc.

Supports: line-line, line-arc, arc-arc corners.

Algorithm (unified offset-curve approach):
  For a fillet of radius R:
    - Offset line inward by R  → parallel line
    - Offset arc  inward by R  → concentric arc, radius ± R
  Intersect the two offset curves → fillet center candidates.
  Pick the candidate closest to the corner.
  Tangent points = foot of perpendicular from center to each entity.
"""

from __future__ import annotations
import math
import numpy as np
from cad.sketch_tools.base import BaseTool


# ---------------------------------------------------------------------------
# Corner detection — shared endpoint between any two drawable entities
# ---------------------------------------------------------------------------

def _find_corner(cursor: np.ndarray, entities: list,
                 radius: float, tol: float = 1e-3) -> tuple | None:
    """
    Find nearest shared endpoint between any two LineEntity/ArcEntity within radius.
    Returns (pt, ent_a, ent_b, end_a, end_b) or None.
    end_a/b: 0=p0, 1=p1 of that entity.
    """
    from cad.sketch import LineEntity, ArcEntity
    drawn = [e for e in entities if isinstance(e, (LineEntity, ArcEntity))]

    best_d = radius
    best   = None

    for i, a in enumerate(drawn):
        for b in drawn[i+1:]:
            for ea, pa in ((0, a.p0), (1, a.p1)):
                for eb, pb in ((0, b.p0), (1, b.p1)):
                    if np.linalg.norm(pa - pb) > tol:
                        continue
                    d = float(np.linalg.norm(cursor - pa))
                    if d < best_d:
                        best_d = d
                        best   = (pa.copy(), a, b, ea, eb)
    return best


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-10 else v


def _line_dir_away(ent, end_idx: int):
    """Unit direction along line away from the corner endpoint."""
    if end_idx == 0:
        return _unit(ent.p1 - ent.p0), float(np.linalg.norm(ent.p1 - ent.p0))
    return _unit(ent.p0 - ent.p1), float(np.linalg.norm(ent.p1 - ent.p0))


def _arc_tangent_away(arc, end_idx: int):
    """
    Unit tangent direction along arc away from the corner endpoint,
    and the arc length.
    """
    pt    = arc.p0 if end_idx == 0 else arc.p1
    angle = arc.start_angle if end_idx == 0 else arc.end_angle
    # Tangent at angle θ on CCW arc: (-sin θ, cos θ)
    # Away from corner: if end_idx==0 (start), tangent points forward (+)
    #                   if end_idx==1 (end),   tangent points backward (-)
    sign  = 1.0 if end_idx == 0 else -1.0
    tang  = sign * np.array([-math.sin(angle), math.cos(angle)])
    arc_len = arc.radius * (arc.end_angle - arc.start_angle)
    return tang, arc_len


def _left_normal(v: np.ndarray) -> np.ndarray:
    """90° CCW rotation."""
    return np.array([-v[1], v[0]])


# ---------------------------------------------------------------------------
# Offset-curve intersection — finds fillet center candidates
# ---------------------------------------------------------------------------

def _line_offset(p0: np.ndarray, p1: np.ndarray,
                 offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Offset a line segment to the left by `offset`."""
    d = _unit(p1 - p0)
    n = _left_normal(d) * offset
    return p0 + n, p1 + n


def _intersect_lines_inf(p0, p1, q0, q1, tol=1e-9):
    """Intersect two infinite lines. Returns point or None if parallel."""
    d = p1 - p0; e = q1 - q0
    cross = float(d[0]*e[1] - d[1]*e[0])
    if abs(cross) < tol:
        return None
    f = q0 - p0
    t = (f[0]*e[1] - f[1]*e[0]) / cross
    return p0 + t * d


def _intersect_line_circle(p0, p1, center, r, tol=1e-9):
    """Intersect infinite line through p0,p1 with circle. Returns list of pts."""
    d = p1 - p0
    f = p0 - center
    a = float(np.dot(d, d))
    b = 2*float(np.dot(f, d))
    c = float(np.dot(f, f)) - r*r
    disc = b*b - 4*a*c
    if disc < 0 or a < tol:
        return []
    pts = []
    for sign in (-1, 1):
        t = (-b + sign*math.sqrt(disc)) / (2*a)
        pts.append(p0 + t*d)
    return pts


def _intersect_circles(c1, r1, c2, r2, tol=1e-9):
    """Intersect two circles. Returns list of pts (0, 1, or 2)."""
    d = c2 - c1
    dist = float(np.linalg.norm(d))
    if dist < tol or dist > r1+r2+tol or dist < abs(r1-r2)-tol:
        return []
    a = (r1*r1 - r2*r2 + dist*dist) / (2*dist)
    h2 = r1*r1 - a*a
    if h2 < 0:
        return []
    mid = c1 + (a/dist)*d
    if h2 < tol*tol:
        return [mid]
    h = math.sqrt(h2)
    perp = np.array([-d[1], d[0]]) / dist
    return [mid + h*perp, mid - h*perp]


def _inward_normal(ent, end_idx: int, corner_pt: np.ndarray) -> np.ndarray:
    """
    Unit normal pointing from the entity toward the interior of the corner.
    For a line: perpendicular pointing toward the other entity's side.
    For an arc: radial direction toward/away from center depending on concavity.
    """
    from cad.sketch import LineEntity, ArcEntity
    if isinstance(ent, LineEntity):
        dir_away, _ = _line_dir_away(ent, end_idx)
        # Both left and right normals are candidates; return both for the caller
        return _left_normal(dir_away)
    else:
        # Inward = toward arc center (concave side)
        return _unit(ent.center - corner_pt)


def _fillet_center_candidates(ent_a, end_a, ent_b, end_b,
                               corner_pt, R) -> list[np.ndarray]:
    """
    Find candidate fillet centers by intersecting offset curves.
    Generates candidates for all sign combinations, caller picks best.
    """
    from cad.sketch import LineEntity, ArcEntity

    def line_offsets(ent, end_idx):
        """Both offset lines (left and right of direction)."""
        dir_away, _ = _line_dir_away(ent, end_idx)
        left = _left_normal(dir_away)
        results = []
        for sign in (1, -1):
            n = left * sign * R
            results.append(('line', ent.p0 + n, ent.p1 + n))
        return results

    def arc_offsets(ent):
        """Both concentric offset arcs (inside and outside)."""
        results = []
        for r_off in (ent.radius - R, ent.radius + R):
            if r_off > 0:
                results.append(('arc', ent.center, r_off))
        return results

    def intersect_pair(a_desc, b_desc):
        at, *ad = a_desc
        bt, *bd = b_desc
        if at == 'line' and bt == 'line':
            pt = _intersect_lines_inf(ad[0], ad[1], bd[0], bd[1])
            return [pt] if pt is not None else []
        elif at == 'line' and bt == 'arc':
            return _intersect_line_circle(ad[0], ad[1], bd[0], bd[1])
        elif at == 'arc' and bt == 'line':
            return _intersect_line_circle(bd[0], bd[1], ad[0], ad[1])
        else:
            return _intersect_circles(ad[0], ad[1], bd[0], bd[1])

    a_descs = (line_offsets(ent_a, end_a) if isinstance(ent_a, LineEntity)
               else arc_offsets(ent_a))
    b_descs = (line_offsets(ent_b, end_b) if isinstance(ent_b, LineEntity)
               else arc_offsets(ent_b))

    all_cands = []
    for a_desc in a_descs:
        for b_desc in b_descs:
            all_cands.extend(intersect_pair(a_desc, b_desc))
    return all_cands


def _foot_on_line(pt, ent):
    """Nearest point on infinite line to pt, plus parameter t in [0,1]."""
    d = ent.p1 - ent.p0
    len_sq = float(np.dot(d, d))
    if len_sq < 1e-12:
        return ent.p0.copy(), 0.0
    t = float(np.dot(pt - ent.p0, d)) / len_sq
    return ent.p0 + t * d, t


def _foot_on_arc(pt, arc):
    """Nearest point on arc circle to pt, plus normalised arc parameter [0,1]."""
    delta = pt - arc.center
    dist  = float(np.linalg.norm(delta))
    if dist < 1e-10:
        return arc.p0.copy(), 0.0
    foot_pt = arc.center + delta / dist * arc.radius
    angle = math.atan2(float(foot_pt[1]-arc.center[1]),
                       float(foot_pt[0]-arc.center[0]))
    span = arc.end_angle - arc.start_angle
    t = ((angle - arc.start_angle) % (2*math.pi)) / span
    t = max(0.0, min(1.0, t))
    return foot_pt, t


# ---------------------------------------------------------------------------
# Unified compute_fillet
# ---------------------------------------------------------------------------

def compute_fillet_all(corner_pt, ent_a, end_a, ent_b, end_b,
                       radius: float, tol: float = 1e-6) -> list:
    """
    Return all valid fillet results sorted by preference (interior first).
    Each result: (fillet_center, tp_a, tp_b, start_angle, end_angle, t_a, t_b)
    """
    from cad.sketch import LineEntity, ArcEntity

    results = []

    candidates = _fillet_center_candidates(ent_a, end_a, ent_b, end_b,
                                           corner_pt, radius)
    if not candidates:
        return []

    # Determine the interior side of the corner.
    # For each line entity, the interior normal is the left-normal of the
    # away-direction that points toward the other entity.
    # We pick the candidate that is on the interior side of BOTH entities.
    candidates.sort(key=lambda p: float(np.linalg.norm(p - corner_pt)))

    # For each entity, which t-end is the corner?
    # end_idx=0 → corner at t=0, tangent point must be t in (0,1]
    # end_idx=1 → corner at t=1, tangent point must be t in [0,1)
    def _t_valid(t, end_idx):
        if not (0.0 - tol <= t <= 1.0 + tol):
            return False   # outside entity extent entirely
        if end_idx == 0:
            # Corner at t=0: tangent point must be away from corner but not at far end
            return tol < t < 1.0 - tol
        else:
            # Corner at t=1: tangent point must be before the corner but not at far end
            return tol < t < 1.0 - tol

    for center in candidates:
        if isinstance(ent_a, LineEntity):
            tp_a, t_a = _foot_on_line(center, ent_a)
        else:
            tp_a, t_a = _foot_on_arc(center, ent_a)
        if not _t_valid(t_a, end_a):
            continue

        if isinstance(ent_b, LineEntity):
            tp_b, t_b = _foot_on_line(center, ent_b)
        else:
            tp_b, t_b = _foot_on_arc(center, ent_b)
        if not _t_valid(t_b, end_b):
            continue

        # Score by proximity to corner — interior candidate is typically closer
        v_to_center = center - corner_pt

        # Verify fillet radius is approximately correct
        da = float(np.linalg.norm(center - tp_a))
        db = float(np.linalg.norm(center - tp_b))
        if abs(da - radius) > radius * 0.01 + tol:
            continue
        if abs(db - radius) > radius * 0.01 + tol:
            continue

        # Compute arc angles
        start_angle = math.atan2(float(tp_a[1]-center[1]),
                                 float(tp_a[0]-center[0]))
        end_angle   = math.atan2(float(tp_b[1]-center[1]),
                                 float(tp_b[0]-center[0]))

        span_ccw = (end_angle - start_angle) % (2*math.pi)
        span_cw  = (2*math.pi) - span_ccw

        if span_ccw < tol or span_cw < tol:
            continue

        # Both arc spans connect tp_a to tp_b with the same center/radius.
        # Store both; tag which one is interior (midpoint closer to corner).
        def _arc_mid_pt(sa, sp):
            a = sa + sp / 2.0
            return center + radius * np.array([math.cos(a), math.sin(a)])

        mid_ccw = _arc_mid_pt(start_angle, span_ccw)
        mid_cw  = _arc_mid_pt(start_angle + span_ccw, span_cw)
        d_ccw = float(np.linalg.norm(mid_ccw - corner_pt))
        d_cw  = float(np.linalg.norm(mid_cw  - corner_pt))

        # Interior arc: midpoint closer to corner; exterior: farther
        if d_ccw <= d_cw:
            sa_in, ea_in = start_angle, start_angle + span_ccw
            sa_ex, ea_ex = start_angle + span_ccw, start_angle + 2*math.pi
        else:
            sa_in, ea_in = start_angle + span_ccw, start_angle + 2*math.pi
            sa_ex, ea_ex = start_angle, start_angle + span_ccw

        duplicate = any(float(np.linalg.norm(r[0] - center)) < tol * 100
                        for r in results)
        if not duplicate:
            # Interior arc first, exterior (flip) arc second
            results.append((center, tp_a, tp_b, sa_in, ea_in, t_a, t_b))
            results.append((center, tp_a, tp_b, sa_ex, ea_ex, t_a, t_b))

    return results





def compute_fillet(corner_pt, ent_a, end_a, ent_b, end_b,
                   radius: float, tol: float = 1e-6) -> tuple | None:
    """Return the best fillet result, or None."""
    results = compute_fillet_all(corner_pt, ent_a, end_a, ent_b, end_b, radius, tol)
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Apply fillet — trim entities and insert arc
# ---------------------------------------------------------------------------

def _apply_fillet_result(corner_pt, ent_a, end_a, ent_b, end_b,
                         radius: float, result: tuple, sketch) -> bool:
    from cad.sketch import LineEntity, ArcEntity
    from cad.sketch_tools.trim import _split_line, _split_arc

    center, tp_a, tp_b, start_angle, end_angle, t_a, t_b = result

    # Recompute t_a/t_b if not provided (e.g. mirrored result)
    if t_a is None:
        if isinstance(ent_a, LineEntity):
            _, t_a = _foot_on_line(tp_a, ent_a)
        else:
            _, t_a = _foot_on_arc(tp_a, ent_a)
    if t_b is None:
        if isinstance(ent_b, LineEntity):
            _, t_b = _foot_on_line(tp_b, ent_b)
        else:
            _, t_b = _foot_on_arc(tp_b, ent_b)

    def _split(ent, t):
        if isinstance(ent, LineEntity):
            return _split_line(ent, [t])
        return _split_arc(ent, [t])

    def _keep(pieces, end_idx):
        if len(pieces) < 2:
            return pieces[0] if pieces else None
        corner_piece = 0 if end_idx == 0 else len(pieces) - 1
        survivors = [p for i, p in enumerate(pieces) if i != corner_piece]
        return survivors[0] if survivors else None

    pieces_a = _split(ent_a, t_a)
    pieces_b = _split(ent_b, t_b)
    kept_a   = _keep(pieces_a, end_a)
    kept_b   = _keep(pieces_b, end_b)
    if kept_a is None or kept_b is None:
        return False

    arc = ArcEntity(center, radius, start_angle, end_angle)

    sketch.push_undo_snapshot()
    idx_a = sketch.entities.index(ent_a)
    idx_b = sketch.entities.index(ent_b)
    hi, lo = (idx_a, idx_b) if idx_a > idx_b else (idx_b, idx_a)
    hi_kept = kept_a if idx_a > idx_b else kept_b
    lo_kept = kept_b if idx_a > idx_b else kept_a
    from cad.sketch import splice_entities
    splice_entities(sketch, hi, hi + 1, [hi_kept])
    splice_entities(sketch, lo, lo + 1, [lo_kept])
    splice_entities(sketch, lo + 1, lo + 1, [arc])
    return True


def apply_fillet(corner_pt, ent_a, end_a, ent_b, end_b,
                 radius: float, sketch) -> bool:
    """Convenience wrapper — computes and applies the best fillet."""
    result = compute_fillet(corner_pt, ent_a, end_a, ent_b, end_b, radius)
    if result is None:
        return False
    return _apply_fillet_result(corner_pt, ent_a, end_a, ent_b, end_b,
                                radius, result, sketch)


# ---------------------------------------------------------------------------
# FilletTool
# ---------------------------------------------------------------------------

class FilletTool(BaseTool):

    STATE_HOVER    = 'hover'
    STATE_SELECTED = 'selected'

    def __init__(self):
        self._cursor_2d: np.ndarray | None = None
        self._state = self.STATE_HOVER
        self.hovered_corner: tuple | None = None
        self.selected_corner: tuple | None = None
        self.preview_radius: float | None = None
        self._all_results: list = []   # all valid fillet candidates
        self._result_idx: int = 0      # which one is active

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = (snap_result.cursor_raw.copy()
                           if snap_result.cursor_raw is not None
                           else snap_result.point)
        if self._state == self.STATE_HOVER and self._cursor_2d is not None:
            self.hovered_corner = _find_corner(
                self._cursor_2d, sketch.entities,
                radius=sketch.snap.snap_radius_mm * 3)

    def handle_click(self, snap_result, sketch) -> bool:
        if self._state != self.STATE_HOVER:
            return False
        if self.hovered_corner is None:
            return False
        self.selected_corner = self.hovered_corner
        self._state = self.STATE_SELECTED
        return True

    def flip(self):
        """Cycle to the next valid fillet candidate."""
        if len(self._all_results) > 1:
            self._result_idx = (self._result_idx + 1) % len(self._all_results)

    def current_result(self) -> tuple | None:
        """Returns (fillet_result, end_a, end_b) or None."""
        if not self._all_results:
            return None
        return self._all_results[self._result_idx]

    def _refresh_results(self, radius_mm: float):
        if self.selected_corner is None:
            self._all_results = []
            return
        pt, ent_a, ent_b, end_a, end_b = self.selected_corner
        all_results = compute_fillet_all(pt, ent_a, end_a, ent_b, end_b, radius_mm)
        self._all_results = [(r, end_a, end_b) for r in all_results]
        self._result_idx = 0

    def apply_fillet(self, radius_mm: float, sketch) -> bool:
        if self.selected_corner is None:
            return False
        # Don't refresh — use whatever the user flipped to
        if not self._all_results:
            self._refresh_results(radius_mm)
        tagged = self.current_result()
        if tagged is None:
            return False
        result, end_a, end_b = tagged
        pt, ent_a, ent_b, _ea, _eb = self.selected_corner
        ok = _apply_fillet_result(pt, ent_a, end_a, ent_b, end_b,
                                  radius_mm, result, sketch)
        self._state = self.STATE_HOVER
        self.selected_corner = None
        self._all_results = []
        self.preview_radius = None
        return ok

    def cancel(self) -> None:
        self._state          = self.STATE_HOVER
        self._cursor_2d      = None
        self.hovered_corner  = None
        self.selected_corner = None
