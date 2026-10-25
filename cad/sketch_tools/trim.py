"""
cad/sketch_tools/trim.py

TrimTool — click a segment of a line or arc to remove it.

Algorithm per click:
  1. Find all intersection points between the clicked entity and every other entity.
  2. Split the clicked entity at those intersection points into sub-segments/sub-arcs.
  3. Delete the sub-piece whose interior is closest to the raw click point.
  4. Replace the original entity with the surviving pieces.

Intersection math is pure 2-D (UV space) — no OCCT needed.
"""

from __future__ import annotations
import math
import numpy as np
from cad.sketch_tools.base import BaseTool


# ---------------------------------------------------------------------------
# Intersection helpers
# ---------------------------------------------------------------------------

def _line_line_intersect(p0: np.ndarray, p1: np.ndarray,
                          q0: np.ndarray, q1: np.ndarray,
                          tol: float = 1e-9
                          ) -> list[tuple[float, float]]:
    """
    Intersect segment p0-p1 with segment q0-q1.
    Returns list of (t_on_p, t_on_q) in [0,1] x [0,1].
    """
    d = p1 - p0
    e = q1 - q0
    cross = float(d[0] * e[1] - d[1] * e[0])
    if abs(cross) < tol:
        return []
    f = q0 - p0
    t = (f[0] * e[1] - f[1] * e[0]) / cross
    u = (f[0] * d[1] - f[1] * d[0]) / cross
    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        t = max(0.0, min(1.0, t))
        u = max(0.0, min(1.0, u))
        return [(t, u)]
    return []


def _line_arc_intersect(p0: np.ndarray, p1: np.ndarray,
                         arc, tol: float = 1e-9
                         ) -> list[tuple[float, float]]:
    """
    Intersect segment p0-p1 with arc.
    Returns list of (t_on_line, t_on_arc) where t_on_arc is the angle
    parameter normalised to [0,1] over the arc's angular span.
    """
    d = p1 - p0
    f = p0 - arc.center
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(f, d))
    c = float(np.dot(f, f)) - arc.radius ** 2

    disc = b * b - 4 * a * c
    if disc < 0 or a < tol:
        return []

    results = []
    for sign in (-1, 1):
        t = (-b + sign * math.sqrt(disc)) / (2 * a)
        if -tol <= t <= 1 + tol:
            pt = p0 + t * d
            angle = math.atan2(float(pt[1] - arc.center[1]),
                               float(pt[0] - arc.center[0]))
            span = arc.end_angle - arc.start_angle
            ta = (angle - arc.start_angle) % (2 * math.pi)
            if ta <= span + tol:
                ta = max(0.0, min(span, ta))
                results.append((max(0.0, min(1.0, t)), ta / span))
    return results


def _arc_arc_intersect(arc1, arc2, tol: float = 1e-9
                        ) -> list[tuple[float, float]]:
    """
    Intersect two arcs.  Returns (t_on_arc1, t_on_arc2) normalised to [0,1].
    """
    d = arc2.center - arc1.center
    dist = float(np.linalg.norm(d))
    if dist < tol:
        return []
    r1, r2 = arc1.radius, arc2.radius

    a = (r1 * r1 - r2 * r2 + dist * dist) / (2 * dist)
    h2 = r1 * r1 - a * a
    if h2 < 0:
        return []

    mid = arc1.center + (a / dist) * d
    if h2 < tol * tol:
        pts = [mid]
    else:
        h = math.sqrt(h2)
        perp = np.array([-d[1], d[0]]) / dist
        pts = [mid + h * perp, mid - h * perp]

    results = []
    for pt in pts:
        a1 = math.atan2(float(pt[1] - arc1.center[1]),
                        float(pt[0] - arc1.center[0]))
        a2 = math.atan2(float(pt[1] - arc2.center[1]),
                        float(pt[0] - arc2.center[0]))
        span1 = arc1.end_angle - arc1.start_angle
        span2 = arc2.end_angle - arc2.start_angle
        t1 = (a1 - arc1.start_angle) % (2 * math.pi)
        t2 = (a2 - arc2.start_angle) % (2 * math.pi)
        if t1 <= span1 + tol and t2 <= span2 + tol:
            t1 = max(0.0, min(span1, t1))
            t2 = max(0.0, min(span2, t2))
            results.append((t1 / span1, t2 / span2))
    return results


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _split_line(line, t_params: list[float], tol: float = 1e-6):
    """Split a LineEntity at parameter values in [0,1]. Returns list of LineEntity."""
    from cad.sketch import LineEntity
    params = sorted(set(max(0.0, min(1.0, t)) for t in t_params))
    params = [0.0] + [p for p in params if tol < p < 1.0 - tol] + [1.0]
    if len(params) < 3:
        return [line]
    segs = []
    for i in range(len(params) - 1):
        p0 = line.p0 + params[i]     * (line.p1 - line.p0)
        p1 = line.p0 + params[i + 1] * (line.p1 - line.p0)
        if np.linalg.norm(p1 - p0) > tol:
            segs.append(LineEntity(p0, p1))
    return segs if segs else [line]


def _split_arc(arc, t_params: list[float], tol: float = 1e-6):
    """Split an ArcEntity at normalised parameter values in [0,1]. Returns list of ArcEntity."""
    from cad.sketch import ArcEntity
    span = arc.end_angle - arc.start_angle
    params = sorted(set(max(0.0, min(1.0, t)) for t in t_params))
    params = [0.0] + [p for p in params if tol / span < p < 1.0 - tol / span] + [1.0]
    if len(params) < 3:
        return [arc]
    pieces = []
    for i in range(len(params) - 1):
        a0 = arc.start_angle + params[i]     * span
        a1 = arc.start_angle + params[i + 1] * span
        if abs(a1 - a0) > tol:
            pieces.append(ArcEntity(arc.center.copy(), arc.radius, a0, a1))
    return pieces if pieces else [arc]


# ---------------------------------------------------------------------------
# Piece-interior point helpers (for hit detection)
# ---------------------------------------------------------------------------

def _line_mid(line) -> np.ndarray:
    return (line.p0 + line.p1) * 0.5


def _arc_mid(arc):
    from cad.sketch import ArcEntity
    mid_a = (arc.start_angle + arc.end_angle) * 0.5
    return arc.center + arc.radius * np.array(
        [math.cos(mid_a), math.sin(mid_a)])


# ---------------------------------------------------------------------------
# TrimTool
# ---------------------------------------------------------------------------

class TrimTool(BaseTool):
    """
    Click any drawn entity segment to trim it at its intersections with
    all other entities.  The sub-piece closest to the click is removed.
    """

    def __init__(self):
        self._cursor_2d: np.ndarray | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    def handle_mouse_move(self, snap_result, sketch) -> None:
        # For trim we want the raw cursor, not a snapped point
        self._cursor_2d = (snap_result.cursor_raw.copy()
                           if snap_result.cursor_raw is not None
                           else snap_result.point)

    def handle_click(self, snap_result, sketch) -> bool:
        from cad.sketch import LineEntity, ArcEntity
        click_pt = snap_result.cursor_raw   # raw position — trim uses proximity

        entities = sketch.entities
        drawn = [e for e in entities
                 if isinstance(e, (LineEntity, ArcEntity))]
        if not drawn:
            return False

        # Find the entity closest to click_pt (by perpendicular distance)
        target, target_idx = _closest_entity(click_pt, drawn, entities)
        if target is None:
            return False

        # Collect split parameters from intersections with all other entities
        t_params = _gather_split_params(target, entities, target_idx)

        if not t_params:
            # No intersections — delete the whole entity
            sketch.push_undo_snapshot()
            from cad.sketch import splice_entities
            i = sketch.entities.index(target)
            splice_entities(sketch, i, i + 1, [])
            return True

        # Split target into pieces
        if isinstance(target, LineEntity):
            pieces = _split_line(target, t_params)
        else:
            pieces = _split_arc(target, t_params)

        if len(pieces) <= 1:
            return False

        # Find the piece the click falls on by projecting onto each piece.
        # For lines: use the parameter of the click projection on the original
        # line so that tail segments (sharing an endpoint with inner segments)
        # are disambiguated correctly even when the nearest-point distances tie.
        from cad.sketch_tools.snap import _nearest_on_segment, _nearest_on_arc

        if isinstance(target, LineEntity):
            d = target.p1 - target.p0
            len_sq = float(np.dot(d, d))
            t_click = (float(np.dot(click_pt - target.p0, d)) / len_sq
                       if len_sq > 1e-12 else 0.0)

            def piece_score_line(p):
                # parameter range of this piece on the original line
                d2 = target.p1 - target.p0
                ta = float(np.dot(p.p0 - target.p0, d2)) / len_sq
                tb = float(np.dot(p.p1 - target.p0, d2)) / len_sq
                lo, hi = min(ta, tb), max(ta, tb)
                # distance from t_click to this piece's interval
                return max(0.0, lo - t_click, t_click - hi)

            remove_idx = min(range(len(pieces)),
                             key=lambda i: piece_score_line(pieces[i]))
        else:
            def piece_nearest_dist(p):
                nearest, _ = _nearest_on_arc(click_pt, p)
                return float(np.linalg.norm(nearest - click_pt))

            remove_idx = min(range(len(pieces)),
                             key=lambda i: piece_nearest_dist(pieces[i]))

        survivors = [p for i, p in enumerate(pieces) if i != remove_idx]

        # Commit
        sketch.push_undo_snapshot()
        real_idx = entities.index(target)
        from cad.sketch import splice_entities
        splice_entities(sketch, real_idx, real_idx + 1, survivors)
        return True

    def cancel(self) -> None:
        self._cursor_2d = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _closest_entity(click_pt: np.ndarray, drawn: list, all_entities: list):
    """Return (entity, index_in_all_entities) of drawn entity closest to click."""
    from cad.sketch import LineEntity, ArcEntity
    from cad.sketch_tools.snap import _nearest_on_segment, _nearest_on_arc

    best_d = np.inf
    best   = None
    best_i = -1
    for ent in drawn:
        if isinstance(ent, LineEntity):
            p = _nearest_on_segment(click_pt, ent.p0, ent.p1)
        else:
            p, _ = _nearest_on_arc(click_pt, ent)
        d = float(np.linalg.norm(click_pt - p))
        if d < best_d:
            best_d = d
            best   = ent
            best_i = all_entities.index(ent)
    return best, best_i


def _gather_split_params(target, all_entities: list, target_idx: int,
                          tol: float = 1e-6) -> list[float]:
    """Collect normalised [0,1] split parameters on target from all intersections."""
    from cad.sketch import LineEntity, ArcEntity
    params = []
    for i, other in enumerate(all_entities):
        if i == target_idx:
            continue
        if not isinstance(other, (LineEntity, ArcEntity)):
            continue

        if isinstance(target, LineEntity) and isinstance(other, LineEntity):
            hits = _line_line_intersect(target.p0, target.p1, other.p0, other.p1)
            params.extend(t for t, _ in hits)

        elif isinstance(target, LineEntity) and isinstance(other, ArcEntity):
            hits = _line_arc_intersect(target.p0, target.p1, other)
            params.extend(t for t, _ in hits)

        elif isinstance(target, ArcEntity) and isinstance(other, LineEntity):
            hits = _line_arc_intersect(other.p0, other.p1, target)
            params.extend(u for _, u in hits)

        elif isinstance(target, ArcEntity) and isinstance(other, ArcEntity):
            hits = _arc_arc_intersect(target, other)
            params.extend(t for t, _ in hits)

    return [p for p in params if tol < p < 1.0 - tol]
