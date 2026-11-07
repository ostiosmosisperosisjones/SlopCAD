"""
cad/sketch_tools/dimension.py

DimensionTool — length, gap, angle or diameter constraint.

  Line, same line again   → length of that line
  Line, a parallel line   → gap between them (perpendicular distance)
  Line, any other line    → angle between them (second line turns)
  Line, an arc / circle   → gap from the line to its edge
  Arc / circle            → diameter (one click)

Point picks — press C (centre), M (midpoint) or E (endpoint) before a click:
  point, point            → distance between them
  point, line (or line, point) → perpendicular distance from the point

Included geometry works everywhere a drawn line / circle does, but it is a
fixed datum: the drawn side moves, and two includes have nothing to dimension.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool


class DimensionTool(BaseTool):

    def __init__(self):
        self._cursor_2d: np.ndarray | None = None
        self._hover_idx: int | None = None
        self.first_idx: int | None = None      # first line picked
        self.first_pt:  tuple | None = None    # first point picked: (idx, which)
        self._hover_pt: tuple | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    @property
    def hover_entity_idx(self) -> int | None:
        return self._hover_idx

    def marker_points(self, sketch) -> list:
        """[(uv, picked?)] for the overlay: the picked and hovered points."""
        from cad.sketch import ref_point
        out = []
        for ref, picked in ((self.first_pt, True), (self._hover_pt, False)):
            p = ref_point(sketch.entities, *ref) if ref else None
            if p is not None:
                out.append((p, picked))
        return out

    def handle_mouse_move(self, snap_result, sketch) -> None:
        pt = snap_result.point
        if pt is None:
            self._cursor_2d = None
            self._hover_idx = None
            return
        self._cursor_2d = pt.copy()
        self._hover_pt = _point_pick(snap_result, sketch)
        self._hover_idx = None if self._hover_pt else \
            _nearest_dimensionable(pt, sketch.entities)

    def handle_click(self, snap_result, sketch) -> bool:
        from cad.sketch import ArcEntity, circle_of, is_datum, line_ends
        pt = snap_result.point
        if pt is None:
            return False
        cb = getattr(sketch, '_dimension_callback', None)
        ents = sketch.entities
        is_line = lambda i: i is not None and i < len(ents) and \
            line_ends(ents[i]) is not None
        if sketch.snap.declared_type in _snap_types():
            pick = _point_pick(snap_result, sketch)
            if pick is None:
                print("[Sketch] No point of that kind there")
                return False
            first, line = self.first_pt, self.first_idx
            self.first_pt = self.first_idx = None
            if first is not None:
                return self._point_to_point(first, pick, sketch, cb)
            if is_line(line):
                return self._point_to_line(pick, line, pick[0], sketch, cb)
            self.first_pt = pick
            return True
        idx = _nearest_dimensionable(pt, sketch.entities)
        if idx is None:
            self.first_idx = self.first_pt = None
            return False
        ent = sketch.entities[idx]
        if self.first_pt is not None:
            if is_line(idx):
                first, self.first_pt = self.first_pt, None
                return self._point_to_line(first, idx, idx, sketch, cb)
            print("[Sketch] Point to a circle: press C to use its centre")
            return True
        if circle_of(ent) is not None:
            first, self.first_idx = self.first_idx, None
            if is_line(first):
                return self._line_to_edge(first, idx, sketch, cb)
            if is_datum(ent):
                # A datum has no diameter to set; a line picked next gaps to it.
                self.first_idx = idx
                return True
            if cb:
                cb(idx, float(ent.radius * 2.0), 'arc')
            return True
        first, self.first_idx = self.first_idx, None
        if first is None or first >= len(ents):
            self.first_idx = idx
            return True
        if circle_of(ents[first]) is not None:
            return self._line_to_edge(idx, first, sketch, cb)
        if first == idx:
            if is_datum(ent):
                print(_FIXED)
            elif cb:
                cb(idx, float(np.linalg.norm(ent.p1 - ent.p0)), 'line')
            return True
        if is_datum(ents[first]) and is_datum(ent):
            print(_FIXED)
            return True
        if is_datum(ent):                     # the drawn line is the one to move
            first, idx = idx, first
        ent = ents[idx]
        gap = parallel_gap(ents[first], ent)
        if gap is None:
            from cad.sketch import SketchConstraint, ray_angle_deg
            ang = ray_angle_deg(sketch.entities,
                                SketchConstraint('angle', (first, idx), 0.0))
            if ang is None:
                print("[Sketch] Those lines are collinear — "
                      "starting over from this line")
                self.first_idx = idx
                return True
            if cb:
                cb(idx, ang, 'angle', first)
            return True
        if cb:
            cb(idx, gap, 'gap', first)
        return True

    def _line_to_edge(self, line, arc, sketch, cb) -> bool:
        from cad.sketch import (SketchConstraint, circle_of, edge_gap_ends,
                                is_datum, line_ends)
        ents = sketch.entities
        if is_datum(ents[line]) and is_datum(ents[arc]):
            print(_FIXED)
            return True
        center, radius = circle_of(ents[arc])
        p0, p1 = (np.asarray(p, float)[:2] for p in line_ends(ents[line]))
        if _dist_to_line(center, p0, p1) <= radius + 1e-6:
            print("[Sketch] The line cuts that circle — no gap to dimension")
            return True
        foot, rim = edge_gap_ends(ents,
                                  SketchConstraint('edge_gap', (line, arc), 0.0))
        if cb:
            mover, other = (line, arc) if is_datum(ents[arc]) else (arc, line)
            cb(mover, float(np.linalg.norm(foot - rim)), 'edge', other)
        return True

    def _point_to_point(self, a, b, sketch, cb) -> bool:
        from cad.sketch import is_datum, ref_point
        ents = sketch.entities
        if is_datum(ents[a[0]]) and is_datum(ents[b[0]]):
            print(_FIXED)
            return True
        pa, pb = ref_point(ents, *a), ref_point(ents, *b)
        d = float(np.linalg.norm(pa - pb))
        if d < 1e-6:
            print("[Sketch] Those points coincide — nothing to dimension")
            self.first_pt = b
            return True
        if is_datum(ents[b[0]]):              # the second point is the mover
            a, b = b, a
        if cb:
            cb(b[0], d, 'pt', ((a[0], b[0]), (a[1], b[1])))
        return True

    def _point_to_line(self, p, line, mover, sketch, cb) -> bool:
        from cad.sketch import is_datum, line_ends, ref_point
        ents = sketch.entities
        if is_datum(ents[p[0]]) and is_datum(ents[line]):
            print(_FIXED)
            return True
        q = ref_point(ents, *p)
        p0, p1 = (np.asarray(v, float)[:2] for v in line_ends(ents[line]))
        d = _dist_to_line(q, p0, p1)
        if p[0] == line or d < 1e-6:
            print("[Sketch] That point is on the line — pick another line")
            self.first_pt = p
            return True
        if is_datum(ents[mover]):
            mover = p[0] if mover == line else line
        if cb:
            cb(mover, d, 'pt', ((p[0], line), (p[1], None)))
        return True

    def cancel(self) -> None:
        self._hover_idx = None
        self._hover_pt = None
        self._cursor_2d = None
        self.first_idx = None
        self.first_pt = None


PARALLEL_TOL_DEG = 0.5

_FIXED = "[Sketch] Included geometry is fixed — dimension it to something drawn"


def _snap_types():
    from cad.sketch_tools.snap import SnapType
    return {SnapType.CENTER: 'center', SnapType.MIDPOINT: 'mid',
            SnapType.ENDPOINT: 'end'}


def _point_pick(snap_result, sketch, tol: float = 5.0):
    """(entity idx, 'center' | 'mid' | 'p0' | 'p1') for the point the
    declared snap key (C / M / E) names near the cursor, else None."""
    from cad.sketch import (LineEntity, ArcEntity, ReferenceEntity,
                            _is_full_circle, circle_of, line_ends)
    kind = _snap_types().get(sketch.snap.declared_type)
    pt = getattr(snap_result, 'cursor_raw', None)
    if pt is None:
        pt = snap_result.point
    if kind is None or pt is None:
        return None
    pt = np.asarray(pt, float)[:2]
    best, best_d = None, None
    for i, e in enumerate(sketch.entities):
        ends, circ = line_ends(e), circle_of(e)
        if kind == 'center' and circ is not None:
            # On the rim or near the centre — a big circle's centre is far.
            r = float(np.linalg.norm(pt - circ[0]))
            cand = [(min(r, abs(r - circ[1])), 'center')]
        elif kind == 'mid' and ends is not None:
            cand = [(_pt_to_seg_dist(pt, *(np.asarray(v, float)[:2] for v in ends)),
                     'mid')]
        elif kind == 'end' and isinstance(e, ReferenceEntity):
            pts = [np.asarray(e.points[k], float)[:2] for k in (0, -1)]
            if float(np.linalg.norm(pts[0] - pts[1])) < 1e-9 and len(e.points) > 1:
                continue                      # closed: a circle has no ends
            cand = [(float(np.linalg.norm(pt - q)), w)
                    for q, w in zip(pts, ('p0', 'p1'))]
        elif kind == 'end' and (isinstance(e, LineEntity) or isinstance(
                e, ArcEntity) and not _is_full_circle(e)):
            cand = [(float(np.linalg.norm(pt - getattr(e, w)[:2])), w)
                    for w in ('p0', 'p1')]
        else:
            continue
        for d, w in cand:
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = (i, w), d
    return best


def _dist_to_line(p, a, b) -> float:
    d = b - a
    n = float(np.linalg.norm(d))
    return abs(float(d[0] * (p - a)[1] - d[1] * (p - a)[0])) / max(n, 1e-12)


def parallel_gap(a, b) -> float | None:
    """Perpendicular distance between lines a and b if they are parallel
    (within PARALLEL_TOL_DEG) and apart; else None."""
    from cad.sketch import line_ends
    (a0, a1), (b0, _b1) = line_ends(a), line_ends(b)
    da, db = a1 - a0, _b1 - b0
    la, lb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    if la < 1e-9 or lb < 1e-9:
        return None
    sin = abs(float(da[0] * db[1] - da[1] * db[0])) / (la * lb)
    if sin > np.sin(np.radians(PARALLEL_TOL_DEG)):
        return None
    w = b0 - a0
    gap = abs(float(da[0] * w[1] - da[1] * w[0])) / la
    return gap if gap > 1e-6 else None


def _nearest_dimensionable(pt: np.ndarray, entities, tol: float = 5.0) -> int | None:
    """Index of the closest line or arc/circle, drawn or included, within
    tol mm, or None.  On a tie (a line drawn over an include) the drawn one."""
    from cad.sketch import circle_of, is_datum, line_ends
    best_idx  = None
    best_dist = tol
    for i, ent in enumerate(entities):
        ends, circ = line_ends(ent), circle_of(ent)
        if ends is not None:
            d = _pt_to_seg_dist(pt, *ends)
        elif circ is not None:
            d = abs(float(np.linalg.norm(pt - circ[0])) - circ[1])
        else:
            continue
        drawn_over = best_idx is not None and d <= best_dist + 1e-9 and \
            is_datum(entities[best_idx]) and not is_datum(ent)
        if d < best_dist or drawn_over:
            best_dist = d
            best_idx  = i
    return best_idx


def _nearest_line(pt: np.ndarray, entities, tol: float = 5.0) -> int | None:
    """Return index of the closest LineEntity to pt within tol mm, or None."""
    from cad.sketch import LineEntity
    best_idx  = None
    best_dist = tol
    for i, ent in enumerate(entities):
        if not isinstance(ent, LineEntity):
            continue
        d = _pt_to_seg_dist(pt, ent.p0, ent.p1)
        if d < best_dist:
            best_dist = d
            best_idx  = i
    return best_idx


def _pt_to_seg_dist(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab  = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        return float(np.linalg.norm(pt - a))
    t = float(np.dot(pt - a, ab)) / ab2
    t = max(0.0, min(1.0, t))
    return float(np.linalg.norm(pt - (a + t * ab)))
