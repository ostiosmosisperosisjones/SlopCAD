"""
cad/sketch_tools/dimension.py

DimensionTool — length, gap or diameter constraint.

  Line, same line again   → length of that line
  Line, a parallel line   → gap between them (perpendicular distance)
  Line, an arc / circle   → gap from the line to its edge
  Arc / circle            → diameter (one click)

Point picks — press C (centre), M (midpoint) or E (endpoint) before a click:
  point, point            → distance between them
  point, line (or line, point) → perpendicular distance from the point
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
        from cad.sketch import LineEntity, ArcEntity
        pt = snap_result.point
        if pt is None:
            return False
        cb = getattr(sketch, '_dimension_callback', None)
        if sketch.snap.declared_type in _snap_types():
            pick = _point_pick(snap_result, sketch)
            if pick is None:
                print("[Sketch] No point of that kind there")
                return False
            first, line = self.first_pt, self.first_idx
            self.first_pt = self.first_idx = None
            if first is not None:
                return self._point_to_point(first, pick, sketch, cb)
            if line is not None:
                return self._point_to_line(pick, line, pick[0], sketch, cb)
            self.first_pt = pick
            return True
        idx = _nearest_dimensionable(pt, sketch.entities)
        if idx is None:
            self.first_idx = self.first_pt = None
            return False
        ent = sketch.entities[idx]
        if self.first_pt is not None:
            if isinstance(ent, LineEntity):
                first, self.first_pt = self.first_pt, None
                return self._point_to_line(first, idx, idx, sketch, cb)
            print("[Sketch] Point to a circle: press C to use its centre")
            return True
        if isinstance(ent, ArcEntity):
            first, self.first_idx = self.first_idx, None
            if first is not None and first < len(sketch.entities) and \
                    isinstance(sketch.entities[first], LineEntity):
                return self._line_to_edge(first, idx, sketch, cb)
            if cb:
                cb(idx, float(ent.radius * 2.0), 'arc')
            return True
        first, self.first_idx = self.first_idx, None
        if first is None or first >= len(sketch.entities):
            self.first_idx = idx
            return True
        if first == idx:
            if cb:
                cb(idx, float(np.linalg.norm(ent.p1 - ent.p0)), 'line')
            return True
        gap = parallel_gap(sketch.entities[first], ent)
        if gap is None:
            print("[Sketch] Gap needs two parallel lines (apart) — "
                  "starting over from this line")
            self.first_idx = idx
            return True
        if cb:
            cb(idx, gap, 'gap', first)
        return True

    def _line_to_edge(self, line, arc, sketch, cb) -> bool:
        from cad.sketch import SketchConstraint, edge_gap_ends
        A = sketch.entities[arc]
        L = sketch.entities[line]
        if _dist_to_line(A.center[:2], L.p0[:2], L.p1[:2]) <= A.radius + 1e-6:
            print("[Sketch] The line cuts that circle — no gap to dimension")
            return True
        foot, rim = edge_gap_ends(sketch.entities,
                                  SketchConstraint('edge_gap', (line, arc), 0.0))
        if cb:
            cb(arc, float(np.linalg.norm(foot - rim)), 'edge', line)
        return True

    def _point_to_point(self, a, b, sketch, cb) -> bool:
        from cad.sketch import ref_point
        pa, pb = ref_point(sketch.entities, *a), ref_point(sketch.entities, *b)
        d = float(np.linalg.norm(pa - pb))
        if d < 1e-6:
            print("[Sketch] Those points coincide — nothing to dimension")
            self.first_pt = b
            return True
        if cb:
            cb(b[0], d, 'pt', ((a[0], b[0]), (a[1], b[1])))
        return True

    def _point_to_line(self, p, line, mover, sketch, cb) -> bool:
        from cad.sketch import ref_point
        L = sketch.entities[line]
        q = ref_point(sketch.entities, *p)
        d = _dist_to_line(q, L.p0[:2], L.p1[:2])
        if p[0] == line or d < 1e-6:
            print("[Sketch] That point is on the line — pick another line")
            self.first_pt = p
            return True
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


def _snap_types():
    from cad.sketch_tools.snap import SnapType
    return {SnapType.CENTER: 'center', SnapType.MIDPOINT: 'mid',
            SnapType.ENDPOINT: 'end'}


def _point_pick(snap_result, sketch, tol: float = 5.0):
    """(entity idx, 'center' | 'mid' | 'p0' | 'p1') for the point the
    declared snap key (C / M / E) names near the cursor, else None."""
    from cad.sketch import LineEntity, ArcEntity, _is_full_circle
    kind = _snap_types().get(sketch.snap.declared_type)
    pt = getattr(snap_result, 'cursor_raw', None)
    if pt is None:
        pt = snap_result.point
    if kind is None or pt is None:
        return None
    pt = np.asarray(pt, float)[:2]
    best, best_d = None, None
    for i, e in enumerate(sketch.entities):
        if kind == 'center' and isinstance(e, ArcEntity):
            # On the rim or near the centre — a big circle's centre is far.
            d = min(float(np.linalg.norm(pt - e.center[:2])),
                    abs(float(np.linalg.norm(pt - e.center[:2])) - e.radius))
            cand = [(d, 'center')]
        elif kind == 'mid' and isinstance(e, LineEntity):
            cand = [(_pt_to_seg_dist(pt, e.p0[:2], e.p1[:2]), 'mid')]
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
    da, db = a.p1 - a.p0, b.p1 - b.p0
    la, lb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    if la < 1e-9 or lb < 1e-9:
        return None
    sin = abs(float(da[0] * db[1] - da[1] * db[0])) / (la * lb)
    if sin > np.sin(np.radians(PARALLEL_TOL_DEG)):
        return None
    w = b.p0 - a.p0
    gap = abs(float(da[0] * w[1] - da[1] * w[0])) / la
    return gap if gap > 1e-6 else None


def _nearest_dimensionable(pt: np.ndarray, entities, tol: float = 5.0) -> int | None:
    """Return index of the closest LineEntity or ArcEntity within tol mm, or None."""
    from cad.sketch import LineEntity, ArcEntity
    best_idx  = None
    best_dist = tol
    for i, ent in enumerate(entities):
        if isinstance(ent, LineEntity):
            d = _pt_to_seg_dist(pt, ent.p0, ent.p1)
        elif isinstance(ent, ArcEntity):
            d = abs(float(np.linalg.norm(pt - ent.center)) - ent.radius)
        else:
            continue
        if d < best_dist:
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
