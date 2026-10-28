"""
cad/sketch_tools/line.py

LineTool — single-shot line segment (click start, click end).

For a continuous polyline chain, see continuous_line.py (future tool).
Each click pair produces one LineEntity.  The pending start point is
held until the second click; ESC clears it without leaving sketch mode.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool
from cad.sketch_tools.snap import SnapType


_HLINE_EXTENT = 100_000.0   # mm — large enough to behave as infinite
_INFER_TOL_DEG = 0.5        # locked ray vs. snapped-to line, for perp/parallel


def _snap_meta(snap_result) -> tuple | None:
    """Extract (entity_idx, snap_type) for snaps that need a coincidence constraint.

    ENDPOINT snaps are handled by the solver's canonicalization (points within
    1mm tolerance share a solver variable), so we skip them here. FREE and GRID
    have no source entity. Everything else — MIDPOINT, CENTER, NEAREST, TANGENT —
    lands on a position that isn't another endpoint, so the solver needs an
    explicit coincidence constraint to preserve it.
    """
    if snap_result is None:
        return None
    skip = {SnapType.FREE, SnapType.GRID, SnapType.ANGLE, SnapType.ENDPOINT, SnapType.INTERSECTION}
    if snap_result.type in skip or snap_result.entity_idx is None:
        return None
    return (snap_result.entity_idx, snap_result.type)


def _bake_constraints(sketch, idx, cands) -> list:
    """Append the inferred constraints *cands* for line *idx*, skipping any
    already implied, and solve so the drawing matches them exactly.  If the
    solve refuses, the line stays and the inferred constraints go."""
    from cad.sketch import SketchEntry
    from cad.sketch_tools.geometric import _writeback_entities
    added = []
    for c in cands:
        sketch.constraints.append(c)
        # (reads only .entities / .constraints, so works on the live sketch)
        if len(sketch.constraints) - 1 in \
                SketchEntry._implied_direction_cons(sketch):
            sketch.constraints.pop()
        else:
            added.append(c)
    if not added:
        return []
    tmp = SketchEntry.from_sketch_mode(sketch)
    if tmp.apply_constraints_local({idx}):
        _writeback_entities(sketch, tmp)
        return added
    sketch.constraints = [c for c in sketch.constraints
                          if not any(c is a for a in added)]
    return []


def _inferred(sketch, idx, start, raw_end, partners) -> list:
    """Constraints a Shift-drawn (angle-locked) line *idx* implies:
    horizontal / vertical for a 0° / 90° ray, and perpendicular / parallel
    to each line in *partners* (the lines its ends snapped onto, the previous
    line of the chain) that the locked ray is square or parallel to."""
    from cad.sketch import SketchConstraint, line_ends
    d = np.asarray(raw_end, float)[:2] - np.asarray(start, float)[:2]
    if np.linalg.norm(d) < 1e-9:
        return []
    step = sketch.snap.ANGLE_STEP_DEG
    ang = (round(np.degrees(np.arctan2(d[1], d[0])) / step) * step) % 180.0
    out = []
    if ang == 0.0:
        out.append(SketchConstraint('horizontal', (idx,), 0.0))
    elif ang == 90.0:
        out.append(SketchConstraint('vertical', (idx,), 0.0))
    seen = set()
    for j in partners:
        if j is None or j == idx or j in seen or j >= len(sketch.entities):
            continue
        seen.add(j)
        ends = line_ends(sketch.entities[j])
        if ends is None:
            continue
        e = np.asarray(ends[1], float)[:2] - np.asarray(ends[0], float)[:2]
        if np.linalg.norm(e) < 1e-9:
            continue
        rel = (ang - np.degrees(np.arctan2(e[1], e[0]))) % 180.0
        if min(rel, 180.0 - rel) < _INFER_TOL_DEG:
            out.append(SketchConstraint('parallel', (j, idx), 0.0))
        elif abs(rel - 90.0) < _INFER_TOL_DEG:
            out.append(SketchConstraint('perpendicular', (j, idx), 0.0))
    return out


def _tangent_target(sketch, snap_result) -> int | None:
    """The drawn arc/circle a TANGENT snap landed on (includes are fixed
    geometry the tangent constraint can't name), else None."""
    from cad.sketch import ArcEntity
    j = getattr(snap_result, 'entity_idx', None)
    if getattr(snap_result, 'type', None) != SnapType.TANGENT or j is None \
            or j >= len(sketch.entities):
        return None
    return j if isinstance(sketch.entities[j], ArcEntity) else None


class LineTool(BaseTool):

    def __init__(self, constrain: str | None = None):
        self._start:      np.ndarray | None = None
        self._cursor_2d:  np.ndarray | None = None
        self._constrain:  str | None = constrain  # 'H', 'V', or None
        self._start_snap: object | None = None    # SnapResult from first click
        self._prev_idx:   int | None = None       # previous line of the chain

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    def handle_mouse_move(self, snap_result, sketch) -> None:
        pt = snap_result.point
        if pt is None:
            self._cursor_2d = None
            return
        self._cursor_2d = pt.copy()
        # Keep snap engine informed of anchor so tangent snap is stable
        sketch.snap.anchor_pt = self._start

    def handle_click(self, snap_result, sketch) -> bool:
        from cad.sketch import LineEntity
        pt = snap_result.point
        if pt is None:
            return False

        if self._constrain in ('H', 'V'):
            from cad.sketch import SketchConstraint
            sketch.push_undo_snapshot()
            h = self._constrain == 'H'
            off = np.array([_HLINE_EXTENT, 0.0] if h else [0.0, _HLINE_EXTENT])
            sketch.entities.append(LineEntity(pt[:2] - off, pt[:2] + off))
            sketch.constraints.append(SketchConstraint(
                'horizontal' if h else 'vertical',
                (len(sketch.entities) - 1,), 0.0))
            return True

        if self._start is None:
            self._start = pt.copy()
            self._start_snap = snap_result
        else:
            sketch.push_undo_snapshot()
            ent = LineEntity(self._start, pt)
            ent.p0_snap = _snap_meta(self._start_snap)
            ent.p1_snap = _snap_meta(snap_result)
            sketch.entities.append(ent)
            idx = len(sketch.entities) - 1
            if sketch.snap.angle_snap_active and \
                    _tangent_target(sketch, snap_result) is not None:
                # Shift + tangent snap = "I mean tangent": the line stays
                # tangent at that point, not at the locked ray's angle.
                from cad.sketch import SketchConstraint
                _bake_constraints(sketch, idx, [SketchConstraint(
                    'tangent', (snap_result.entity_idx, idx), 0.0)])
            elif sketch.snap.angle_snap_active:
                # Shift = "I mean this angle": bake it in as constraints.
                raw = (snap_result.cursor_raw
                       if snap_result.cursor_raw is not None else pt)
                partners = [self._prev_idx,
                            getattr(self._start_snap, 'entity_idx', None),
                            snap_result.entity_idx]
                _bake_constraints(sketch, idx, _inferred(
                    sketch, idx, self._start, raw, partners))
            self._start = sketch.entities[idx].p1.copy()   # chain: end → next start
            self._start_snap = snap_result
            self._prev_idx = idx
        return True

    def cancel(self) -> None:
        """ESC within line tool — drop the pending start point."""
        self._start      = None
        self._cursor_2d  = None
        self._start_snap = None
        self._prev_idx   = None

    # ------------------------------------------------------------------
    # Read by sketch_overlay for the preview line
    # ------------------------------------------------------------------

    @property
    def line_start(self) -> np.ndarray | None:
        return self._start
