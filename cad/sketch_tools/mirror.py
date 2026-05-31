"""
cad/sketch_tools/mirror.py

Mirror — reflect selected sketch entities across an axis line.

Per the draw-first philosophy this is a "dumb" copy operation: it appends
independent reflected LineEntity/ArcEntity copies.  The caller may also emit
symmetric constraints linking each source/copy pair (see emit_symmetric_*),
so a solved-and-linked mirror is available as an interchangeable option — but
the geometry stands on its own without it.

The reflection math lives here as pure functions so the mirror, and later the
linear/circular pattern tools, can share a single transform core.

Flow (driven from the viewport, like Include):
  1. user selects entities to mirror
  2. activates Mirror
  3. clicks a line (typically a construction line) → it becomes the axis
  4. reflected copies are appended; optional symmetric constraints linked
"""

from __future__ import annotations
import numpy as np


def reflect_point(p: np.ndarray, a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """Reflect 2-D point p across the infinite line through a0→a1."""
    p = np.asarray(p, dtype=np.float64)
    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    d = a1 - a0
    n2 = float(np.dot(d, d))
    if n2 < 1e-18:
        # Degenerate axis — reflect through the point a0.
        return 2.0 * a0 - p
    t = float(np.dot(p - a0, d)) / n2
    foot = a0 + t * d
    return 2.0 * foot - p


def reflect_entity(ent, a0: np.ndarray, a1: np.ndarray):
    """
    Return a new entity that is `ent` reflected across the line a0→a1.

    Lines: reflect both endpoints.
    Arcs:  reflect center and endpoints.  Reflection reverses orientation, so
           the CCW sweep flips — we rebuild start/end angles from the reflected
           endpoints and keep the arc sweeping CCW (start_angle < end_angle),
           which is the invariant ArcEntity / face-building rely on.
    """
    from cad.sketch import LineEntity, ArcEntity

    if isinstance(ent, LineEntity):
        out = LineEntity(reflect_point(ent.p0, a0, a1),
                         reflect_point(ent.p1, a0, a1),
                         construction=ent.construction)
        return out

    if isinstance(ent, ArcEntity):
        import math
        rc = reflect_point(ent.center, a0, a1)
        # Reflect the two endpoints; angles are taken from the reflected center.
        rp0 = reflect_point(ent.p0, a0, a1)
        rp1 = reflect_point(ent.p1, a0, a1)
        full = abs((ent.end_angle - ent.start_angle) - 2 * math.pi) < 1e-9

        if full:
            return ArcEntity(rc, ent.radius, 0.0, 2 * math.pi,
                             construction=ent.construction)

        # Original sweeps CCW p0→p1.  Reflection makes that traversal CW, so
        # the reflected arc sweeps CCW from rp1 to rp0.  Build CCW angles.
        a_from = math.atan2(float(rp1[1] - rc[1]), float(rp1[0] - rc[0]))
        a_to   = math.atan2(float(rp0[1] - rc[1]), float(rp0[0] - rc[0]))
        span = (a_to - a_from) % (2 * math.pi)
        if span < 1e-12:
            span = 2 * math.pi
        return ArcEntity(rc, ent.radius, a_from, a_from + span,
                         construction=ent.construction)

    # Unsupported entity types (points/references) are skipped by the caller.
    return None


def _point_on_axis(p, a0, a1, tol=1e-6) -> bool:
    """True if p lies on the infinite line a0→a1 (within tol)."""
    p = np.asarray(p, float); a0 = np.asarray(a0, float); a1 = np.asarray(a1, float)
    d = a1 - a0
    n = float(np.hypot(d[0], d[1]))
    if n < 1e-12:
        return float(np.hypot(*(p - a0))) < tol
    # perpendicular distance
    return abs(float(np.cross(d, p - a0)) / n) < tol


# ---------------------------------------------------------------------------
# MirrorTool
# ---------------------------------------------------------------------------

from cad.sketch_tools.base import BaseTool


class MirrorTool(BaseTool):
    """
    Reflect the pre-selected entities across an axis line.

    Activation captures the current selection (the entities to mirror).  The
    user then hovers/clicks a line — typically a construction line — to use as
    the mirror axis.  On click the reflected copies are appended and symmetric
    constraints link each source/copy pair.
    """

    # Auto-link mirrored geometry with symmetric constraints.  The solver
    # wiring is in place and verified, but naive per-entity emission double-
    # constrains endpoints shared between adjacent source lines, which the
    # status checker reports as over-constrained.  Until the emission
    # deduplicates shared points, default to dumb copies (geometry is exact
    # without the constraints).  Flip this on once point-dedup lands.
    EMIT_SYMMETRIC = False

    def __init__(self):
        self._cursor_2d = None
        self._src_indices: list[int] = []
        self.hovered_axis = None          # LineEntity under cursor (axis candidate)

    @property
    def cursor_2d(self):
        return self._cursor_2d

    def on_activate(self, sketch, selected_indices):
        from cad.sketch import LineEntity, ArcEntity
        self._src_indices = [i for i in selected_indices
                             if i < len(sketch.entities)
                             and isinstance(sketch.entities[i],
                                            (LineEntity, ArcEntity))]

    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = (snap_result.cursor_raw.copy()
                           if snap_result.cursor_raw is not None
                           else (snap_result.point.copy()
                                 if snap_result.point is not None else None))
        self.hovered_axis = self._nearest_line(sketch)

    def _nearest_line(self, sketch):
        """Nearest LineEntity to the cursor — the axis candidate."""
        from cad.sketch import LineEntity
        from cad.sketch_tools.snap import _nearest_on_segment
        if self._cursor_2d is None:
            return None
        best_d, best = np.inf, None
        for e in sketch.entities:
            if not isinstance(e, LineEntity):
                continue
            p = _nearest_on_segment(self._cursor_2d, e.p0, e.p1)
            d = float(np.linalg.norm(self._cursor_2d - p))
            if d < best_d:
                best_d, best = d, e
        return best

    def handle_click(self, snap_result, sketch) -> bool:
        axis = self._nearest_line(sketch)
        if axis is None or not self._src_indices:
            return False
        a0, a1 = axis.p0, axis.p1

        sketch.push_undo_snapshot()
        n_before = len(sketch.entities)
        pairs: list[tuple[int, int]] = []   # (src_idx, copy_idx)
        for si in self._src_indices:
            src = sketch.entities[si]
            # Don't duplicate geometry that lies on the axis (the axis itself,
            # or a coincident edge) — it would create a degenerate overlap.
            if _entity_on_axis(src, a0, a1):
                continue
            copy = reflect_entity(src, a0, a1)
            if copy is None:
                continue
            sketch.entities.append(copy)
            pairs.append((si, len(sketch.entities) - 1))

        if len(sketch.entities) == n_before:
            # Nothing mirrored — roll back the snapshot.
            sketch._entity_snapshots.pop()
            return False

        if self.EMIT_SYMMETRIC:
            self._emit_symmetric_constraints(sketch, pairs, axis)
        # One-shot: deactivate after committing.
        self._src_indices = []
        return True

    def _emit_symmetric_constraints(self, sketch, pairs, axis):
        """Link each source/copy pair symmetric about the axis line.

        Uses SketchConstraint('symmetric', (src_idx, copy_idx, axis_idx)).  The
        solver wires endpoint↔endpoint and center↔center symmetry.  Skipped
        silently if the axis isn't a tracked entity.
        """
        from cad.sketch import SketchConstraint
        try:
            axis_idx = sketch.entities.index(axis)
        except ValueError:
            return
        for src_idx, copy_idx in pairs:
            sketch.constraints.append(
                SketchConstraint('symmetric', (src_idx, copy_idx, axis_idx), 0.0))

    def cancel(self) -> None:
        self._src_indices = []
        self.hovered_axis = None
        self._cursor_2d = None


def _entity_on_axis(ent, a0, a1) -> bool:
    """True if the whole entity lies on the mirror axis (so reflecting it is a
    no-op that would just duplicate it)."""
    from cad.sketch import LineEntity, ArcEntity
    if isinstance(ent, LineEntity):
        return _point_on_axis(ent.p0, a0, a1) and _point_on_axis(ent.p1, a0, a1)
    if isinstance(ent, ArcEntity):
        return False
    return False
