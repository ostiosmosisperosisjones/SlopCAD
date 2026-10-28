"""
cad/sketch_tools/geometric.py

GeometricConstraintTool — apply parallel, perpendicular, horizontal, or
vertical constraints.

Tab cycles through modes.  Horizontal/vertical are single-click; parallel,
perpendicular and equal take two clicks (reference line, then target line).
Either click of a two-click mode may be a straight included edge: it's
fixed geometry, so it always becomes the reference.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool
from cad.sketch_tools.dimension import _nearest_line, _pt_to_seg_dist


MODES = ['parallel', 'perpendicular', 'tangent', 'horizontal', 'vertical', 'equal']


def _writeback_entities(sketch, tmp):
    """Copy solved line and arc coordinates from tmp back to sketch."""
    from cad.sketch import LineEntity, ArcEntity
    for i, ent in enumerate(tmp.entities):
        if isinstance(ent, LineEntity):
            sketch.entities[i].p0 = ent.p0.copy()
            sketch.entities[i].p1 = ent.p1.copy()
        elif isinstance(ent, ArcEntity):
            sketch.entities[i].center      = ent.center.copy()
            sketch.entities[i].radius      = ent.radius
            sketch.entities[i].start_angle = ent.start_angle
            sketch.entities[i].end_angle   = ent.end_angle
MODE_LABELS = {
    'parallel':      'PARALLEL  (Tab→)',
    'perpendicular': 'PERP  (Tab→)',
    'tangent':       'TANGENT  (Tab→)',
    'horizontal':    'HORIZONTAL  (Tab→)',
    'vertical':      'VERTICAL  (Tab→)',
    'equal':         'EQUAL  (Tab→)',
}


def _nearest_line_or_include(pt, entities, tol: float = 5.0) -> int | None:
    """Closest drawn line or straight include within tol mm."""
    from cad.sketch import line_ends
    best_idx, best_dist = None, tol
    for i, ent in enumerate(entities):
        ends = line_ends(ent)
        if ends is None:
            continue
        d = _pt_to_seg_dist(pt, ends[0], ends[1])
        if d < best_dist:
            best_idx, best_dist = i, d
    return best_idx


class GeometricConstraintTool(BaseTool):

    def __init__(self, mode: str = 'parallel'):
        self._mode      = mode if mode in MODES else 'parallel'
        self._first_idx: int | None = None
        self._hover_idx: int | None = None
        self._cursor_2d: np.ndarray | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def first_idx(self) -> int | None:
        return self._first_idx

    @property
    def hover_entity_idx(self) -> int | None:
        return self._hover_idx

    def cycle_mode(self):
        idx = MODES.index(self._mode)
        self._mode = MODES[(idx + 1) % len(MODES)]
        self._first_idx = None

    def handle_mouse_move(self, snap_result, sketch) -> None:
        pt = snap_result.point
        if pt is None:
            self._cursor_2d = None
            self._hover_idx = None
            return
        self._cursor_2d = pt.copy()
        self._hover_idx = self._pick(pt, sketch)

    def _pick(self, pt, sketch) -> int | None:
        if self._mode in ('horizontal', 'vertical'):
            return _nearest_line(pt, sketch.entities)
        if self._mode == 'tangent':
            from cad.sketch_tools.dimension import _nearest_dimensionable
            return _nearest_dimensionable(pt, sketch.entities)
        return _nearest_line_or_include(pt, sketch.entities)

    def handle_click(self, snap_result, sketch) -> bool:
        pt = snap_result.point
        if pt is None:
            return False
        idx = self._pick(pt, sketch)
        if idx is None:
            return False

        if self._mode in ('horizontal', 'vertical'):
            from cad.sketch import SketchConstraint, SketchEntry
            sketch.push_undo_snapshot()
            sketch.constraints.append(SketchConstraint(self._mode, (idx,), 0.0))
            tmp = SketchEntry.from_sketch_mode(sketch)
            if tmp.apply_constraints_local():
                _writeback_entities(sketch, tmp)
            else:
                sketch.constraints.pop()
                sketch._entity_snapshots.pop()
            return True

        if self._first_idx is None:
            self._first_idx = idx
            return True

        if idx == self._first_idx:
            return True

        from cad.sketch import SketchConstraint, SketchEntry, ReferenceEntity
        ref_idx, mov_idx = self._first_idx, idx
        self._first_idx = None
        is_ref = lambda i: isinstance(sketch.entities[i], ReferenceEntity)
        if is_ref(ref_idx) and is_ref(mov_idx):
            return True          # both fixed: nothing for the solver to move
        if self._mode == 'tangent':
            from cad.sketch import ArcEntity
            arc = lambda i: isinstance(sketch.entities[i], ArcEntity)
            if not arc(ref_idx):
                ref_idx, mov_idx = mov_idx, ref_idx     # (arc, other)
            if not arc(ref_idx) or is_ref(mov_idx):
                print("[Sketch] Tangent needs an arc/circle and a line or arc")
                return True
        elif is_ref(mov_idx):
            # The solver treats the first index as the reference; an include
            # can't move, so it must be that one.
            ref_idx, mov_idx = mov_idx, ref_idx

        sketch.push_undo_snapshot()
        sketch.constraints.append(
            SketchConstraint(self._mode, (ref_idx, mov_idx), 0.0)
        )
        tmp = SketchEntry.from_sketch_mode(sketch)
        if tmp.apply_constraints_local():
            _writeback_entities(sketch, tmp)
        else:
            sketch.constraints.pop()
            sketch._entity_snapshots.pop()
        return True

    def cancel(self) -> None:
        self._first_idx = None
        self._hover_idx = None
        self._cursor_2d = None
