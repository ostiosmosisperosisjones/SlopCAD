"""
cad/sketch_tools/join.py

JoinTool — click two lines to make them one line.

The joined line runs between the two far ends, so a gap closes and a small
angle difference averages out while whatever the far ends connect to stays
put (see cad.sketch.join_lines).  Lines more than MAX_ANGLE_DEG apart aren't
joined — that's a corner, not a line.

Bare joints left between two collinear lines after a Trim are merged
automatically (cad.sketch.merge_residual_joints); this tool is for the
sloppy case.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool

MAX_ANGLE_DEG = 15.0


def _line_under(pt, entities, tol: float = 5.0) -> int | None:
    from cad.sketch import LineEntity
    from cad.sketch_tools.snap import _nearest_on_segment
    best, best_d = None, tol
    for i, e in enumerate(entities):
        if isinstance(e, LineEntity):
            d = float(np.linalg.norm(_nearest_on_segment(pt, e.p0, e.p1) - pt))
            if d < best_d:
                best, best_d = i, d
    return best


def joinable(entities, a: int, b: int) -> bool:
    """Two distinct lines within MAX_ANGLE_DEG of each other."""
    if a is None or b is None or a == b:
        return False
    da = entities[a].p1 - entities[a].p0
    db = entities[b].p1 - entities[b].p0
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    if na < 1e-9 or nb < 1e-9:
        return False
    cos = abs(float(np.dot(da, db))) / (na * nb)
    return cos >= np.cos(np.radians(MAX_ANGLE_DEG))


def joined_ends(entities, a: int, b: int):
    """(p0, p1) the joined line would have: the two far ends."""
    A, B = entities[a], entities[b]
    pairs = [(np.linalg.norm(pa - pb), qa, qb)
             for pa, qa in ((A.p0, A.p1), (A.p1, A.p0))
             for pb, qb in ((B.p0, B.p1), (B.p1, B.p0))]
    _d, qa, qb = min(pairs, key=lambda t: t[0])
    return qa, qb


class JoinTool(BaseTool):

    def __init__(self):
        self._cursor_2d: np.ndarray | None = None
        self.first: int | None = None        # read by the overlay
        self.hover: int | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    def handle_mouse_move(self, snap_result, sketch) -> None:
        pt = (snap_result.cursor_raw if snap_result.cursor_raw is not None
              else snap_result.point)
        self._cursor_2d = None if pt is None else np.asarray(pt, float).copy()
        self.hover = (None if pt is None
                      else _line_under(self._cursor_2d, sketch.entities))

    def handle_click(self, snap_result, sketch) -> bool:
        from cad.sketch import SketchEntry, join_lines
        from cad.sketch_tools.geometric import _writeback_entities
        pt = (snap_result.cursor_raw if snap_result.cursor_raw is not None
              else snap_result.point)
        if pt is None:
            return False
        idx = _line_under(np.asarray(pt, float), sketch.entities)
        if idx is None:
            return False
        if self.first is None or self.first >= len(sketch.entities):
            self.first = idx
            return True
        if idx == self.first:
            return True
        if not joinable(sketch.entities, self.first, idx):
            print(f"[Sketch] Join: lines {self.first} and {idx} are more than "
                  f"{MAX_ANGLE_DEG:g}° apart — not joined")
            self.first = None
            return True
        sketch.push_undo_snapshot()
        j = join_lines(sketch, self.first, idx)
        self.first = self.hover = None
        # Re-solve only if the averaged line may now break a constraint on it
        # (a horizontal that's 1.5° off): otherwise nothing should move.
        if any(j in c.indices for c in sketch.constraints):
            tmp = SketchEntry.from_sketch_mode(sketch)
            if tmp.apply_constraints_local({j}):
                _writeback_entities(sketch, tmp)
        return True

    def cancel(self) -> None:
        self._cursor_2d = None
        self.first = self.hover = None
