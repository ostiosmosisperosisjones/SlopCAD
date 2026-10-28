"""
cad/sketch_tools/square.py

SquareTool — 2-point square (click one corner, click opposite corner).

Workflow:
  click 1 : set first corner (bottom-left of the square)
  click 2 : set opposite corner (top-right)  →  commit 4-line square

The square is axis-aligned in sketch (u, v) space, constructed from two
diagonally opposite points.  ESC at any point discards state without
leaving sketch mode.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool


class SquareTool(BaseTool):

    def __init__(self):
        self._p1:        np.ndarray | None = None   # first corner (click 1)
        self._cursor_2d: np.ndarray | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    # preview geometry — read by overlay
    @property
    def square_p1(self) -> np.ndarray | None:
        return self._p1

    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = snap_result.point.copy() if snap_result.point is not None else None
        # Anchor for tangent snap is the first corner
        sketch.snap.anchor_pt = self._p1

    def handle_click(self, snap_result, sketch) -> bool:
        pt = snap_result.point
        if pt is None:
            return False

        if self._p1 is None:
            self._p1 = pt.copy()
            return True

        # Second click — build and commit the square
        self._build_square(sketch)
        return True

    def _build_square(self, sketch):
        """
        Given self._p1 and self._cursor_2d (the opposite corner),
        build the square aligned with sketch axes and append 4 edges.
        """
        from cad.sketch import LineEntity, SketchConstraint
        x0, y0 = float(self._p1[0]), float(self._p1[1])
        x1, y1 = float(self._cursor_2d[0]), float(self._cursor_2d[1])

        # Square vertices in sketch (u, v) space
        a = np.array([x0, y0])   # first corner
        b = np.array([x1, y0])   # adjacent on X axis
        c = np.array([x1, y1])   # opposite corner
        d = np.array([x0, y1])   # adjacent on Y axis

        sketch.push_undo_snapshot()
        n0 = len(sketch.entities)
        for vert_a, vert_b in [(a, b), (b, c), (c, d), (d, a)]:
            ent = LineEntity(vert_a, vert_b)
            # Propagate snap metadata from endpoints
            if np.allclose(vert_a, self._p1):
                # This endpoint is the first corner — preserve snap info
                pass
            sketch.entities.append(ent)
        # Bake in "rectangle" the way Onshape does: opposite sides parallel,
        # one corner square, and one horizontal side for orientation only —
        # delete that one and it stays a rectangle that can rotate.
        if abs(x1 - x0) > 1e-9 and abs(y1 - y0) > 1e-9:
            for kind, idx in (('parallel', (n0, n0 + 2)),
                              ('parallel', (n0 + 1, n0 + 3)),
                              ('perpendicular', (n0, n0 + 1)),
                              ('horizontal', (n0,))):
                sketch.constraints.append(SketchConstraint(kind, idx, 0.0))

        self._p1 = None

    def cancel(self) -> None:
        """ESC — drop the first corner."""
        self._p1   = None
        self._cursor_2d = None
