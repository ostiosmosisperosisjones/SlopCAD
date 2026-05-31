"""
cad/sketch_tools/spline.py

SplineTool — interpolating (fit-through-points) spline.

OMAX-Layout-style: click a sequence of points; the curve passes through all of
them.  Keep clicking to extend.  Finish with Enter (or by clicking near the
start point to close the loop).  Esc drops the in-progress spline but stays in
sketch mode (a second Esc at the sketch level exits) — mirrors LineTool.

  click 1..n : add fit points
  click near start (>= 3 pts) : close the loop and commit
  Enter / finish() : commit the open spline (>= 2 pts)
  Esc / cancel()   : drop in-progress points
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool


_CLOSE_PIX_TOL_MM = 1.5   # how near the start counts as "close the loop"


class SplineTool(BaseTool):

    def __init__(self):
        self._pts: list[np.ndarray] = []
        self._cursor_2d: np.ndarray | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    # preview geometry read by the overlay
    @property
    def points(self) -> list[np.ndarray]:
        return self._pts

    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = (snap_result.point.copy()
                           if snap_result.point is not None else None)
        # Anchor the snap engine to the last placed point.
        sketch.snap.anchor_pt = self._pts[-1] if self._pts else None

    def handle_click(self, snap_result, sketch) -> bool:
        pt = snap_result.point
        if pt is None:
            return False

        # Close the loop if clicking near the start (need >= 3 points).
        if (len(self._pts) >= 3 and
                float(np.linalg.norm(pt - self._pts[0])) < _CLOSE_PIX_TOL_MM):
            self._commit(sketch, closed=True)
            return True

        self._pts.append(pt.copy())
        return True

    def finish(self, sketch) -> bool:
        """Enter pressed — commit the open spline if it has enough points.

        Returns True if it consumed the Enter (so the sketch isn't committed)."""
        if len(self._pts) >= 2:
            self._commit(sketch, closed=False)
            return True
        # Not enough to make a spline — let Enter fall through (commit sketch).
        return False

    def _commit(self, sketch, closed: bool):
        from cad.sketch import SplineEntity
        if len(self._pts) < 2:
            self._pts = []
            return
        sketch.push_undo_snapshot()
        sketch.entities.append(SplineEntity(self._pts, closed=closed))
        self._pts = []
        self._cursor_2d = None
        sketch.snap.anchor_pt = None

    def cancel(self) -> None:
        """Esc within the tool — drop the in-progress spline points."""
        self._pts = []
        self._cursor_2d = None
