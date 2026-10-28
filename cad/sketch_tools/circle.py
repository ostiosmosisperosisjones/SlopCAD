"""
cad/sketch_tools/circle.py

CircleTool — three sub-modes cycled by clicking the toolbar corner arrow:
  CENTER_RADIUS  : click center, click radius point
  TWO_POINT      : click two diametrically opposite points
  THREE_POINT    : click three points on the circle (reuses arc math)

A full circle is an ArcEntity with start=0, end=2π.
"""

from __future__ import annotations
import math
import numpy as np
from cad.sketch_tools.base import BaseTool
from cad.sketch_tools.arc import _circle_from_3pts


def _touch_target(sketch, snap_result) -> int | None:
    """Shift-click snapped onto a drawn line or arc (anywhere along it):
    the entity a new circle/arc should be tangent to, else None."""
    from cad.sketch import LineEntity, ArcEntity
    from cad.sketch_tools.snap import SnapType
    j = getattr(snap_result, 'entity_idx', None)
    if not sketch.snap.angle_snap_active or j is None or \
            j >= len(sketch.entities) or snap_result.type not in (
                SnapType.TANGENT, SnapType.NEAREST, SnapType.MIDPOINT):
        return None
    return j if isinstance(sketch.entities[j], (LineEntity, ArcEntity)) else None


class CircleMode:
    CENTER_RADIUS = "center_radius"
    TWO_POINT     = "two_point"
    THREE_POINT   = "three_point"

    ALL    = [CENTER_RADIUS, TWO_POINT, THREE_POINT]
    LABELS = {
        CENTER_RADIUS: "Center + Radius",
        TWO_POINT:     "2-Point",
        THREE_POINT:   "3-Point",
    }


class CircleTool(BaseTool):

    def __init__(self, mode: str = CircleMode.CENTER_RADIUS):
        self._mode       = mode
        self._pts:  list[np.ndarray] = []
        self._touch: list[int | None] = []      # per click: tangent target
        self._cursor_2d: np.ndarray | None = None

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, m: str):
        self._mode = m
        self._pts  = []
        self._touch = []

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    @property
    def pts(self) -> list[np.ndarray]:
        return self._pts

    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = (snap_result.point.copy()
                           if snap_result.point is not None else None)

    def handle_click(self, snap_result, sketch) -> bool:
        from cad.sketch import ArcEntity
        pt = snap_result.point
        if pt is None:
            return False

        self._pts.append(pt.copy())
        # Shift-click on a line/arc = "tangent to that" (not for the centre).
        is_center = self._mode == CircleMode.CENTER_RADIUS and len(self._pts) == 1
        self._touch.append(None if is_center else
                           _touch_target(sketch, snap_result))

        needed = {"center_radius": 2, "two_point": 2, "three_point": 3}[self._mode]
        if len(self._pts) < needed:
            return True

        # Enough points — commit a full circle
        circle = self._build_circle()
        if circle is not None:
            sketch.push_undo_snapshot()
            sketch.entities.append(circle)
            idx = len(sketch.entities) - 1
            from cad.sketch import SketchConstraint
            from cad.sketch_tools.line import _bake_constraints
            targets = list(dict.fromkeys(t for t in self._touch if t is not None))
            if targets:
                _bake_constraints(sketch, idx, [
                    SketchConstraint('tangent', (idx, t), 0.0) for t in targets])

        self._pts = []
        self._touch = []
        return True

    def _build_circle(self):
        from cad.sketch import ArcEntity
        TWO_PI = 2 * math.pi

        if self._mode == CircleMode.CENTER_RADIUS:
            center = self._pts[0]
            radius = float(np.linalg.norm(self._pts[1] - center))
            if radius < 1e-6:
                return None
            return ArcEntity(center, radius, 0.0, TWO_PI)

        if self._mode == CircleMode.TWO_POINT:
            center = (self._pts[0] + self._pts[1]) * 0.5
            radius = float(np.linalg.norm(self._pts[1] - self._pts[0])) * 0.5
            if radius < 1e-6:
                return None
            return ArcEntity(center, radius, 0.0, TWO_PI)

        if self._mode == CircleMode.THREE_POINT:
            result = _circle_from_3pts(self._pts[0], self._pts[1], self._pts[2])
            if result is None:
                return None
            center, radius = result
            return ArcEntity(center, radius, 0.0, TWO_PI)

        return None

    def cancel(self) -> None:
        if len(self._pts) > 0:
            self._pts.pop()   # step back one click at a time
            self._touch.pop()
        self._cursor_2d = None

    # ------------------------------------------------------------------
    # Preview geometry — read by overlay
    # ------------------------------------------------------------------

    def preview_circle(self) -> tuple[np.ndarray, float] | None:
        """Return (center, radius) for the live preview circle, or None."""
        cur = self._cursor_2d
        if cur is None or not self._pts:
            return None

        if self._mode == CircleMode.CENTER_RADIUS:
            if len(self._pts) == 1:
                center = self._pts[0]
                radius = float(np.linalg.norm(cur - center))
                return (center, radius) if radius > 1e-6 else None

        if self._mode == CircleMode.TWO_POINT:
            if len(self._pts) == 1:
                center = (self._pts[0] + cur) * 0.5
                radius = float(np.linalg.norm(cur - self._pts[0])) * 0.5
                return (center, radius) if radius > 1e-6 else None

        if self._mode == CircleMode.THREE_POINT:
            if len(self._pts) == 2:
                result = _circle_from_3pts(self._pts[0], self._pts[1], cur)
                if result is not None:
                    return result

        return None
