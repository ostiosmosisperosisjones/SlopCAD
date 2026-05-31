"""
cad/sketch_tools/pattern.py

Pattern — linear and circular arrays of selected sketch entities.

Draw-first philosophy: produces independent ("dumb") copies via the shared
transform core (cad.sketch_tools.transform).  No constraints are emitted; the
geometry stands alone.

Flow (driven from the viewport, like Mirror):
  1. user selects entities to pattern
  2. activates Pattern (linear or circular)
  3. picks a reference:
       linear   → click a line; its direction sets the array axis
       circular → click a point/vertex (or line endpoint); the center
  4. a panel sets count + spacing (linear) / count + angle span (circular),
     with a live preview; confirm commits the copies.

The tool holds the parameters and regenerates the preview copies on demand.
generate_copies() returns the list of new entities WITHOUT mutating the sketch;
the viewport draws them as a preview and appends them on confirm.
"""

from __future__ import annotations
import math
import numpy as np

from cad.sketch_tools.base import BaseTool
from cad.sketch_tools.transform import translate_fn, rotate_fn, transform_entity


LINEAR   = "linear"
CIRCULAR = "circular"


class PatternTool(BaseTool):

    # Interaction states
    STATE_PICK_REF = "pick_ref"     # waiting for axis line / center point
    STATE_PARAMS   = "params"       # reference set, panel open, live preview

    def __init__(self, mode: str = LINEAR):
        self.mode = mode                      # LINEAR | CIRCULAR
        self._cursor_2d = None
        self._src_indices: list[int] = []
        self._state = self.STATE_PICK_REF

        # Linear reference: a direction (unit) + a per-step spacing along it.
        self._direction: np.ndarray | None = None   # unit vector
        # Circular reference: a center point.
        self._center: np.ndarray | None = None

        # Parameters (panel-driven)
        self.count   = 3            # total instances including the original
        self.spacing = 10.0         # mm per step (linear)
        self.angle   = math.pi / 2  # total span radians (circular)

        # Hover candidates for the reference-pick stage.
        self.hovered_line  = None
        self.hovered_point = None

    @property
    def cursor_2d(self):
        return self._cursor_2d

    # -- activation -----------------------------------------------------
    def on_activate(self, sketch, selected_indices):
        from cad.sketch import LineEntity, ArcEntity
        self._src_indices = [i for i in selected_indices
                             if i < len(sketch.entities)
                             and isinstance(sketch.entities[i],
                                            (LineEntity, ArcEntity))]

    def has_selection(self) -> bool:
        return bool(self._src_indices)

    # -- reference picking ---------------------------------------------
    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = (snap_result.cursor_raw.copy()
                           if snap_result.cursor_raw is not None
                           else (snap_result.point.copy()
                                 if snap_result.point is not None else None))
        if self._state != self.STATE_PICK_REF:
            return
        if self.mode == LINEAR:
            self.hovered_line = self._nearest_line(sketch)
        else:
            self.hovered_point = (snap_result.point.copy()
                                  if snap_result.point is not None else None)

    def _nearest_line(self, sketch):
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
        """Pick the reference (line for linear, point for circular).

        Returns True when the reference is locked in (viewport then opens the
        params panel).  Returns False if nothing usable was under the cursor.
        """
        if self._state != self.STATE_PICK_REF or not self._src_indices:
            return False

        if self.mode == LINEAR:
            line = self._nearest_line(sketch)
            if line is None:
                return False
            d = line.p1 - line.p0
            n = float(np.linalg.norm(d))
            if n < 1e-9:
                return False
            self._direction = d / n
            # Default spacing to the picked line's length — a sensible start.
            self.spacing = n
        else:  # CIRCULAR
            pt = (snap_result.point if snap_result.point is not None
                  else self._cursor_2d)
            if pt is None:
                return False
            self._center = np.asarray(pt, dtype=np.float64).copy()

        self._state = self.STATE_PARAMS
        return True

    # -- copy generation (preview + commit) ----------------------------
    def generate_copies(self, sketch) -> list:
        """Return the new entities for the current parameters, WITHOUT mutating
        the sketch.  The original (instance 0) is not included."""
        if self._state != self.STATE_PARAMS:
            return []
        srcs = [sketch.entities[i] for i in self._src_indices
                if i < len(sketch.entities)]
        out = []
        n = max(int(self.count), 1)

        if self.mode == LINEAR and self._direction is not None:
            for k in range(1, n):
                fn = translate_fn(self._direction * (self.spacing * k))
                for e in srcs:
                    c = transform_entity(e, fn)
                    if c is not None:
                        out.append(c)

        elif self.mode == CIRCULAR and self._center is not None:
            # Distribute (n-1) extra copies across the angle span.  When the
            # span is a full turn, drop the duplicate that lands on the start.
            full = abs(self.angle % (2 * math.pi)) < 1e-9 and self.angle != 0
            denom = n if full else max(n - 1, 1)
            step = self.angle / denom
            for k in range(1, n):
                fn = rotate_fn(self._center, step * k)
                for e in srcs:
                    c = transform_entity(e, fn)
                    if c is not None:
                        out.append(c)

        return out

    def commit(self, sketch) -> bool:
        copies = self.generate_copies(sketch)
        if not copies:
            return False
        sketch.push_undo_snapshot()
        sketch.entities.extend(copies)
        self._src_indices = []
        self._state = self.STATE_PICK_REF
        return True

    def cancel(self) -> None:
        self._src_indices = []
        self._direction = None
        self._center = None
        self._state = self.STATE_PICK_REF
        self.hovered_line = None
        self.hovered_point = None
        self._cursor_2d = None
