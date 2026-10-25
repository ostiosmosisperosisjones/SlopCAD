"""
cad/sketch_tools/divide.py

DivideTool — click an entity to split it at all intersections with other
entities, keeping all pieces.  Complement to TrimTool (which removes one piece).

Workflow:
  Hover: highlight the entity under cursor (red = will be divided)
  Click: split it at every intersection, replace with the pieces
  No panel needed — immediate action like trim.
"""

from __future__ import annotations
import numpy as np
from cad.sketch_tools.base import BaseTool


class DivideTool(BaseTool):

    def __init__(self):
        self._cursor_2d: np.ndarray | None = None

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return self._cursor_2d

    def handle_mouse_move(self, snap_result, sketch) -> None:
        self._cursor_2d = (snap_result.cursor_raw.copy()
                           if snap_result.cursor_raw is not None
                           else snap_result.point)

    def handle_click(self, snap_result, sketch) -> bool:
        from cad.sketch import LineEntity, ArcEntity
        from cad.sketch_tools.trim import (
            _closest_entity, _gather_split_params,
            _split_line, _split_arc)

        click_pt = snap_result.cursor_raw
        entities = sketch.entities
        drawn = [e for e in entities if isinstance(e, (LineEntity, ArcEntity))]
        if not drawn:
            return False

        target, target_idx = _closest_entity(click_pt, drawn, entities)
        if target is None:
            return False

        t_params = _gather_split_params(target, entities, target_idx)
        if not t_params:
            return False   # no intersections — nothing to divide

        if isinstance(target, LineEntity):
            pieces = _split_line(target, t_params)
        else:
            pieces = _split_arc(target, t_params)

        if len(pieces) <= 1:
            return False

        sketch.push_undo_snapshot()
        from cad.sketch import splice_entities
        splice_entities(sketch, target_idx, target_idx + 1, pieces)
        return True

    def cancel(self) -> None:
        self._cursor_2d = None
