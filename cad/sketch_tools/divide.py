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
            _split_line, _split_arc, _attach_cut_ends)

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

        _attach_cut_ends(target, pieces, entities)
        joints = _tied_joints(target, target_idx, pieces, entities)
        sketch.push_undo_snapshot()
        from cad.sketch import splice_entities, SketchConstraint
        splice_entities(sketch, target_idx, target_idx + 1, pieces)
        # A mid/on tie at a cut point becomes a plain shared end; keep that
        # end on the original line by keeping the pieces either side of it
        # collinear (parallel + shared point) — "mid" downgrades to "on".
        for k in joints:
            sketch.constraints.append(SketchConstraint(
                'parallel', (target_idx + k, target_idx + k + 1), 0.0))
        return True

    def cancel(self) -> None:
        self._cursor_2d = None


def _tied_joints(target, target_idx, pieces, entities) -> list[int]:
    """k for each joint between line pieces k and k+1 where some line end
    was tied (MIDPOINT/NEAREST) to *target*."""
    from cad.sketch import LineEntity
    if not isinstance(target, LineEntity):
        return []
    tied = []
    for ent in entities:
        if not isinstance(ent, LineEntity):
            continue
        for which in ('p0', 'p1'):
            snap = getattr(ent, f'{which}_snap', None)
            if snap and snap[0] == target_idx and \
                    getattr(snap[1], 'name', '') in ('MIDPOINT', 'NEAREST'):
                tied.append(np.asarray(getattr(ent, which), float)[:2])
    return [k for k in range(len(pieces) - 1)
            if any(np.linalg.norm(pt - np.asarray(pieces[k].p1, float)[:2]) < 1e-3
                   for pt in tied)]
