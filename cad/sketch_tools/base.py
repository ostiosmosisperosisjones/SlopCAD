"""
cad/sketch_tools/base.py

BaseTool — interface every sketch drawing tool must implement.

A tool instance is created when the user activates that tool and discarded
when they switch tools or exit sketch mode.  All in-progress state (e.g.
the first point of a line, the center of an arc) lives on the instance.

Methods
-------
handle_click(snap_result, sketch)
    Called on every left-click while this tool is active.
    snap_result : SnapResult  — resolved cursor (already snapped)
    sketch      : SketchMode  — append entities here
    Returns True if the click was consumed.

handle_mouse_move(snap_result, sketch)
    Called on every mouse-move.  Update preview state here.

cancel()
    Called when ESC is pressed within the tool.  Reset in-progress state
    (e.g. drop the pending line start) but do NOT switch tools — the
    tool stays active.  A second ESC at SketchMode level switches to NONE.

cursor_2d
    Property — current snapped cursor position for the overlay, or None.
"""

from __future__ import annotations
import numpy as np


class BaseTool:
    """Abstract base — subclass this for every sketch tool."""

    @property
    def cursor_2d(self) -> np.ndarray | None:
        return None

    def handle_click(self, snap_result, sketch) -> bool:
        raise NotImplementedError

    def handle_mouse_move(self, snap_result, sketch) -> None:
        pass

    def on_activate(self, sketch, selected_indices: list[int]) -> None:
        """Hook for selection-driven tools (e.g. Mirror) — receives the entity
        indices that were selected when the tool was activated.  No-op by
        default."""
        pass

    def finish(self, sketch) -> bool:
        """Enter/Return pressed while this tool is active.  Return True to
        consume it (e.g. a spline commits its accumulated points); False lets
        Enter fall through to committing the whole sketch.  No-op by default."""
        return False

    def cancel(self) -> None:
        pass
