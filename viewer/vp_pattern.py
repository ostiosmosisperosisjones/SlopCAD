"""
viewer/vp_pattern.py

Pattern tool panel wiring — mixed into Viewport.

Activation (from a keybind/toolbar) stashes the current selection and sets the
PatternTool.  The user clicks a reference (line → linear axis, point → circular
center); the viewport then opens PatternPanel.  Panel changes sync into the
tool and trigger a live preview; OK commits the copies.
"""

from __future__ import annotations


class VpPatternMixin:

    def _activate_pattern(self, mode: str):
        """Stash selected entities and activate a pattern tool (linear/circular).

        mode: "linear" | "circular".  Next click picks the reference.
        """
        from cad.sketch import SketchTool, LineEntity, ArcEntity
        if self._sketch is None:
            return
        idxs = sorted({se.entity_idx for se in self.selection.sketch_edges
                       if se.history_idx == -1
                       and se.entity_idx < len(self._sketch.entities)
                       and isinstance(self._sketch.entities[se.entity_idx],
                                      (LineEntity, ArcEntity))})
        if not idxs:
            print("[Sketch] Select lines or arcs to pattern first.")
            return
        tool_enum = (SketchTool.PATTERN_LINEAR if mode == "linear"
                     else SketchTool.PATTERN_CIRCULAR)
        self._sketch.pending_selection = idxs
        self._sketch.set_tool(tool_enum)
        self.selection.clear()
        self.selection_changed.emit()
        ref = "a line for the array direction" if mode == "linear" \
              else "a point for the center"
        print(f"[Sketch] {mode.capitalize()} pattern: click {ref} "
              f"({len(idxs)} selected).")
        self.update()

    # -- panel lifecycle -------------------------------------------------
    def _show_pattern_panel(self):
        from gui.pattern_panel import PatternPanel
        from cad.sketch_tools.pattern import PatternTool
        if getattr(self, '_pattern_panel', None) is not None:
            return
        tool = self._sketch._active_tool if self._sketch else None
        if not isinstance(tool, PatternTool):
            return
        panel = PatternPanel(tool.mode, parent=self)
        # Seed spacing from the picked line length (the tool set a sensible default).
        if tool.mode == "linear":
            panel.set_spacing_mm(tool.spacing)
        panel.changed.connect(self._on_pattern_changed)
        panel.confirmed.connect(self._on_pattern_confirmed)
        panel.cancelled.connect(self._close_pattern_panel)
        self._pattern_panel = panel
        self._position_pattern_panel()
        panel.show()
        panel.setFocus()
        self._on_pattern_changed()   # initial preview

    def _position_pattern_panel(self):
        p = getattr(self, '_pattern_panel', None)
        if p is not None:
            p.move(16, 16)

    def _close_pattern_panel(self):
        p = getattr(self, '_pattern_panel', None)
        if p is not None:
            p.close()
            self._pattern_panel = None
        if self._sketch is not None:
            self._sketch.cancel_tool()
        self.setFocus()
        self.update()

    def _on_pattern_changed(self):
        """Sync panel params into the tool and refresh the preview."""
        from cad.sketch_tools.pattern import PatternTool
        panel = getattr(self, '_pattern_panel', None)
        tool = self._sketch._active_tool if self._sketch else None
        if panel is None or not isinstance(tool, PatternTool):
            return
        tool.count = panel.count()
        if tool.mode == "linear":
            tool.spacing = panel.spacing_mm()
        else:
            tool.angle = panel.angle_rad()
        self.update()

    def _on_pattern_confirmed(self):
        from cad.sketch_tools.pattern import PatternTool
        tool = self._sketch._active_tool if self._sketch else None
        if isinstance(tool, PatternTool):
            tool.commit(self._sketch)
            self._rebuild_sketch_faces()
        self._close_pattern_panel()
