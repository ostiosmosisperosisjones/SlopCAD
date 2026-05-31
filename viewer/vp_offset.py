"""
viewer/vp_offset.py

Offset tool panel wiring — mixed into Viewport.
"""

from __future__ import annotations


class VpOffsetMixin:

    def _activate_offset_selection(self):
        """Select-first offset: if lines/arcs are pre-selected, offset them all.

        Returns True if a pre-selection drove activation (panel opened);
        False if there was no usable selection (caller falls back to the
        normal click-to-select offset tool).
        """
        from cad.sketch import SketchTool, LineEntity, ArcEntity
        from cad.sketch_tools.offset import OffsetTool
        if self._sketch is None:
            return False
        idxs = sorted({se.entity_idx for se in self.selection.sketch_edges
                       if se.history_idx == -1
                       and se.entity_idx < len(self._sketch.entities)
                       and isinstance(self._sketch.entities[se.entity_idx],
                                      (LineEntity, ArcEntity))})
        if not idxs:
            return False
        self._sketch.pending_selection = idxs
        self._sketch.set_tool(SketchTool.OFFSET)
        self.selection.clear()
        self.selection_changed.emit()
        tool = self._sketch._active_tool
        if isinstance(tool, OffsetTool) and tool._state == OffsetTool.STATE_SELECTED:
            self._show_offset_panel()
            return True
        return False

    def _show_offset_panel(self):
        from gui.offset_panel import OffsetPanel
        if getattr(self, '_offset_panel', None) is not None:
            return
        panel = OffsetPanel(parent=self)
        panel.confirmed.connect(self._on_offset_confirmed)
        panel.cancelled.connect(self._close_offset_panel)
        panel.changed.connect(self._on_offset_changed)
        panel.flipped.connect(self._on_offset_flipped)
        self._offset_panel = panel
        self._position_offset_panel()
        panel.show()
        panel.setFocus()
        self._on_offset_changed()   # seed preview

    def _position_offset_panel(self):
        p = getattr(self, '_offset_panel', None)
        if p is None:
            return
        margin = 16
        p.move(margin, margin)

    def _close_offset_panel(self):
        p = getattr(self, '_offset_panel', None)
        if p is None:
            return
        p.close()
        self._offset_panel = None
        self._offset_preview = []
        if self._sketch is not None:
            self._sketch.cancel_tool()
        self.setFocus()
        self.update()

    def _offset_panel_dist(self):
        p = getattr(self, '_offset_panel', None)
        if p is None:
            return None
        v = p._spinbox.mm_value()
        return v if (v is not None and v > 0) else None

    def _on_offset_changed(self):
        """Refresh the ghosted offset preview from the current distance/flip."""
        from cad.sketch_tools.offset import OffsetTool
        tool = self._sketch._active_tool if self._sketch else None
        dist = self._offset_panel_dist()
        if isinstance(tool, OffsetTool) and dist is not None:
            self._offset_preview = tool.generate_offset(dist, self._sketch)
        else:
            self._offset_preview = []
        self.update()

    def _on_offset_flipped(self):
        from cad.sketch_tools.offset import OffsetTool
        tool = self._sketch._active_tool if self._sketch else None
        if isinstance(tool, OffsetTool):
            tool.flip_direction()
        self._on_offset_changed()

    def _on_offset_confirmed(self, dist_mm: float):
        from cad.sketch_tools.offset import OffsetTool
        if self._sketch is None:
            return
        tool = self._sketch._active_tool
        if not isinstance(tool, OffsetTool):
            return
        tool.apply_offset(dist_mm, self._sketch)
        self._rebuild_sketch_faces()
        self._close_offset_panel()
        self.update()
