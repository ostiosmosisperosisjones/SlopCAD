"""
viewer/vp_fillet.py

Fillet tool panel wiring — mixed into Viewport.
"""

from __future__ import annotations


class VpFilletMixin:

    def _show_fillet_panel(self):
        from gui.fillet_panel import FilletPanel
        from cad.sketch_tools.fillet import FilletTool
        if getattr(self, '_fillet_panel', None) is not None:
            return
        panel = FilletPanel(parent=self)
        panel.confirmed.connect(self._on_fillet_confirmed)
        panel.cancelled.connect(self._close_fillet_panel)
        # Pass corner so live validation works immediately
        tool = self._sketch._active_tool if self._sketch else None
        if isinstance(tool, FilletTool) and tool.selected_corner is not None:
            panel.set_corner(tool.selected_corner, tool)
        self._fillet_panel = panel
        self._position_fillet_panel()
        panel.show()
        panel.setFocus()

    def _position_fillet_panel(self):
        p = getattr(self, '_fillet_panel', None)
        if p is None:
            return
        p.move(16, 16)

    def _close_fillet_panel(self):
        p = getattr(self, '_fillet_panel', None)
        if p is None:
            return
        p.close()
        self._fillet_panel = None
        if self._sketch is not None:
            self._sketch.cancel_tool()
        self.setFocus()
        self.update()

    def _on_fillet_confirmed(self, radius_mm: float):
        from cad.sketch_tools.fillet import FilletTool
        if self._sketch is None:
            return
        tool = self._sketch._active_tool
        if not isinstance(tool, FilletTool):
            return
        sk = self._sketch
        ok = sk.checked_edit(lambda: tool.apply_fillet(radius_mm, sk))
        if sk.integrity_warning:
            self.status_message.emit(sk.integrity_warning)
        if not ok:
            print("[Fillet] Could not apply — radius may be too large for this corner.")
        self._close_fillet_panel()
        self.update()

