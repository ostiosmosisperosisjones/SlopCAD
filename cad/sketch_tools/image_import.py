"""
cad/sketch_tools/image_import.py

ImageImportTool — a one-shot action (like IncludeTool), NOT a persistent
drawing tool.  It opens the trace modal, and on accept appends the fitted
LineEntity chain into the sketch at the sketch origin (top-left of the image
at UV (0,0)); the user drags/dimensions it afterward.

Call ImageImportTool.apply(sketch, parent_widget) directly from the viewport.

The heavy lifting (raster -> polylines) lives in cad/image_trace.py and is
Qt-free; this module only bridges the modal result into sketch.entities.
"""

from __future__ import annotations


class ImageImportTool:
    """One-shot: open modal, append fitted entities.  Not registered in TOOLS."""

    @staticmethod
    def apply(sketch, parent_widget=None) -> int:
        """
        Open the image-trace modal and, if accepted, append the resulting
        LineEntity chain to *sketch*.entities.

        Returns the number of entities added (0 if cancelled / nothing traced).
        The caller owns push_undo_snapshot() bookkeeping so cancel can pop it,
        mirroring the IncludeTool activation path in viewport.py.
        """
        from viewer.image_trace_modal import ImageTraceModal

        modal = ImageTraceModal(parent_widget)
        ents = modal.run()               # fitted LineEntity/ArcEntity, or None
        if not ents:
            return 0
        sketch.entities.extend(ents)
        return len(ents)
