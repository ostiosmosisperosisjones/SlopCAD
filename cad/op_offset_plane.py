"""
cad/op_offset_plane.py

OffsetPlaneOp — datum plane history entry.

Offset planes carry a SketchPlaneSource (typically an OffsetPlaneSource around
a WorldPlaneSource or FacePlaneSource) and produce no shape change.  Their
parametric resolution is handled entirely by cad.plane_ref; replay just calls
plane_source.resolve() to surface any errors so the history panel can mark a
plane red when its parent face vanishes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from cad.op_base import Op

if TYPE_CHECKING:
    from cad.history import History


@dataclass
class OffsetPlaneOp(Op):
    """
    Datum plane parented to a WorldPlaneSource or a FacePlaneSource, offset
    along the parent normal by a fixed distance.

    plane_source : a SketchPlaneSource (typically OffsetPlaneSource)
    name         : human-readable identifier shown in the history panel
    """
    plane_source: Any         # cad.plane_ref.SketchPlaneSource
    name:         str = ""

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        # Resolve to surface parent-chain errors; result discarded.
        self.plane_source.resolve(history, entry_index)
        return shape

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        from cad.units import format_op_label as _lbl
        params = self.to_params()
        if extra_params:
            params.update(extra_params)
        label = _lbl("offset_plane", params)
        entry = viewport.history.push(
            label=label, operation="offset_plane", params=params,
            body_id=None, face_ref=None,
            shape_before=None, shape_after=None,
        )
        # Surface a resolve failure immediately as a red entry.
        try:
            self.plane_source.resolve(viewport.history, viewport.history.cursor)
        except Exception as ex:
            entry.error     = True
            entry.error_msg = str(ex)
        # Mid-history insert needs a full cascade so downstream entries that
        # might reference this plane re-resolve.  Tip-of-history append needs
        # nothing — no mesh changed.
        if viewport.history.is_mid_history:
            viewport._post_push_cascade(None)
        viewport.history_changed.emit()
        return None

    def to_params(self) -> dict:
        from cad.serializer import _plane_source_to_dict
        p: dict[str, Any] = {
            "plane_source": _plane_source_to_dict(self.plane_source),
        }
        if self.name:
            p["name"] = self.name
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "OffsetPlaneOp":
        from cad.serializer import _plane_source_from_dict
        return cls(
            plane_source = _plane_source_from_dict(params.get("plane_source")),
            name         = params.get("name", ""),
        )
