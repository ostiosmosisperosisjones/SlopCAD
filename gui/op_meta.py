"""
gui/op_meta.py

Single source of truth for operation metadata shown in the UI:
  - the keybind action name (matches cad.prefs KEYBIND_DEFAULTS, "" if none)
  - a short tooltip description (leads with the op name, like every op tooltip)
  - a longer explanation for the bottom status bar on hover

Both the toolbar tooltips (BUGFIX 02) and the status-bar help (BUGFIX 03) read
from this table so the two never drift apart.

The keybind suffix is resolved at tooltip-build time via prefs.key(action), so
tooltips reflect the user's current bindings rather than hardcoded letters.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OpMeta:
    action: str   # cad.prefs keybind action, or "" when the op has no shortcut
    short:  str   # tooltip text — should start with the op name
    long:   str   # detailed explanation for the bottom status bar


# Keyed by the toolbar's internal op name (matches the OpsToolbar buttons).
OP_META: dict[str, OpMeta] = {
    "sketch": OpMeta(
        action="",
        short="Sketch on a selected face or datum plane",
        long="Start a 2D sketch. Select a planar face or a datum plane first, "
             "then draw profiles to drive extrude, revolve, loft, and cuts.",
    ),
    "offset_plane": OpMeta(
        action="",
        short="Plane — create an offset datum plane",
        long="Create a datum plane offset from a world plane or an existing "
             "face. Use it as a construction reference or a sketch base.",
    ),
    "extrude": OpMeta(
        action="extrude",
        short="Extrude or cut a selected face / sketch profile",
        long="Push a face or sketch profile along its normal to add material, "
             "or reverse the direction to cut. Distance is set by drag or value.",
    ),
    "thicken": OpMeta(
        action="thicken",
        short="Thicken selected face(s) outward to add material",
        long="Offset one or more faces outward by a uniform thickness to add "
             "material, turning a surface into a solid wall.",
    ),
    "revolve": OpMeta(
        action="revolve",
        short="Revolve a sketch profile around an axis",
        long="Sweep a profile around a chosen axis to create a solid of "
             "revolution, or cut one away. Pick the profile, then the axis.",
    ),
    "loft": OpMeta(
        action="",
        short="Loft between two or more sketch profiles",
        long="Blend a smooth solid through two or more profiles on different "
             "planes. Add profiles in order from one end to the other.",
    ),
    "fillet": OpMeta(
        action="fillet",
        short="Fillet (round) selected edges",
        long="Round off selected edges with a radius. Each edge can take its "
             "own radius; runs in an isolated process with a timeout guard.",
    ),
    "chamfer": OpMeta(
        action="",
        short="Chamfer selected edges with distance + angle",
        long="Bevel selected edges by a distance and angle, producing a flat "
             "transition instead of a rounded one.",
    ),
    "boolean": OpMeta(
        action="",
        short="Boolean union / subtract / intersect between bodies",
        long="Combine bodies: union to merge, subtract to cut one with another, "
             "or intersect to keep only the overlapping volume.",
    ),
}


def tooltip_for(name: str) -> str:
    """Rich tooltip for op *name*: a bold heading ('<short>  (<key>)' when a key
    is bound) followed by the detailed explanation on its own line."""
    from cad.prefs import prefs
    from html import escape
    meta = OP_META.get(name)
    if meta is None:
        return ""
    key = prefs.key(meta.action) if meta.action else ""
    heading = f"{meta.short}  ({key})" if key else meta.short
    return (
        f'<p style="margin:0;"><b>{escape(heading)}</b></p>'
        f'<p style="margin:4px 0 0 0; color:#aaa;">{escape(meta.long)}</p>'
    )
