"""
cad/units.py

Unit system for the CAD app.  All internal values are stored in millimetres.
Units are a display/input concern only — history params are always raw mm floats.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Unit table — each entry: display_suffix -> mm_per_unit
# ---------------------------------------------------------------------------

UNITS: dict[str, float] = {
    "mm":  1.0,
    "cm":  10.0,
    "m":   1000.0,
    "in":  25.4,
    "ft":  304.8,
    "deg": 1.0,   # dimensionless pass-through for angle inputs
}

# Display labels for the prefs combo
UNIT_LABELS: list[str] = list(UNITS.keys())


def to_mm(value: float, unit: str) -> float:
    """Convert a value in *unit* to millimetres."""
    return value * UNITS.get(unit, 1.0)


def from_mm(value_mm: float, unit: str) -> float:
    """Convert a millimetre value to *unit*."""
    factor = UNITS.get(unit, 1.0)
    return value_mm / factor if factor else value_mm


def format_value(value_mm: float, unit: str, decimals: int = 3) -> str:
    """Format a mm value in the given unit with the unit suffix."""
    return f"{from_mm(value_mm, unit):.{decimals}f} {unit}"


def format_op_label(operation: str, params: dict) -> str:
    """
    Build a human-readable history entry label for *operation* using the
    current default unit preference.  Always reads prefs at call time so
    labels update immediately when the user changes their unit preference.
    """
    from cad.prefs import prefs
    unit     = prefs.default_unit
    decimals = prefs.display_decimals
    if operation == "extrude":
        dist = params.get("distance", 0)
        return f"Extrude  +{format_value(dist, unit, decimals)}"
    if operation == "cut":
        dist = abs(params.get("distance", 0))
        return f"Cut  -{format_value(dist, unit, decimals)}"
    if operation == "revolve":
        angle = params.get("angle_deg", 360)
        return f"Revolve  {angle:.1f}°"
    if operation == "revolve_cut":
        angle = params.get("angle_deg", 360)
        return f"Revolve cut  {angle:.1f}°"
    if operation == "offset_plane":
        ps   = params.get("plane_source") or {}
        dist = float(ps.get("distance", 0.0)) if isinstance(ps, dict) else 0.0
        name = params.get("name") or "Plane"
        return f"{name}  ({format_value(abs(dist), unit, decimals)})"
    if operation == "loft":
        n = len(params.get("from_profiles") or
                params.get("from_sketch_ids") or [])
        return f"Loft  ({n} profiles)"
    if operation == "loft_cut":
        n = len(params.get("from_profiles") or
                params.get("from_sketch_ids") or [])
        return f"Loft cut  ({n} profiles)"
    if operation == "chamfer":
        dist  = params.get("distance", 0.0)
        angle = params.get("angle_deg", 45.0)
        return f"Chamfer  {format_value(dist, unit, decimals)} × {angle:.1f}°"
    if operation == "union":
        n = len(params.get("body_ids", []))
        return f"Union  ({n} bodies)"
    if operation == "subtract":
        n = len(params.get("body_ids", [])) - 1  # subtract first (target)
        return f"Subtract  ({n} tool(s))"
    if operation == "intersect":
        n = len(params.get("body_ids", []))
        return f"Intersect  ({n} bodies)"
    return operation.capitalize()
