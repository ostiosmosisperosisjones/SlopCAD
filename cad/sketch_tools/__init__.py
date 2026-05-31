"""
cad/sketch_tools/__init__.py

Sketch tool registry.

Adding a new draw tool:
  1. Create cad/sketch_tools/mytool.py with a class subclassing BaseTool
  2. Import it here and add it to TOOLS
  3. Add its enum value to SketchTool in cad/sketch.py
  4. Add a keybind in cad/prefs.py KEYBIND_DEFAULTS + KEYBIND_LABELS
  5. Add a prefs.matches() call in viewer/viewport.py keyPressEvent
  That's it — snap, overlay, history, and commit all work automatically.
"""

from cad.sketch_tools.base      import BaseTool
from cad.sketch_tools.line      import LineTool
from cad.sketch_tools.arc       import Arc3Tool
from cad.sketch_tools.circle    import CircleTool, CircleMode
from cad.sketch_tools.trim      import TrimTool
from cad.sketch_tools.divide    import DivideTool
from cad.sketch_tools.point     import PointTool
from cad.sketch_tools.offset    import OffsetTool
from cad.sketch_tools.fillet    import FilletTool
from cad.sketch_tools.include   import IncludeTool
from cad.sketch_tools.dimension import DimensionTool
from cad.sketch_tools.geometric import GeometricConstraintTool
from cad.sketch_tools.square  import SquareTool
from cad.sketch_tools.mirror  import MirrorTool
from cad.sketch_tools.pattern import PatternTool, LINEAR, CIRCULAR
from cad.sketch_tools.spline  import SplineTool
from cad.sketch_tools.snap    import SnapEngine, SnapResult, SnapType

from functools import partial
from cad.sketch import SketchTool

# Maps SketchTool enum value → tool factory (called with no args on activation).
TOOLS: dict[SketchTool, type[BaseTool]] = {
    SketchTool.LINE:      LineTool,
    SketchTool.ARC3:      Arc3Tool,
    SketchTool.CIRCLE:    CircleTool,
    SketchTool.SQUARE:    SquareTool,
    SketchTool.TRIM:      TrimTool,
    SketchTool.DIVIDE:    DivideTool,
    SketchTool.POINT:     PointTool,
    SketchTool.OFFSET:    OffsetTool,
    SketchTool.FILLET:    FilletTool,
    SketchTool.DIMENSION: DimensionTool,
    SketchTool.GEOMETRIC: GeometricConstraintTool,
    SketchTool.MIRROR:    MirrorTool,
    SketchTool.PATTERN_LINEAR:   partial(PatternTool, mode=LINEAR),
    SketchTool.PATTERN_CIRCULAR: partial(PatternTool, mode=CIRCULAR),
    SketchTool.SPLINE:    SplineTool,
}

# IncludeTool is a one-shot action, not a persistent drawing tool.
# Call IncludeTool.apply(sketch, selection, meshes) directly.

__all__ = [
    "BaseTool", "LineTool", "Arc3Tool", "TrimTool", "OffsetTool", "IncludeTool",
    "DimensionTool", "GeometricConstraintTool",
    "SnapEngine", "SnapResult", "SnapType",
    "TOOLS",
]
