"""
viewer/tool_cursor.py

Pointer cursors that show the active tool: a crosshair (hotspot at its
centre) with the tool's toolbar icon as a badge at the lower right, or a
pair of scissors for Trim.  Real OS cursors, so they cost nothing per frame.

`cursor_key(vp)` names the cursor the viewport should show right now (None =
default arrow); `cursor_for(key, dpr)` builds and caches the QCursor.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QByteArray, QPointF, QRectF
from PyQt6.QtGui import QCursor, QPixmap, QPainter, QPen, QColor
from PyQt6.QtSvg import QSvgRenderer

_SIZE = 32          # logical px; Windows rescales anything larger
_HOT = 9            # crosshair centre / hotspot
_ARM = 7
_BADGE = QRectF(15, 15, 16, 16)

_SCISSORS = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <g fill="none" stroke-linecap="round">
    <g stroke="#111" stroke-width="4">
      <line x1="5" y1="2" x2="18" y2="18"/><line x1="2" y1="5" x2="18" y2="18"/>
      <line x1="13" y1="12" x2="19" y2="22"/><line x1="12" y1="13" x2="22" y2="19"/>
      <circle cx="21" cy="26" r="4"/><circle cx="26" cy="21" r="4"/>
    </g>
    <g stroke="#eeeeee" stroke-width="1.8">
      <line x1="5" y1="2" x2="15" y2="14"/><line x1="2" y1="5" x2="14" y2="15"/>
    </g>
    <g stroke="#ef9a9a" stroke-width="2">
      <line x1="13" y1="12" x2="19" y2="22"/><line x1="12" y1="13" x2="22" y2="19"/>
      <circle cx="21" cy="26" r="4"/><circle cx="26" cy="21" r="4"/>
    </g>
  </g>
</svg>"""
_SCISSORS_HOT = (3, 3)   # between the blade tips

_cache: dict = {}


def _svg_for(key: str) -> str | None:
    from gui import toolbar as tb
    if key.startswith("geo:"):
        return tb._CONSTRAINT_ICONS.get(key[4:])
    return {
        "LINE": tb._SVG_LINE, "ARC3": tb._SVG_ARC, "CIRCLE": tb._SVG_CIRCLE,
        "SQUARE": tb._SVG_SQUARE, "SPLINE": tb._SVG_SPLINE,
        "DIVIDE": tb._SVG_DIVIDE, "JOIN": tb._SVG_JOIN,
        "FILLET": tb._SVG_FILLET,
        "OFFSET": tb._SVG_OFFSET, "MIRROR": tb._SVG_MIRROR,
        "PATTERN_LINEAR": tb._SVG_PATTERN_LINEAR,
        "PATTERN_CIRCULAR": tb._SVG_PATTERN_CIRCULAR,
        "DIMENSION": tb._SVG_CONSTRAINT_DISTANCE,
        "pick:extrude": tb._SVG_EXTRUDE, "pick:revolve": tb._SVG_REVOLVE,
        "pick:fillet": tb._SVG_FILLET,
        "pick:chamfer": tb._SVG_CHAMFER, "pick:thicken": tb._SVG_THICKEN,
        "pick:plane": tb._SVG_OFFSET_PLANE,
    }.get(key)


def _render_svg(p: QPainter, svg: str, rect: QRectF):
    QSvgRenderer(QByteArray(svg.encode())).render(p, rect)


def _build(key: str, dpr: float) -> QCursor:
    pm = QPixmap(round(_SIZE * dpr), round(_SIZE * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    if key == "TRIM":
        _render_svg(p, _SCISSORS, QRectF(0, 0, _SIZE, _SIZE))
        p.end()
        return QCursor(pm, *_SCISSORS_HOT)
    c, g = _HOT + 0.5, 2          # centre on a pixel, gap around the hotspot
    arms = [(c - _ARM, c, c - g, c), (c + g, c, c + _ARM, c),
            (c, c - _ARM, c, c - g), (c, c + g, c, c + _ARM)]
    for color, width in ((QColor(0, 0, 0, 200), 3.0), (QColor(240, 240, 240), 1.0)):
        p.setPen(QPen(color, width, cap=Qt.PenCapStyle.FlatCap))
        for x0, y0, x1, y1 in arms:
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))
    svg = _svg_for(key)
    if svg:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 20, 20, 190))
        p.drawRoundedRect(_BADGE, 3, 3)
        _render_svg(p, svg, _BADGE.adjusted(1, 1, -1, -1))
    p.end()
    return QCursor(pm, _HOT, _HOT)


def cursor_for(key: str, dpr: float) -> QCursor:
    k = (key, dpr)
    cur = _cache.get(k)
    if cur is None:
        cur = _cache[k] = _build(key, dpr)
    return cur


def cursor_key(vp) -> str | None:
    """Which tool cursor the viewport should show, or None for the arrow."""
    sk = vp._sketch
    if sk is not None:
        from cad.sketch import SketchTool
        t = sk.tool
        if t in (SketchTool.NONE, SketchTool.IMPORT_IMAGE):
            return None
        if t == SketchTool.GEOMETRIC:
            return "geo:" + getattr(sk._active_tool, "mode", "")
        return t.name
    g = lambda name: getattr(vp, name, False)
    if g('_extrude_pick_active') or g('_extrude_face_active'):
        return "pick:extrude"
    if g('_revolve_axis_active') or g('_revolve_face_active'):
        return "pick:revolve"
    if g('_fillet3d_pick_edge') or g('_fillet3d_pick_face'):
        return "pick:fillet"
    if g('_chamfer_pick_edge') or g('_chamfer_pick_face'):
        return "pick:chamfer"
    if g('_offset_plane_face_active'):
        return "pick:plane"
    thicken = getattr(vp, '_thicken_panel', None)
    if thicken is not None and getattr(thicken, '_picking_face', False):
        return "pick:thicken"
    return None
