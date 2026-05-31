"""
viewer/sketch_overlay.py

Renders the sketch grid, axes, entities, and cursor.
All colors read from cad.prefs.  All coordinates in world mm.

draw()           — active sketch session (grid + axes + entities + cursor)
draw_committed() — persistent committed SketchEntry overlay (entities only)
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import *
import math
from cad.sketch import (SketchMode, SketchTool, LineEntity, ArcEntity,
                        PointEntity, ReferenceEntity)
from cad.prefs import prefs


# ---------------------------------------------------------------------------
# Adaptive grid spacing
# ---------------------------------------------------------------------------

_GRID_STEPS = [
    0.01, 0.02, 0.05,
    0.1,  0.2,  0.5,
    1.0,  2.0,  5.0,
    10.0, 20.0, 50.0,
    100.0, 200.0, 500.0,
]

def _choose_spacing(camera_distance: float) -> float:
    target = camera_distance / 15.0
    for s in _GRID_STEPS:
        if s >= target:
            return s
    return _GRID_STEPS[-1]


# ---------------------------------------------------------------------------
# SketchOverlay
# ---------------------------------------------------------------------------

class SketchOverlay:
    """Stateless helper — call draw() every paintGL when in sketch mode."""

    def draw(self, sketch: SketchMode, camera_distance: float,
             hovered_edge=None, selection=None) -> list[dict]:
        """Returns list of label descriptors for QPainter text pass."""
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        spacing = _choose_spacing(camera_distance)
        extent  = camera_distance * 2.0

        sel_indices = set()
        if selection is not None:
            for se in selection.sketch_edges:
                if se.history_idx == -1:
                    sel_indices.add(se.entity_idx)

        self._draw_grid(sketch.plane, spacing, extent)
        self._draw_axes(sketch.plane, extent)
        self._draw_references(sketch)
        self._draw_entities(sketch, hovered_edge=hovered_edge,
                            selected_indices=sel_indices)
        self._draw_preview(sketch)
        self._draw_cursor(sketch, camera_distance)
        self._draw_snap_indicator(sketch, camera_distance)

        labels = self._draw_dimension_constraints(sketch.entities,
                                                   sketch.constraints
                                                   if hasattr(sketch, 'constraints') else [],
                                                   lambda u, v: sketch.plane.to_3d(u, v),
                                                   camera_distance)
        glPopAttrib()
        return labels

    def draw_committed(self, entry, camera_distance: float,
                       hovered_edge=None, history_idx=None,
                       selection=None, dim_dimensions: bool = False) -> list[dict]:
        """
        Draw a committed SketchEntry as a persistent overlay.
        No grid, no axes, no cursor — just the entities, slightly dimmed.
        Returns list of label descriptors for QPainter text pass.
        """
        from cad.sketch import SketchEntry
        if not isinstance(entry, SketchEntry) or not entry.visible:
            return []

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        sel_indices = set()
        if selection is not None and history_idx is not None:
            for se in selection.sketch_edges:
                if se.history_idx == history_idx:
                    sel_indices.add(se.entity_idx)

        self._draw_committed_references(entry)
        self._draw_committed_lines(entry, hovered_edge=hovered_edge,
                                   history_idx=history_idx,
                                   selected_indices=sel_indices)

        labels = self._draw_dimension_constraints(
            entry.entities,
            entry.constraints if hasattr(entry, 'constraints') else [],
            lambda u, v: entry.plane_origin + u * entry.plane_x_axis + v * entry.plane_y_axis,
            camera_distance,
            dimmed=dim_dimensions,
        )
        glPopAttrib()
        return labels

    # ------------------------------------------------------------------
    # Helpers shared by active sketch
    # ------------------------------------------------------------------

    def _pt(self, plane, u, v):
        p = plane.to_3d(u, v)
        return (float(p[0]), float(p[1]), float(p[2]))

    def _pt_from_entry(self, entry, u, v):
        """Convert (u, v) sketch coords to world 3D using SketchEntry axes."""
        p = entry.plane_origin + u * entry.plane_x_axis + v * entry.plane_y_axis
        return (float(p[0]), float(p[1]), float(p[2]))

    def _draw_grid(self, plane, spacing, extent):
        n  = int(extent / spacing) + 1
        nm = int(extent / (spacing * 5)) + 1

        glEnable(GL_BLEND)
        glBegin(GL_LINES)

        r, g, b = prefs.sketch_grid_minor_color
        glColor4f(r, g, b, 0.35)
        for i in range(-n, n + 1):
            t = i * spacing
            if abs(t) < spacing * 0.01:
                continue
            glVertex3f(*self._pt(plane,  t,      -extent))
            glVertex3f(*self._pt(plane,  t,       extent))
            glVertex3f(*self._pt(plane, -extent,  t))
            glVertex3f(*self._pt(plane,  extent,  t))

        major = spacing * 5.0
        r, g, b = prefs.sketch_grid_major_color
        glColor4f(r, g, b, 0.70)
        for i in range(-nm, nm + 1):
            t = i * major
            if abs(t) < major * 0.01:
                continue
            glVertex3f(*self._pt(plane,  t,      -extent))
            glVertex3f(*self._pt(plane,  t,       extent))
            glVertex3f(*self._pt(plane, -extent,  t))
            glVertex3f(*self._pt(plane,  extent,  t))

        glEnd()
        glDisable(GL_BLEND)

    def _draw_axes(self, plane, extent):
        glLineWidth(1.8)
        glBegin(GL_LINES)
        glColor3f(*prefs.sketch_axis_x_color)
        glVertex3f(*self._pt(plane, -extent, 0))
        glVertex3f(*self._pt(plane,  extent, 0))
        glColor3f(*prefs.sketch_axis_y_color)
        glVertex3f(*self._pt(plane, 0, -extent))
        glVertex3f(*self._pt(plane, 0,  extent))
        glEnd()
        glLineWidth(1.0)

    def _draw_references(self, sketch: SketchMode):
        refs = [e for e in sketch.entities if isinstance(e, ReferenceEntity)]
        if not refs:
            return
        r, g, b = prefs.sketch_reference_color
        glEnable(GL_BLEND)
        glColor4f(r, g, b, 0.75)
        glLineWidth(prefs.sketch_reference_width)
        for ref in refs:
            if len(ref.points) == 1:
                p  = ref.points[0]
                cr = 0.8
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, p[0] - cr, p[1]))
                glVertex3f(*self._pt(sketch.plane, p[0] + cr, p[1]))
                glVertex3f(*self._pt(sketch.plane, p[0], p[1] - cr))
                glVertex3f(*self._pt(sketch.plane, p[0], p[1] + cr))
                glEnd()
            else:
                # If we have an OCC edge (e.g. arc from another sketch), tessellate
                # it in world space and project — ref.points only has endpoints.
                occ_pts = None
                if ref.occ_edges:
                    try:
                        from OCP.BRepAdaptor import BRepAdaptor_Curve
                        from OCP.GCPnts import GCPnts_QuasiUniformAbscissa
                        pts_3d = []
                        for occ_edge in ref.occ_edges:
                            adp = BRepAdaptor_Curve(occ_edge)
                            sampler = GCPnts_QuasiUniformAbscissa()
                            sampler.Initialize(adp, 65)
                            for k in range(1, sampler.NbPoints() + 1):
                                p3 = adp.Value(sampler.Parameter(k))
                                pts_3d.append(sketch.plane.project_point(
                                    np.array([p3.X(), p3.Y(), p3.Z()])))
                        occ_pts = pts_3d
                    except Exception:
                        pass
                draw_pts = occ_pts if occ_pts else ref.points
                glBegin(GL_LINE_STRIP)
                for p in draw_pts:
                    glVertex3f(*self._pt(sketch.plane, p[0], p[1]))
                glEnd()
        glLineWidth(1.0)
        glDisable(GL_BLEND)

    def _draw_entities(self, sketch: SketchMode, hovered_edge=None,
                       selected_indices=None):
        from viewer.hover import parse_sketch_key
        hov_entity_idx = None
        if hovered_edge is not None:
            hov_body, _ = hovered_edge
            if hov_body is not None:
                sk = parse_sketch_key(hov_body)
                if sk is not None and sk[0] == -1:
                    hov_entity_idx = sk[1]
        sel = selected_indices or set()

        # Draw construction points
        points = [e for e in sketch.entities if isinstance(e, PointEntity)]
        if points:
            glColor3f(*prefs.sketch_line_color)
            for pt in points:
                u, v = float(pt.pos[0]), float(pt.pos[1])
                cr = 0.6
                glLineWidth(1.5)
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, u - cr, v))
                glVertex3f(*self._pt(sketch.plane, u + cr, v))
                glVertex3f(*self._pt(sketch.plane, u, v - cr))
                glVertex3f(*self._pt(sketch.plane, u, v + cr))
                glEnd()
                glPointSize(3.0)
                glBegin(GL_POINTS)
                glVertex3f(*self._pt(sketch.plane, u, v))
                glEnd()
                glPointSize(1.0)

        glLineWidth(prefs.sketch_line_width)
        for j, ent in enumerate(sketch.entities):
            is_construction = getattr(ent, "construction", False)
            if j in sel:
                glColor3f(*prefs.edge_selected_color)
            elif j == hov_entity_idx:
                glColor3f(*prefs.edge_hovered_color)
            elif is_construction:
                # Dimmed so construction geometry reads as reference-only.
                r, g, b = prefs.sketch_line_color
                glColor3f(r * 0.6, g * 0.6, b * 0.6)
            else:
                glColor3f(*prefs.sketch_line_color)
            # Dashed line style for construction geometry (unless highlighted).
            stipple = is_construction and j not in sel and j != hov_entity_idx
            if stipple:
                glEnable(GL_LINE_STIPPLE)
                glLineStipple(2, 0x00FF)
            if isinstance(ent, LineEntity):
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, ent.p0[0], ent.p0[1]))
                glVertex3f(*self._pt(sketch.plane, ent.p1[0], ent.p1[1]))
                glEnd()
            elif isinstance(ent, ArcEntity):
                pts = ent.tessellate(64)
                glBegin(GL_LINE_STRIP)
                for p in pts:
                    glVertex3f(*self._pt(sketch.plane, p[0], p[1]))
                glEnd()
            if stipple:
                glDisable(GL_LINE_STIPPLE)
        glLineWidth(1.0)

    def _draw_preview(self, sketch: SketchMode):
        r, g, b = prefs.sketch_preview_color
        glColor3f(r, g, b)
        glLineWidth(1.6)

        # Angle-lock ray: dashed line from anchor along the locked angle
        snap = getattr(sketch, 'last_snap', None)
        from cad.sketch_tools.snap import SnapType as _ST
        if (snap is not None and snap.type == _ST.ANGLE
                and sketch.snap.anchor_pt is not None):
            import math as _math
            anchor = sketch.snap.anchor_pt
            cur_s  = snap.point
            delta  = cur_s - anchor
            dist   = float(np.linalg.norm(delta))
            if dist > 1e-6:
                # Extend the ray well beyond the cursor so it reads as a guide
                direction = delta / dist
                far = anchor + direction * max(dist * 2.0, 40.0)
                glColor4f(0.40, 0.85, 1.00, 0.45)
                glLineWidth(1.0)
                # Dashed: draw short segments with gaps
                seg_len  = 3.0   # mm on
                gap_len  = 3.0   # mm off
                total    = float(np.linalg.norm(far - anchor))
                t = 0.0
                drawing = True
                glBegin(GL_LINES)
                while t < total:
                    t_end = min(t + (seg_len if drawing else gap_len), total)
                    if drawing:
                        p0 = anchor + direction * t
                        p1 = anchor + direction * t_end
                        glVertex3f(*self._pt(sketch.plane, p0[0], p0[1]))
                        glVertex3f(*self._pt(sketch.plane, p1[0], p1[1]))
                    t = t_end
                    drawing = not drawing
                glEnd()
            glColor3f(r, g, b)
            glLineWidth(1.6)

        if sketch.tool == SketchTool.LINE:
            from cad.sketch_tools.line import LineTool, _HLINE_EXTENT
            tool = sketch._active_tool
            cur  = sketch._cursor_2d
            if cur is None:
                return
            constrain = tool._constrain if isinstance(tool, LineTool) else None

            if constrain == 'H':
                # Full infinite horizontal preview
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, cur[0] - _HLINE_EXTENT, cur[1]))
                glVertex3f(*self._pt(sketch.plane, cur[0] + _HLINE_EXTENT, cur[1]))
                glEnd()
            elif constrain == 'V':
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, cur[0], cur[1] - _HLINE_EXTENT))
                glVertex3f(*self._pt(sketch.plane, cur[0], cur[1] + _HLINE_EXTENT))
                glEnd()
            elif sketch._line_start is not None:
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane,
                                     sketch._line_start[0], sketch._line_start[1]))
                glVertex3f(*self._pt(sketch.plane, cur[0], cur[1]))
                glEnd()
                glPointSize(5.0)
                glBegin(GL_POINTS)
                glVertex3f(*self._pt(sketch.plane,
                                     sketch._line_start[0], sketch._line_start[1]))
                glEnd()
                glPointSize(1.0)

        elif sketch.tool in (SketchTool.DIMENSION, SketchTool.GEOMETRIC):
            self._draw_constraint_tool_preview(sketch)

        elif sketch.tool == SketchTool.ARC3:
            self._draw_arc_preview(sketch)

        elif sketch.tool == SketchTool.TRIM:
            self._draw_trim_preview(sketch)

        elif sketch.tool == SketchTool.DIVIDE:
            self._draw_divide_preview(sketch)

        elif sketch.tool == SketchTool.OFFSET:
            self._draw_offset_preview(sketch)

        elif sketch.tool == SketchTool.CIRCLE:
            self._draw_circle_preview(sketch)

        elif sketch.tool == SketchTool.FILLET:
            self._draw_fillet_preview(sketch)

        elif sketch.tool == SketchTool.SQUARE:
            self._draw_square_preview(sketch)

        glLineWidth(1.0)

    def _draw_arc_preview(self, sketch: SketchMode):
        from cad.sketch_tools.arc import Arc3Tool, _circle_from_3pts, _arc_angles
        tool = sketch._active_tool
        if not isinstance(tool, Arc3Tool):
            return
        p1 = tool.arc_p1
        p2 = tool.arc_p2
        cur = sketch._cursor_2d
        if p1 is None or cur is None:
            return

        r, g, b = prefs.sketch_preview_color
        glColor3f(r, g, b)

        # Draw fixed anchor point
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glVertex3f(*self._pt(sketch.plane, p1[0], p1[1]))
        glEnd()
        glPointSize(1.0)

        if p2 is None:
            # Only p1 set — draw rubber-band line to cursor
            glBegin(GL_LINES)
            glVertex3f(*self._pt(sketch.plane, p1[0], p1[1]))
            glVertex3f(*self._pt(sketch.plane, cur[0], cur[1]))
            glEnd()
        else:
            # p1 and p2 set — draw p2 marker and arc preview to cursor
            glPointSize(5.0)
            glBegin(GL_POINTS)
            glVertex3f(*self._pt(sketch.plane, p2[0], p2[1]))
            glEnd()
            glPointSize(1.0)

            result = _circle_from_3pts(p1, p2, cur)
            if result is not None:
                center, radius = result
                start_a, end_a = _arc_angles(center, p1, p2, cur)
                import math
                n = 64
                glBegin(GL_LINE_STRIP)
                for i in range(n + 1):
                    a = start_a + (end_a - start_a) * i / n
                    pu = center[0] + radius * math.cos(a)
                    pv = center[1] + radius * math.sin(a)
                    glVertex3f(*self._pt(sketch.plane, pu, pv))
                glEnd()
            else:
                # Collinear — show line
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, p1[0], p1[1]))
                glVertex3f(*self._pt(sketch.plane, cur[0], cur[1]))
                glEnd()

    def _draw_fillet_preview(self, sketch: SketchMode):
        from cad.sketch_tools.fillet import FilletTool, compute_fillet
        tool = sketch._active_tool
        if not isinstance(tool, FilletTool):
            return

        corner = (tool.selected_corner if tool._state == FilletTool.STATE_SELECTED
                  else tool.hovered_corner)
        if corner is None:
            return

        pt, ent_a, ent_b, end_a, end_b = corner

        glEnable(GL_BLEND)

        # Highlight the corner point
        glColor4f(1.0, 0.65, 0.1, 0.9)
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glVertex3f(*self._pt(sketch.plane, pt[0], pt[1]))
        glEnd()
        glPointSize(1.0)

        # Highlight the two entities meeting at the corner
        glColor4f(1.0, 0.75, 0.2, 0.7)
        glLineWidth(2.5)
        for ent in (ent_a, ent_b):
            if isinstance(ent, ArcEntity):
                pts = ent.tessellate(64)
                glBegin(GL_LINE_STRIP)
                for p in pts:
                    glVertex3f(*self._pt(sketch.plane, p[0], p[1]))
                glEnd()
            else:
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, ent.p0[0], ent.p0[1]))
                glVertex3f(*self._pt(sketch.plane, ent.p1[0], ent.p1[1]))
                glEnd()

        # Live preview arc — radius pushed from panel to tool on every change
        radius = tool.preview_radius

        if radius is not None:
            tagged = tool.current_result()
            if tagged is not None:
                result, _ea, _eb = tagged
            else:
                from cad.sketch_tools.fillet import compute_fillet
                result = compute_fillet(pt, ent_a, end_a, ent_b, end_b, radius)
            if result is not None:
                center, tp_a, tp_b, sa, ea, t_a, t_b = result
                # Draw preview arc
                glColor4f(0.4, 0.9, 0.4, 0.85)
                glLineWidth(2.0)
                n = 48
                span = ea - sa
                glBegin(GL_LINE_STRIP)
                for i in range(n + 1):
                    a = sa + span * i / n
                    glVertex3f(*self._pt(sketch.plane,
                                        center[0] + radius * math.cos(a),
                                        center[1] + radius * math.sin(a)))
                glEnd()
                # Tangent point markers
                glColor4f(0.4, 0.9, 0.4, 0.9)
                glPointSize(6.0)
                glBegin(GL_POINTS)
                glVertex3f(*self._pt(sketch.plane, tp_a[0], tp_a[1]))
                glVertex3f(*self._pt(sketch.plane, tp_b[0], tp_b[1]))
                glEnd()
                glPointSize(1.0)

        glLineWidth(1.0)
        glDisable(GL_BLEND)

    def _draw_square_preview(self, sketch: SketchMode):
        """Draw a 2-point square preview when the first corner is set."""
        from cad.sketch_tools.square import SquareTool
        tool = sketch._active_tool
        if not isinstance(tool, SquareTool):
            return

        r, g, b = prefs.sketch_preview_color
        glColor3f(r, g, b)
        glLineWidth(1.6)

        p1 = tool.square_p1
        cur = tool.cursor_2d
        if p1 is None or cur is None:
            return

        x0, y0 = float(p1[0]), float(p1[1])
        x1, y1 = float(cur[0]), float(cur[1])

        # Square vertices in sketch (u, v) space
        a = (x0, y0)
        b = (x1, y0)
        c = (x1, y1)
        d = (x0, y1)

        # Draw 4 edges
        glBegin(GL_LINE_STRIP)
        glVertex3f(*self._pt(sketch.plane, a[0], a[1]))
        glVertex3f(*self._pt(sketch.plane, b[0], b[1]))
        glVertex3f(*self._pt(sketch.plane, c[0], c[1]))
        glVertex3f(*self._pt(sketch.plane, d[0], d[1]))
        glVertex3f(*self._pt(sketch.plane, a[0], a[1]))
        glEnd()

        # Corner dot at first corner
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glVertex3f(*self._pt(sketch.plane, a[0], a[1]))
        glEnd()
        glPointSize(1.0)

        # Corner dot at opposite corner
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glVertex3f(*self._pt(sketch.plane, c[0], c[1]))
        glEnd()
        glPointSize(1.0)

    def _draw_circle_preview(self, sketch: SketchMode):
        from cad.sketch_tools.circle import CircleTool
        tool = sketch._active_tool
        if not isinstance(tool, CircleTool):
            return

        r, g, b = prefs.sketch_preview_color
        glColor3f(r, g, b)
        glLineWidth(1.6)

        # Draw fixed anchor points
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for p in tool.pts:
            glVertex3f(*self._pt(sketch.plane, p[0], p[1]))
        glEnd()
        glPointSize(1.0)

        # Draw preview circle
        result = tool.preview_circle()
        if result is not None:
            center, radius = result
            n = 96
            glBegin(GL_LINE_LOOP)
            for i in range(n):
                a = 2 * math.pi * i / n
                glVertex3f(*self._pt(sketch.plane,
                                     center[0] + radius * math.cos(a),
                                     center[1] + radius * math.sin(a)))
            glEnd()
            # Center crosshair
            cr = radius * 0.06
            glBegin(GL_LINES)
            glVertex3f(*self._pt(sketch.plane, center[0] - cr, center[1]))
            glVertex3f(*self._pt(sketch.plane, center[0] + cr, center[1]))
            glVertex3f(*self._pt(sketch.plane, center[0], center[1] - cr))
            glVertex3f(*self._pt(sketch.plane, center[0], center[1] + cr))
            glEnd()

    def _draw_offset_preview(self, sketch: SketchMode):
        """Highlight hovered or selected entities for offset."""
        from cad.sketch_tools.offset import OffsetTool
        tool = sketch._active_tool
        if not isinstance(tool, OffsetTool):
            return

        if tool._state == OffsetTool.STATE_SELECTED:
            entities = tool.selected_entities
            color = (0.30, 0.75, 1.00, 0.9)   # blue = selected
        else:
            entities = tool.hovered_entities
            color = (1.00, 0.80, 0.20, 0.75)  # yellow = hover

        if not entities:
            return

        glEnable(GL_BLEND)
        glColor4f(*color)
        glLineWidth(3.0)
        for ent in entities:
            if isinstance(ent, LineEntity):
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, ent.p0[0], ent.p0[1]))
                glVertex3f(*self._pt(sketch.plane, ent.p1[0], ent.p1[1]))
                glEnd()
            elif isinstance(ent, ArcEntity):
                pts = ent.tessellate(64)
                glBegin(GL_LINE_STRIP)
                for p in pts:
                    glVertex3f(*self._pt(sketch.plane, p[0], p[1]))
                glEnd()
        glLineWidth(1.0)
        glDisable(GL_BLEND)

    def _draw_divide_preview(self, sketch: SketchMode):
        """Highlight the entity that would be divided, showing split points."""
        from cad.sketch_tools.trim import (_closest_entity,
                                           _gather_split_params,
                                           _split_line, _split_arc)
        from cad.sketch_tools.divide import DivideTool
        tool = sketch._active_tool
        if not isinstance(tool, DivideTool) or tool.cursor_2d is None:
            return

        click_pt = tool.cursor_2d
        entities = sketch.entities
        drawn = [e for e in entities if isinstance(e, (LineEntity, ArcEntity))]
        if not drawn:
            return

        target, target_idx = _closest_entity(click_pt, drawn, entities)
        if target is None:
            return

        t_params = _gather_split_params(target, entities, target_idx)

        glEnable(GL_BLEND)
        glColor4f(0.30, 0.80, 1.00, 0.85)  # blue = will be divided
        glLineWidth(3.0)

        # Highlight the target entity
        if isinstance(target, LineEntity):
            glBegin(GL_LINES)
            glVertex3f(*self._pt(sketch.plane, target.p0[0], target.p0[1]))
            glVertex3f(*self._pt(sketch.plane, target.p1[0], target.p1[1]))
            glEnd()
        else:
            pts = target.tessellate(64)
            glBegin(GL_LINE_STRIP)
            for p in pts:
                glVertex3f(*self._pt(sketch.plane, p[0], p[1]))
            glEnd()

        # Show split points as orange dots
        if t_params:
            glColor4f(1.0, 0.55, 0.10, 1.0)
            glPointSize(7.0)
            glBegin(GL_POINTS)
            for t in t_params:
                if isinstance(target, LineEntity):
                    pt = target.p0 + t * (target.p1 - target.p0)
                else:
                    import math as _m
                    a = target.start_angle + t * (target.end_angle - target.start_angle)
                    pt = target.center + target.radius * np.array([_m.cos(a), _m.sin(a)])
                glVertex3f(*self._pt(sketch.plane, pt[0], pt[1]))
            glEnd()
            glPointSize(1.0)

        glLineWidth(1.0)
        glDisable(GL_BLEND)

    def _draw_trim_preview(self, sketch: SketchMode):
        """Highlight the sub-piece that would be removed on click."""
        from cad.sketch_tools.trim import (TrimTool, _closest_entity,
                                           _gather_split_params,
                                           _split_line, _split_arc)
        tool = sketch._active_tool
        if not isinstance(tool, TrimTool) or tool.cursor_2d is None:
            return

        click_pt = tool.cursor_2d
        entities = sketch.entities
        drawn = [e for e in entities if isinstance(e, (LineEntity, ArcEntity))]
        if not drawn:
            return

        target, target_idx = _closest_entity(click_pt, drawn, entities)
        if target is None:
            return

        t_params = _gather_split_params(target, entities, target_idx)

        if not t_params:
            piece = target
        else:
            if isinstance(target, LineEntity):
                pieces = _split_line(target, t_params)
            else:
                pieces = _split_arc(target, t_params)

            if len(pieces) <= 1:
                piece = target
            else:
                from cad.sketch_tools.snap import _nearest_on_arc
                if isinstance(target, LineEntity):
                    d = target.p1 - target.p0
                    len_sq = float(np.dot(d, d))
                    t_click = (float(np.dot(click_pt - target.p0, d)) / len_sq
                               if len_sq > 1e-12 else 0.0)

                    def piece_score_line(p):
                        ta = float(np.dot(p.p0 - target.p0, d)) / len_sq
                        tb = float(np.dot(p.p1 - target.p0, d)) / len_sq
                        lo, hi = min(ta, tb), max(ta, tb)
                        return max(0.0, lo - t_click, t_click - hi)

                    remove_idx = min(range(len(pieces)),
                                     key=lambda i: piece_score_line(pieces[i]))
                else:
                    def piece_nearest_dist(p):
                        nearest, _ = _nearest_on_arc(click_pt, p)
                        return float(np.linalg.norm(nearest - click_pt))

                    remove_idx = min(range(len(pieces)),
                                     key=lambda i: piece_nearest_dist(pieces[i]))
                piece = pieces[remove_idx]
        glEnable(GL_BLEND)
        glColor4f(1.0, 0.25, 0.25, 0.85)
        glLineWidth(3.5)
        if isinstance(piece, LineEntity):
            glBegin(GL_LINES)
            glVertex3f(*self._pt(sketch.plane, piece.p0[0], piece.p0[1]))
            glVertex3f(*self._pt(sketch.plane, piece.p1[0], piece.p1[1]))
            glEnd()
        else:
            pts = piece.tessellate(64)
            glBegin(GL_LINE_STRIP)
            for p in pts:
                glVertex3f(*self._pt(sketch.plane, p[0], p[1]))
            glEnd()
        glLineWidth(1.0)
        glDisable(GL_BLEND)

    def _draw_cursor(self, sketch: SketchMode, camera_distance: float):
        if sketch._cursor_2d is None:
            return
        u, v = sketch._cursor_2d
        cr   = camera_distance * 0.008
        glColor3f(*prefs.sketch_cursor_color)
        glLineWidth(1.2)
        glBegin(GL_LINES)
        glVertex3f(*self._pt(sketch.plane, u - cr, v))
        glVertex3f(*self._pt(sketch.plane, u + cr, v))
        glVertex3f(*self._pt(sketch.plane, u,      v - cr))
        glVertex3f(*self._pt(sketch.plane, u,      v + cr))
        glEnd()
        glLineWidth(1.0)

    def _draw_snap_indicator(self, sketch: SketchMode, camera_distance: float):
        """
        Draw a snap type indicator at the current snap point.
        Each snap type gets a distinct symbol so you always know what's active:
          ENDPOINT    — square
          MIDPOINT    — triangle
          CENTER      — circle (octagon approximation)
          NEAREST     — X mark
          GRID        — small diamond
          FREE        — nothing (cursor crosshair is enough)
        """
        snap = getattr(sketch, 'last_snap', None)
        if snap is None or snap.type.name == 'FREE':
            return

        from cad.sketch_tools.snap import SnapType
        u, v = float(snap.point[0]), float(snap.point[1])
        s    = camera_distance * 0.012   # symbol half-size

        # Color per snap type
        colors = {
            SnapType.ENDPOINT:     (1.00, 0.55, 0.10),   # orange
            SnapType.MIDPOINT:     (0.20, 0.85, 0.40),   # green
            SnapType.CENTER:       (0.30, 0.70, 1.00),   # blue
            SnapType.NEAREST:      (0.85, 0.85, 0.20),   # yellow
            SnapType.TANGENT:      (0.85, 0.30, 1.00),   # purple
            SnapType.INTERSECTION: (1.00, 0.20, 0.50),   # pink-red
            SnapType.GRID:         (0.55, 0.55, 0.55),   # grey
            SnapType.ANGLE:        (0.40, 0.85, 1.00),   # cyan
        }
        r, g, b = colors.get(snap.type, (1.0, 1.0, 1.0))

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(r, g, b, 0.9)
        glLineWidth(1.5)

        if snap.type == SnapType.ENDPOINT:
            # Square
            glBegin(GL_LINE_LOOP)
            glVertex3f(*self._pt(sketch.plane, u - s, v - s))
            glVertex3f(*self._pt(sketch.plane, u + s, v - s))
            glVertex3f(*self._pt(sketch.plane, u + s, v + s))
            glVertex3f(*self._pt(sketch.plane, u - s, v + s))
            glEnd()

        elif snap.type == SnapType.MIDPOINT:
            # Triangle (pointing up)
            glBegin(GL_LINE_LOOP)
            glVertex3f(*self._pt(sketch.plane, u,       v + s * 1.2))
            glVertex3f(*self._pt(sketch.plane, u - s,   v - s * 0.7))
            glVertex3f(*self._pt(sketch.plane, u + s,   v - s * 0.7))
            glEnd()

        elif snap.type == SnapType.CENTER:
            # Octagon (circle approximation)
            glBegin(GL_LINE_LOOP)
            for i in range(8):
                a = math.pi * 2 * i / 8
                glVertex3f(*self._pt(sketch.plane,
                                     u + s * math.cos(a),
                                     v + s * math.sin(a)))
            glEnd()

        elif snap.type == SnapType.NEAREST:
            # X mark
            glBegin(GL_LINES)
            glVertex3f(*self._pt(sketch.plane, u - s, v - s))
            glVertex3f(*self._pt(sketch.plane, u + s, v + s))
            glVertex3f(*self._pt(sketch.plane, u + s, v - s))
            glVertex3f(*self._pt(sketch.plane, u - s, v + s))
            glEnd()

        elif snap.type == SnapType.INTERSECTION:
            # X inside a circle
            glBegin(GL_LINE_LOOP)
            for i in range(8):
                a = math.pi * 2 * i / 8
                glVertex3f(*self._pt(sketch.plane,
                                     u + s * math.cos(a),
                                     v + s * math.sin(a)))
            glEnd()
            glBegin(GL_LINES)
            glVertex3f(*self._pt(sketch.plane, u - s*0.6, v - s*0.6))
            glVertex3f(*self._pt(sketch.plane, u + s*0.6, v + s*0.6))
            glVertex3f(*self._pt(sketch.plane, u + s*0.6, v - s*0.6))
            glVertex3f(*self._pt(sketch.plane, u - s*0.6, v + s*0.6))
            glEnd()

        elif snap.type == SnapType.TANGENT:
            # Small circle with a tangent tick line
            glBegin(GL_LINE_LOOP)
            for i in range(8):
                a = math.pi * 2 * i / 8
                glVertex3f(*self._pt(sketch.plane,
                                     u + s * math.cos(a),
                                     v + s * math.sin(a)))
            glEnd()
            glBegin(GL_LINES)
            glVertex3f(*self._pt(sketch.plane, u - s, v))
            glVertex3f(*self._pt(sketch.plane, u + s, v))
            glEnd()

        elif snap.type == SnapType.GRID:
            # Diamond
            glBegin(GL_LINE_LOOP)
            glVertex3f(*self._pt(sketch.plane, u,     v + s))
            glVertex3f(*self._pt(sketch.plane, u + s, v))
            glVertex3f(*self._pt(sketch.plane, u,     v - s))
            glVertex3f(*self._pt(sketch.plane, u - s, v))
            glEnd()

        elif snap.type == SnapType.ANGLE:
            # Rotated square (diamond) + small arc tick
            glBegin(GL_LINE_LOOP)
            glVertex3f(*self._pt(sketch.plane, u,       v + s))
            glVertex3f(*self._pt(sketch.plane, u + s,   v))
            glVertex3f(*self._pt(sketch.plane, u,       v - s))
            glVertex3f(*self._pt(sketch.plane, u - s,   v))
            glEnd()
            # Tick lines (cross inside diamond)
            glBegin(GL_LINES)
            glVertex3f(*self._pt(sketch.plane, u - s * 0.5, v))
            glVertex3f(*self._pt(sketch.plane, u + s * 0.5, v))
            glVertex3f(*self._pt(sketch.plane, u, v - s * 0.5))
            glVertex3f(*self._pt(sketch.plane, u, v + s * 0.5))
            glEnd()

        glLineWidth(1.0)
        glDisable(GL_BLEND)

    # ------------------------------------------------------------------
    # Committed sketch rendering (SketchEntry)
    # ------------------------------------------------------------------

    def _draw_committed_lines(self, entry, hovered_edge=None, history_idx=None,
                              selected_indices=None):
        from cad.sketch import ArcEntity
        from viewer.hover import parse_sketch_key
        hov_entity_idx = None
        if hovered_edge is not None and history_idx is not None:
            hov_body, _ = hovered_edge
            if hov_body is not None:
                sk = parse_sketch_key(hov_body)
                if sk is not None and sk[0] == history_idx:
                    hov_entity_idx = sk[1]

        sel = selected_indices or set()
        r, g, b = prefs.sketch_line_color
        glLineWidth(prefs.sketch_line_width)
        for j, ent in enumerate(entry.entities):
            is_construction = getattr(ent, "construction", False)
            if j in sel:
                glColor4f(*prefs.edge_selected_color, 1.0)
            elif j == hov_entity_idx:
                glColor4f(*prefs.edge_hovered_color, 1.0)
            elif is_construction:
                glColor4f(r, g, b, 0.30)
            else:
                glColor4f(r, g, b, 0.55)
            stipple = is_construction and j not in sel and j != hov_entity_idx
            if stipple:
                glEnable(GL_LINE_STIPPLE)
                glLineStipple(2, 0x00FF)
            if isinstance(ent, LineEntity):
                glBegin(GL_LINES)
                glVertex3f(*self._pt_from_entry(entry, ent.p0[0], ent.p0[1]))
                glVertex3f(*self._pt_from_entry(entry, ent.p1[0], ent.p1[1]))
                glEnd()
            elif isinstance(ent, ArcEntity):
                pts = ent.tessellate(64)
                glBegin(GL_LINE_STRIP)
                for p in pts:
                    glVertex3f(*self._pt_from_entry(entry, p[0], p[1]))
                glEnd()
            if stipple:
                glDisable(GL_LINE_STIPPLE)
        glLineWidth(1.0)

    def _draw_committed_references(self, entry):
        refs = [e for e in entry.entities if isinstance(e, ReferenceEntity)]
        if not refs:
            return
        r, g, b = prefs.sketch_reference_color
        glColor4f(r, g, b, 0.45)
        glLineWidth(prefs.sketch_reference_width)
        for ref in refs:
            if len(ref.points) == 1:
                p  = ref.points[0]
                cr = 0.8
                glBegin(GL_LINES)
                glVertex3f(*self._pt_from_entry(entry, p[0] - cr, p[1]))
                glVertex3f(*self._pt_from_entry(entry, p[0] + cr, p[1]))
                glVertex3f(*self._pt_from_entry(entry, p[0],      p[1] - cr))
                glVertex3f(*self._pt_from_entry(entry, p[0],      p[1] + cr))
                glEnd()
            else:
                glBegin(GL_LINE_STRIP)
                for p in ref.points:
                    glVertex3f(*self._pt_from_entry(entry, p[0], p[1]))
                glEnd()
        glLineWidth(1.0)

    def _draw_constraint_tool_preview(self, sketch: SketchMode):
        """Highlight hovered/selected lines for dimension and parallel tools."""
        from cad.sketch import LineEntity
        tool = sketch._active_tool
        if tool is None:
            return

        first_idx = getattr(tool, 'first_idx', None)
        hover_idx = getattr(tool, 'hover_entity_idx', None)

        glEnable(GL_BLEND)
        glLineWidth(3.0)

        # First selected line — solid blue
        if first_idx is not None and first_idx < len(sketch.entities):
            ent = sketch.entities[first_idx]
            if isinstance(ent, LineEntity):
                glColor4f(0.25, 0.60, 1.00, 0.95)
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, ent.p0[0], ent.p0[1]))
                glVertex3f(*self._pt(sketch.plane, ent.p1[0], ent.p1[1]))
                glEnd()

        # Hovered line — yellow
        if hover_idx is not None and hover_idx != first_idx and hover_idx < len(sketch.entities):
            ent = sketch.entities[hover_idx]
            if isinstance(ent, LineEntity):
                glColor4f(1.00, 0.80, 0.20, 0.85)
                glBegin(GL_LINES)
                glVertex3f(*self._pt(sketch.plane, ent.p0[0], ent.p0[1]))
                glVertex3f(*self._pt(sketch.plane, ent.p1[0], ent.p1[1]))
                glEnd()

        glLineWidth(1.0)
        glDisable(GL_BLEND)

    def _draw_dimension_constraints(self, entities, constraints, to_3d, camera_distance, dimmed: bool = False) -> list[dict]:
        """
        Draw blueprint-style dimension annotations for all distance constraints.

        Returns a list of label dicts:
          { 'world': np.ndarray(3,), 'text': str, 'constraint_idx': int,
            'entity_idx': int }
        The viewport uses these to draw text via QPainter and handle clicks.
        """
        from cad.sketch import LineEntity, SketchConstraint
        labels = []

        # Default offset distance scales with camera distance.
        default_offset = max(camera_distance * 0.04, 2.0)
        arrow_size = default_offset * 0.35

        glEnable(GL_BLEND)
        glLineWidth(1.2)

        for ci, con in enumerate(constraints):
            if con.type != 'distance':
                continue
            ei = con.indices[0]
            if ei >= len(entities) or not isinstance(entities[ei], LineEntity):
                continue
            ent = entities[ei]
            p0 = np.array([float(ent.p0[0]), float(ent.p0[1])])
            p1 = np.array([float(ent.p1[0]), float(ent.p1[1])])

            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            seg_dir = seg / seg_len
            perp = np.array([-seg_dir[1], seg_dir[0]])  # 90° CCW

            # Use stored label_offset if dragged, otherwise default.
            offset = float(con.label_offset) if con.label_offset is not None else default_offset

            d0 = p0 + perp * offset
            d1 = p1 + perp * offset

            ext_gap    = abs(offset) * 0.15
            ext_beyond = abs(offset) * 0.15
            # Extension lines always go from entity toward dimension line.
            ext_dir = np.sign(offset) * perp if offset != 0 else perp

            alpha = 0.25 if dimmed else 0.90
            glColor4f(0.35, 0.70, 1.00, alpha)  # blueprint blue

            glBegin(GL_LINES)
            glVertex3f(*_v3(to_3d(*(p0 + ext_dir * ext_gap))))
            glVertex3f(*_v3(to_3d(*(d0 + ext_dir * ext_beyond))))
            glVertex3f(*_v3(to_3d(*(p1 + ext_dir * ext_gap))))
            glVertex3f(*_v3(to_3d(*(d1 + ext_dir * ext_beyond))))
            glVertex3f(*_v3(to_3d(*d0)))
            glVertex3f(*_v3(to_3d(*d1)))
            glEnd()

            for tip, away in ((d0, seg_dir), (d1, -seg_dir)):
                base_centre = tip + away * arrow_size
                base_half   = perp * (arrow_size * 0.35)
                glBegin(GL_TRIANGLES)
                glVertex3f(*_v3(to_3d(*tip)))
                glVertex3f(*_v3(to_3d(*(base_centre + base_half))))
                glVertex3f(*_v3(to_3d(*(base_centre - base_half))))
                glEnd()

            # Label at midpoint of dimension line.
            mid_uv = (d0 + d1) * 0.5
            mid_3d = to_3d(*mid_uv)
            from cad.prefs import prefs
            from cad.units import format_value
            labels.append({
                'world':          np.array([float(mid_3d[0]), float(mid_3d[1]), float(mid_3d[2])]),
                'text':           format_value(con.value, prefs.default_unit, prefs.display_decimals),
                'constraint_idx': ci,
                'entity_idx':     ei,
                'dimmed':         dimmed,
                # World-space perp direction (unit vector) for drag projection.
                'perp_world':     np.array(_v3(to_3d(*perp))) -
                                  np.array(_v3(to_3d(0.0, 0.0))),
                'perp_uv':        perp,
                'p0_uv':          p0,
                'constraints':    constraints,  # reference for live update
            })

        # Diameter constraint annotations — full chord across circle, draggable.
        from cad.sketch import ArcEntity as _AE
        from cad.prefs import prefs
        from cad.units import format_value
        for ci, con in enumerate(constraints):
            if con.type != 'diameter':
                continue
            ei = con.indices[0]
            if ei >= len(entities):
                continue
            ent = entities[ei]
            if not isinstance(ent, _AE):
                continue

            cx, cy = float(ent.center[0]), float(ent.center[1])
            r      = float(ent.radius)
            angle  = float(con.label_angle) if con.label_angle is not None else 0.0
            # Radial offset beyond circle edge (positive = outside).
            radial_off = float(con.label_offset) if con.label_offset is not None \
                         else default_offset

            cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
            chord_dir  = np.array([cos_a, sin_a])
            chord_perp = np.array([-sin_a, cos_a])  # 90° CCW from chord

            # Chord endpoints on circle.
            q0 = np.array([cx, cy]) - chord_dir * r
            q1 = np.array([cx, cy]) + chord_dir * r

            # If label_offset != 0, the chord shifts perpendicular to itself
            # (like a linear dimension) so the label sits outside the circle.
            chord_shift = radial_off  # mm perpendicular to chord direction
            d0 = q0 + chord_perp * chord_shift
            d1 = q1 + chord_perp * chord_shift

            alpha = 0.25 if dimmed else 0.90
            glColor4f(0.35, 0.70, 1.00, alpha)

            # Extension lines from circle edge to shifted chord line.
            ext_gap    = abs(chord_shift) * 0.15
            ext_beyond = abs(chord_shift) * 0.15
            ext_dir    = np.sign(chord_shift) * chord_perp if chord_shift != 0 \
                         else chord_perp

            glBegin(GL_LINES)
            if abs(chord_shift) > 1e-3:
                glVertex3f(*_v3(to_3d(*(q0 + ext_dir * ext_gap))))
                glVertex3f(*_v3(to_3d(*(d0 + ext_dir * ext_beyond))))
                glVertex3f(*_v3(to_3d(*(q1 + ext_dir * ext_gap))))
                glVertex3f(*_v3(to_3d(*(d1 + ext_dir * ext_beyond))))
            glVertex3f(*_v3(to_3d(*d0)))
            glVertex3f(*_v3(to_3d(*d1)))
            glEnd()

            # Arrowheads pointing inward along chord.
            for tip, away in ((d0, chord_dir), (d1, -chord_dir)):
                base_centre = tip + away * arrow_size
                base_half   = chord_perp * (arrow_size * 0.35)
                glBegin(GL_TRIANGLES)
                glVertex3f(*_v3(to_3d(*tip)))
                glVertex3f(*_v3(to_3d(*(base_centre + base_half))))
                glVertex3f(*_v3(to_3d(*(base_centre - base_half))))
                glEnd()

            # Label at midpoint of chord line.
            mid_uv = (d0 + d1) * 0.5
            mid_3d = to_3d(*mid_uv)

            # perp_world for radial (in/out) drag — perpendicular to chord.
            perp_uv_for_drag = chord_perp
            perp_world_for_drag = (np.array(_v3(to_3d(*perp_uv_for_drag))) -
                                   np.array(_v3(to_3d(0.0, 0.0))))

            # chord_world for angular drag — along the chord direction.
            chord_world = (np.array(_v3(to_3d(*chord_dir))) -
                           np.array(_v3(to_3d(0.0, 0.0))))

            labels.append({
                'world':            np.array([float(mid_3d[0]), float(mid_3d[1]),
                                              float(mid_3d[2])]),
                'text':             'ø' + format_value(con.value, prefs.default_unit,
                                                       prefs.display_decimals),
                'constraint_idx':   ci,
                'entity_idx':       ei,
                'dimmed':           dimmed,
                'constraints':      constraints,
                # Drag geometry.
                'perp_world':       perp_world_for_drag,
                'perp_uv':          perp_uv_for_drag,
                'p0_uv':            mid_uv,
                # Diameter-specific drag data.
                'is_diameter':      True,
                'chord_world':      chord_world,
                'chord_dir_uv':     chord_dir,
                'arc_center_uv':    np.array([cx, cy]),
                'arc_radius':       r,
                'arc_center_world': np.array(_v3(to_3d(cx, cy))),
            })

        # Parallel constraint labels — "||" floating at each line's midpoint.
        # Group by constraint so paired lines share the same label index.
        from cad.sketch import LineEntity as _LE
        for ci, con in enumerate(constraints):
            if con.type not in ('parallel', 'perpendicular', 'horizontal', 'vertical', 'equal'):
                continue
            for ei in con.indices:
                if ei >= len(entities) or not isinstance(entities[ei], _LE):
                    continue
                ent   = entities[ei]
                mid_uv = (np.array([float(ent.p0[0]), float(ent.p0[1])]) +
                          np.array([float(ent.p1[0]), float(ent.p1[1])])) * 0.5
                # Offset slightly perpendicular to the line so it doesn't overlap.
                seg   = np.array([float(ent.p1[0] - ent.p0[0]),
                                   float(ent.p1[1] - ent.p0[1])])
                slen  = float(np.linalg.norm(seg))
                if slen > 1e-6:
                    perp = np.array([-seg[1], seg[0]]) / slen
                    mid_uv = mid_uv + perp * max(camera_distance * 0.025, 1.5)
                mid_3d = to_3d(*mid_uv)
                labels.append({
                    'world':          np.array([float(mid_3d[0]), float(mid_3d[1]),
                                                float(mid_3d[2])]),
                    'text':           {'parallel': '||', 'perpendicular': '⊥',
                                       'horizontal': '—', 'vertical': '|',
                                       'equal': '='
                                       }.get(con.type, con.type),
                    'constraint_idx': ci,
                    'entity_idx':     ei,
                    'parallel':       True,   # flag so click doesn't open value editor
                    'dimmed':         dimmed,
                    'perp_world':     np.array(_v3(to_3d(*perp))) -
                                      np.array(_v3(to_3d(0.0, 0.0))) if slen > 1e-6 else
                                      np.array([0.0, 0.0, 0.0]),
                    'perp_uv':        perp if slen > 1e-6 else np.array([0.0, 1.0]),
                    'p0_uv':          mid_uv,
                    'constraints':    constraints,
                })

        glLineWidth(1.0)
        glDisable(GL_BLEND)
        return labels


def _v3(pt) -> tuple[float, float, float]:
    """Convert a numpy array or sequence to a (x,y,z) float tuple."""
    return (float(pt[0]), float(pt[1]), float(pt[2]))
