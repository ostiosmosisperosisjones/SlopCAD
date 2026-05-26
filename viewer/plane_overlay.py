"""
viewer/plane_overlay.py

Render committed offset-plane datum entries as translucent quads, plus the
in-progress offset-plane panel preview when one is open.

Auto-sizing:
  - parent is a FacePlaneSource → quad spans the parent face's bbox
  - parent is a WorldPlaneSource → quad uses a scene-derived default size
  - failed-to-resolve plane → not drawn (history panel surfaces the error)
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import (
    glDisable, glEnable, glDepthMask, glColor4f, glLineWidth, glBegin, glEnd,
    glVertex3f, glBlendFunc,
    GL_LIGHTING, GL_BLEND, GL_CULL_FACE,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS, GL_LINE_LOOP, GL_TRUE, GL_FALSE,
)


# Cyan-tinted color so planes are distinct from blue world-plane XY.
_FILL  = (0.42, 0.78, 0.90, 0.10)
_EDGE  = (0.42, 0.78, 0.90, 0.55)
_HOV   = (0.42, 0.78, 0.90, 0.18)
_HOV_E = (0.55, 0.90, 1.00, 0.90)
_SEL   = (0.95, 0.85, 0.15, 0.20)
_SEL_E = (1.00, 0.95, 0.30, 0.90)
_PREVIEW_FILL = (0.95, 0.85, 0.15, 0.08)
_PREVIEW_EDGE = (0.95, 0.85, 0.15, 0.55)


def _quad_corners(origin, normal, x_dir, half_size: float):
    """Build four corners of an axis-aligned quad on (origin, normal, x_dir)."""
    n = normal / max(np.linalg.norm(normal), 1e-12)
    x = x_dir - n * float(np.dot(x_dir, n))   # reject n from x_dir
    x = x / max(np.linalg.norm(x), 1e-12)
    y = np.cross(n, x)
    h = float(half_size)
    return [
        origin - x * h - y * h,
        origin + x * h - y * h,
        origin + x * h + y * h,
        origin - x * h + y * h,
    ]


def _draw_quad(corners, fill_color, edge_color):
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_CULL_FACE)
    glDepthMask(GL_FALSE)

    glColor4f(*fill_color)
    glBegin(GL_QUADS)
    for c in corners:
        glVertex3f(float(c[0]), float(c[1]), float(c[2]))
    glEnd()

    glColor4f(*edge_color)
    glLineWidth(1.4)
    glBegin(GL_LINE_LOOP)
    for c in corners:
        glVertex3f(float(c[0]), float(c[1]), float(c[2]))
    glEnd()

    glDepthMask(GL_TRUE)


def collect_offset_planes(viewport):
    """
    Walk history up to the cursor and return a list of (entry_idx, entry, plane,
    half_size) tuples for every offset_plane entry that resolves successfully.

    Used both for rendering and hover/picking, so the geometry stays in one place.
    """
    out = []
    history = viewport.history
    cursor  = history.cursor
    entries = history.entries
    end = min(cursor + 1, len(entries))
    for i in range(end):
        e = entries[i]
        if e.operation != "offset_plane" or e.error:
            continue
        op = e.op
        if op is None:
            continue
        try:
            b3d_plane = op.plane_source.resolve(history, i)
        except Exception:
            continue
        origin = np.array([b3d_plane.origin.X, b3d_plane.origin.Y,
                           b3d_plane.origin.Z], dtype=float)
        normal = np.array([b3d_plane.z_dir.X, b3d_plane.z_dir.Y,
                           b3d_plane.z_dir.Z], dtype=float)
        x_dir  = np.array([b3d_plane.x_dir.X, b3d_plane.x_dir.Y,
                           b3d_plane.x_dir.Z], dtype=float)
        half = _half_size_for(viewport, op.plane_source)
        out.append((i, e, origin, normal, x_dir, half))
    return out


def _half_size_for(viewport, plane_source) -> float:
    """Match the sizing rule used by OffsetPlaneMixin._offset_plane_size()."""
    from cad.plane_ref import OffsetPlaneSource, FacePlaneSource
    node = plane_source
    while isinstance(node, OffsetPlaneSource):
        node = node.parent
    if isinstance(node, FacePlaneSource):
        shape = viewport.workspace.current_shape(node.body_id)
        if shape is not None:
            try:
                _, face = node.face_ref.find_in(shape)
                if face is not None:
                    bb  = face.bounding_box()
                    ext = max(bb.size.X, bb.size.Y, bb.size.Z) * 0.6
                    return max(20.0, ext) * 0.5
            except Exception:
                pass
    if viewport._meshes:
        all_mins = np.vstack([m.bbox_min for m in viewport._meshes.values()])
        all_maxs = np.vstack([m.bbox_max for m in viewport._meshes.values()])
        ext = float(np.linalg.norm(all_maxs.max(axis=0) - all_mins.min(axis=0))) * 0.4
        return max(40.0, ext) * 0.5
    return 25.0


def draw_offset_planes(viewport):
    """Draw all resolved offset planes + the in-progress preview quad."""
    planes = collect_offset_planes(viewport)
    hover_idx = getattr(viewport, '_hovered_plane_idx', None)
    sel_idx   = getattr(viewport, '_selected_plane_idx', None)
    for i, _entry, origin, normal, x_dir, half in planes:
        corners = _quad_corners(origin, normal, x_dir, half)
        if i == sel_idx:
            _draw_quad(corners, _SEL, _SEL_E)
        elif i == hover_idx:
            _draw_quad(corners, _HOV, _HOV_E)
        else:
            _draw_quad(corners, _FILL, _EDGE)

    preview = getattr(viewport, '_offset_plane_preview', None)
    if preview is not None:
        origin, normal, x_dir, size = preview
        corners = _quad_corners(origin, normal, x_dir, size * 0.5)
        _draw_quad(corners, _PREVIEW_FILL, _PREVIEW_EDGE)


def pick_offset_plane(viewport, ray_origin, ray_dir) -> int | None:
    """Return the history index of the closest offset plane hit by the ray."""
    planes  = collect_offset_planes(viewport)
    best_t  = float('inf')
    best_i  = None
    for i, _e, origin, normal, x_dir, half in planes:
        denom = float(np.dot(normal, ray_dir))
        if abs(denom) < 1e-8:
            continue
        t = float(np.dot(normal, origin - ray_origin)) / denom
        if t <= 0:
            continue
        hit = ray_origin + ray_dir * t
        # Local-frame coords against the quad bounds
        n = normal / max(np.linalg.norm(normal), 1e-12)
        x = x_dir - n * float(np.dot(x_dir, n))
        x = x / max(np.linalg.norm(x), 1e-12)
        y = np.cross(n, x)
        d = hit - origin
        u = float(np.dot(d, x))
        v = float(np.dot(d, y))
        if abs(u) > half or abs(v) > half:
            continue
        if t < best_t:
            best_t = t
            best_i = i
    return best_i
