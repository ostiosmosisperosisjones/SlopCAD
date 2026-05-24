"""
viewer/renderer.py

OpenGL draw calls split into two phases:

    draw_opaque(...)   — solid geometry + optional wireframe edges
    draw_overlays(...) — selection/hover highlights + sketch overlays

All colors read from cad.prefs so they can be changed without touching
this file.
"""

import ctypes
from OpenGL.GL import *
from cad.prefs import prefs


def draw_opaque(meshes, workspace, selection):
    """
    Phase 1: solid geometry, optional wireframe, selected face highlights.
    Depth buffer is fully populated after this returns.
    """
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)

    for body_id, mesh in meshes.items():
        body = workspace.bodies.get(body_id)
        if body and not body.visible:
            continue
        c = prefs.body_color_active if body_id == workspace.active_body_id \
            else prefs.body_color
        glColor3f(*c)
        mesh.draw()

    glDisable(GL_LIGHTING)

    # Selected face highlights — depth-tested so they don't bleed through
    # occluding geometry. Polygon offset pulls the highlight slightly toward
    # the camera to avoid z-fighting with the underlying face.
    if selection.face_count > 0:
        glColor3f(*prefs.face_selected_color)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)
        glEnableClientState(GL_VERTEX_ARRAY)
        for face_sel in selection.faces:
            mesh = meshes.get(face_sel.body_id)
            if mesh is None:
                continue
            start, count = mesh.get_face_triangle_range(face_sel.face_idx)
            glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo_verts)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo)
            glDrawElements(GL_TRIANGLES, count * 3, GL_UNSIGNED_INT,
                           ctypes.c_void_p(start * 3 * 4))
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glDisable(GL_POLYGON_OFFSET_FILL)

    # Wireframe edges — optional, respects prefs
    if prefs.show_edges:
        glColor3f(*prefs.edge_color)
        glLineWidth(prefs.edge_width)
        for body_id, mesh in meshes.items():
            body = workspace.bodies.get(body_id)
            if body and not body.visible:
                continue
            mesh.draw_edges()
        glLineWidth(1.0)

    glEnable(GL_LIGHTING)


def draw_overlays(meshes, selection, hovered_vertex, hovered_edge,
                  sketch=None, camera_distance: float = 1.0,
                  history=None, editing_sketch_idx=None, in_sketch: bool = False):
    """
    Phase 2: hover/selection overlays + sketch overlays.

    Draws committed sketches from history first (persistent, dimmed),
    then the active sketch session on top (if any).

    Parameters
    ----------
    history : History | None
        If provided, all visible committed sketch entries are rendered.
    """
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POINT_SMOOTH)

    # Selected mesh edges
    if selection.edge_count > 0:
        glLineWidth(3.0)
        glColor3f(*prefs.edge_selected_color)
        for es in selection.edges:
            mesh = meshes.get(es.body_id)
            if mesh is None or es.edge_idx >= len(mesh.topo_edges):
                continue
            _draw_polyline(mesh.topo_edges[es.edge_idx])
        glLineWidth(1.0)


    # Hovered mesh edge
    hov_ebody, hov_eidx = hovered_edge
    if hov_ebody is not None:
        from viewer.hover import parse_sketch_key
        if parse_sketch_key(hov_ebody) is None:
            mesh = meshes.get(hov_ebody)
            if mesh is not None and hov_eidx < len(mesh.topo_edges):
                glLineWidth(4.0)
                glColor3f(*prefs.edge_hovered_color)
                _draw_polyline(mesh.topo_edges[hov_eidx])
                glLineWidth(1.0)

    # Selected vertices
    if selection.vertex_count > 0:
        glPointSize(10.0)
        glColor3f(*prefs.vertex_selected_color)
        glBegin(GL_POINTS)
        for vs in selection.vertices:
            mesh = meshes.get(vs.body_id)
            if mesh is None or vs.vertex_idx >= len(mesh.topo_verts):
                continue
            p = mesh.topo_verts[vs.vertex_idx]
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        glEnd()

    # Hovered vertex
    hov_body, hov_idx = hovered_vertex
    if hov_body is not None:
        mesh = meshes.get(hov_body)
        if mesh is not None and hov_idx < len(mesh.topo_verts):
            p = mesh.topo_verts[hov_idx]
            glPointSize(12.0)
            glColor3f(*prefs.vertex_hovered_color)
            glBegin(GL_POINTS)
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()

    glPointSize(1.0)
    glDisable(GL_POINT_SMOOTH)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Sketch overlays — drawn last, on top of everything
    # ------------------------------------------------------------------
    from viewer.sketch_overlay import SketchOverlay
    overlay = SketchOverlay()
    all_labels = []

    # Committed sketches from history (persistent, dimmed)
    if history is not None:
        for i, entry in enumerate(history.entries):
            if entry.operation != "sketch":
                continue
            if i == editing_sketch_idx:
                continue  # live sketch session replaces this entry
            se = entry.params.get("sketch_entry")
            if se is not None and se.visible:
                all_labels.extend(overlay.draw_committed(
                    se, camera_distance, hovered_edge=hovered_edge,
                    history_idx=i, selection=selection,
                    dim_dimensions=not in_sketch))

    # Active sketch session (grid + axes + live entities + cursor)
    if sketch is not None:
        all_labels.extend(overlay.draw(
            sketch, camera_distance, hovered_edge=hovered_edge,
            selection=selection))

    return all_labels


def draw_world_planes(world_plane_visible: dict, scene_radius: float = 100.0):
    """
    Draw semi-transparent world-plane quads for any axis marked visible.
    Call before draw_overlays (while depth test is still on).
    """
    if not any(world_plane_visible.values()):
        return

    r = scene_radius
    # (normal_axis, quad_corners, line_color)
    _PLANES = {
        "XY": ([(- r, -r, 0), ( r, -r, 0), ( r,  r, 0), (-r,  r, 0)],
               (0.29, 0.49, 0.71)),   # blue
        "XZ": ([(-r, 0, -r), ( r, 0, -r), ( r, 0,  r), (-r, 0,  r)],
               (0.29, 0.71, 0.49)),   # green
        "YZ": ([(0, -r, -r), (0,  r, -r), (0,  r,  r), (0, -r,  r)],
               (0.71, 0.29, 0.29)),   # red
    }

    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_CULL_FACE)
    glDepthMask(GL_FALSE)   # read depth (stay behind solids) but never write it

    for axis, (corners, color) in _PLANES.items():
        if not world_plane_visible.get(axis, False):
            continue

        # Filled quad — very faint
        glColor4f(color[0], color[1], color[2], 0.07)
        glBegin(GL_QUADS)
        for c in corners:
            glVertex3f(*c)
        glEnd()

        # Border / grid lines
        glColor4f(color[0], color[1], color[2], 0.35)
        glLineWidth(1.0)

        # Outline
        glBegin(GL_LINE_LOOP)
        for c in corners:
            glVertex3f(*c)
        glEnd()

        # Grid lines at 1/4 intervals
        steps = 4
        glBegin(GL_LINES)
        for i in range(1, steps):
            t = -r + (2 * r * i / steps)
            # Determine which two axes the plane spans
            if axis == "XY":
                glVertex3f(t, -r, 0); glVertex3f(t,  r, 0)
                glVertex3f(-r, t, 0); glVertex3f( r, t, 0)
            elif axis == "XZ":
                glVertex3f(t, 0, -r); glVertex3f(t, 0,  r)
                glVertex3f(-r, 0, t); glVertex3f( r, 0, t)
            elif axis == "YZ":
                glVertex3f(0, t, -r); glVertex3f(0, t,  r)
                glVertex3f(0, -r, t); glVertex3f(0,  r, t)
        glEnd()

    # Origin marker — three short axis lines + dot
    arm = scene_radius * 0.06
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor4f(0.85, 0.25, 0.25, 0.9); glVertex3f(0, 0, 0); glVertex3f(arm, 0, 0)  # X red
    glColor4f(0.30, 0.55, 0.90, 0.9); glVertex3f(0, 0, 0); glVertex3f(0, arm, 0)  # Y blue
    glColor4f(0.25, 0.75, 0.25, 0.9); glVertex3f(0, 0, 0); glVertex3f(0, 0, arm)  # Z green (up)
    glEnd()
    glLineWidth(1.0)
    glPointSize(6.0)
    glColor4f(1.0, 1.0, 1.0, 0.9)
    glBegin(GL_POINTS)
    glVertex3f(0, 0, 0)
    glEnd()
    glPointSize(1.0)

    glDepthMask(GL_TRUE)
    glEnable(GL_CULL_FACE)
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)


def _draw_polyline(pts):
    glBegin(GL_LINE_STRIP)
    for p in pts:
        glVertex3f(float(p[0]), float(p[1]), float(p[2]))
    glEnd()
