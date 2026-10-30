"""
viewer/curve_overlay.py

Draw committed helix curves (and the helix panel's preview) as 3D
polylines, and pick them in screen space — they have no mesh, so hover /
GPU picking never sees them.
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import (
    glDisable, glEnable, glColor4f, glLineWidth, glBegin, glEnd, glVertex3f,
    glBlendFunc, glDepthMask, glPointSize,
    GL_LIGHTING, GL_BLEND, GL_DEPTH_TEST, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_LINE_STRIP, GL_POINTS, GL_TRUE, GL_FALSE,
)

_LINE    = (0.88, 0.72, 0.29, 1.0)
_HIDDEN  = (0.88, 0.72, 0.29, 0.25)          # the part behind bodies
_PICKED  = (1.00, 0.95, 0.30, 1.0)
_PREVIEW = (0.95, 0.85, 0.15, 0.9)
_PICK_PX = 8.0


def helix_points(edge, turns: float) -> np.ndarray:
    """Points along a helix edge — 48 per turn."""
    n = int(min(4000, max(64, 48 * turns)))
    return np.array([tuple(edge.position_at(t)) for t in np.linspace(0.0, 1.0, n)])


def _signature(viewport, i, entry):
    op = entry.op
    if op.face_ref is not None:
        src = id(viewport.history._shape_for_body_at(op.body_id, i))
    else:
        sidx = viewport.history.id_to_index(op.circle_sketch_id)
        se = (viewport.history.entries[sidx].params.get("sketch_entry")
              if sidx is not None else None)
        ent = next((e for e in (se.entities if se else [])
                    if getattr(e, "uid", None) == op.circle_uid), None)
        src = (None if ent is None else
               (tuple(np.round(ent.center[:2], 9)), round(ent.radius, 9),
                tuple(np.round(se.plane_origin, 9)), tuple(np.round(se.plane_normal, 9))))
    return (repr(sorted(entry.params.items())), src)


def collect_helices(viewport) -> list:
    """[(entry_idx, (n, 3) points)] for every helix up to the cursor."""
    cache = viewport.__dict__.setdefault('_helix_pts_cache', {})
    out = []
    history = viewport.history
    entries = history.entries
    for i in range(min(history.cursor + 1, len(entries))):
        e = entries[i]
        if e.operation != "helix" or e.error or e.op is None or e.editing:
            continue
        try:
            sig = _signature(viewport, i, e)
            hit = cache.get(e.entry_id)
            if hit is None or hit[0] != sig:
                edge, _axis, turns = e.op.edge(history, i)
                hit = (sig, helix_points(edge, turns))
                cache[e.entry_id] = hit
        except Exception:
            continue
        out.append((i, hit[1]))
    return out


def _strip(pts):
    glBegin(GL_LINE_STRIP)
    for p in pts:
        glVertex3f(float(p[0]), float(p[1]), float(p[2]))
    glEnd()


def draw_helices(viewport):
    helices = collect_helices(viewport)
    preview = getattr(viewport, '_helix_preview_pts', None)
    if not helices and preview is None:
        return
    picked = getattr(viewport, '_helix_highlight_idx', None)
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # Faint everywhere (it lies on the face it wraps), solid where in front.
    glDisable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)
    glLineWidth(1.5)
    glColor4f(*_HIDDEN)
    for _i, pts in helices:
        _strip(pts)
    glEnable(GL_DEPTH_TEST)
    glLineWidth(2.5)
    for i, pts in helices:
        glColor4f(*(_PICKED if i == picked else _LINE))
        _strip(pts)
    glDisable(GL_DEPTH_TEST)
    if preview is not None:
        glColor4f(*_PREVIEW)
        _strip(preview)
    # The start: where a sweep profile goes.
    glPointSize(7.0)
    glBegin(GL_POINTS)
    for i, pts in helices + ([(None, preview)] if preview is not None else []):
        glColor4f(*(_PREVIEW if i is None else _PICKED if i == picked else _LINE))
        glVertex3f(*(float(v) for v in pts[0]))
    glEnd()
    glPointSize(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_TRUE)
    glLineWidth(1.0)


def pick_helix(viewport, mx: float, my: float) -> int | None:
    """History index of the helix drawn within a few pixels of widget
    position (mx, my), or None."""
    if viewport._modelview is None:
        return None
    M = np.asarray(viewport._modelview, float).reshape(4, 4)
    P = np.asarray(viewport._projection, float).reshape(4, 4)
    vx, vy, vw, vh = (float(v) for v in viewport._viewport)
    dpr = viewport.devicePixelRatio()
    px, py = mx * dpr, vh - my * dpr
    best, best_d = None, _PICK_PX * dpr
    for i, pts in collect_helices(viewport):
        clip = np.hstack([pts, np.ones((len(pts), 1))]) @ M @ P   # GL is column-major
        w = clip[:, 3]
        ok = w > 1e-9
        if not ok.any():
            continue
        ndc = clip[ok, :2] / w[ok, None]
        sx = vx + (ndc[:, 0] + 1) * 0.5 * vw
        sy = vy + (ndc[:, 1] + 1) * 0.5 * vh
        # Distance to each segment, not just the vertices.
        a = np.stack([sx[:-1], sy[:-1]], 1)
        b = np.stack([sx[1:], sy[1:]], 1)
        ab = b - a
        t = np.clip(((np.array([px, py]) - a) * ab).sum(1) /
                    np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
        d = np.linalg.norm(a + ab * t[:, None] - np.array([px, py]), axis=1)
        if len(d) and d.min() < best_d:
            best, best_d = i, float(d.min())
    return best
