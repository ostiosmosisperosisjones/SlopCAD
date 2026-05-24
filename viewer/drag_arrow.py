"""
viewer/drag_arrow.py

DragArrow — a 3-D arrow rendered on top of everything (depth test off)
that can be hit-tested against a ray for mouse dragging.

Pass the surface/anchor point as `origin`.  The disc sits at `origin` on
the surface; the shaft and cone extend outward along `direction`.

Usage:
    arrow = DragArrow()
    arrow.draw(origin, direction, scale)          # call from paintGL
    t = arrow.hit_test(ray_origin, ray_dir, origin, direction, scale)
    # t is a drag parameter along `direction` if hit, else None
"""

from __future__ import annotations
import math
import numpy as np


# Number of sides on the cone and shaft cylinder
_SIDES = 12

# Proportions (all in units of `scale`)
_SHAFT_RADIUS  = 0.045
_SHAFT_LENGTH  = 0.62
_CONE_RADIUS   = 0.11
_CONE_LENGTH   = 0.30
_BASE_DISC_RADIUS = 0.13   # flat disc at the base, sits on the surface


def _rotation_to(direction: np.ndarray):
    """Return a 3×3 rotation matrix that maps Z+ to `direction`."""
    d = direction / np.linalg.norm(direction)
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, d)
    cn = np.linalg.norm(cross)
    if cn < 1e-9:
        # parallel or anti-parallel
        if d[2] > 0:
            return np.eye(3)
        else:
            return np.diag([1.0, -1.0, -1.0])
    axis = cross / cn
    angle = math.acos(max(-1.0, min(1.0, float(np.dot(z, d)))))
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    ax, ay, az = axis
    return np.array([
        [t*ax*ax + c,    t*ax*ay - s*az, t*ax*az + s*ay],
        [t*ax*ay + s*az, t*ay*ay + c,    t*ay*az - s*ax],
        [t*ax*az - s*ay, t*ay*az + s*ax, t*az*az + c   ],
    ])


def _circle_pts(radius: float, n: int) -> list[tuple[float, float]]:
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append((radius * math.cos(a), radius * math.sin(a)))
    return pts


class DragArrow:
    """
    Stateless helper — all geometry is derived from (origin, direction, scale).
    The disc sits at `origin` on the surface; shaft and cone extend along
    `direction`. Thread-safe (no mutable state).
    """

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, origin: np.ndarray, direction: np.ndarray,
             scale: float, color: tuple[float, float, float] = (0.95, 0.85, 0.15),
             surface_clearance: float = 0.0):
        """
        Draw the arrow with its disc at `origin`, shaft and cone extending along `direction`.
        `surface_clearance` (in units of `scale`) shifts the whole arrow outward if needed.
        Renders with depth test OFF so it is always on top.
        """
        from OpenGL.GL import (
            glDisable, glEnable, glBegin, glEnd, glVertex3f, glColor4f,
            glColor3f, glNormal3f, glLineWidth,
            GL_TRIANGLES, GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, GL_LINES,
            GL_DEPTH_TEST, GL_LIGHTING, GL_BLEND, GL_CULL_FACE,
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, glBlendFunc,
        )

        d_unit = np.asarray(direction, dtype=float)
        d_unit = d_unit / np.linalg.norm(d_unit)

        # lz=0 (disc, shaft base) sits at origin + clearance offset; cone tip further out.
        o = np.asarray(origin, dtype=float) + d_unit * surface_clearance * scale

        R = _rotation_to(direction)

        def xf(lx, ly, lz):
            p = o + R @ np.array([lx, ly, lz]) * scale
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        r, g, b = color
        rim_r, rim_g, rim_b = min(r + 0.15, 1.0), min(g + 0.15, 1.0), min(b + 0.15, 1.0)
        dark_r, dark_g, dark_b = r * 0.55, g * 0.55, b * 0.55

        circle = _circle_pts(1.0, _SIDES)

        # ── Shaft ──────────────────────────────────────────────────────
        sr = _SHAFT_RADIUS
        sl = _SHAFT_LENGTH
        glColor4f(r, g, b, 0.92)
        glBegin(GL_TRIANGLE_STRIP)
        for cx, cy in circle + [circle[0]]:
            xf(cx * sr, cy * sr, 0.0)
            xf(cx * sr, cy * sr, sl)
        glEnd()

        # Shaft caps
        glColor4f(dark_r, dark_g, dark_b, 0.80)
        glBegin(GL_TRIANGLE_FAN)
        xf(0, 0, 0)
        for cx, cy in circle + [circle[0]]:
            xf(cx * sr, cy * sr, 0.0)
        glEnd()
        glColor4f(r, g, b, 0.92)
        glBegin(GL_TRIANGLE_FAN)
        xf(0, 0, sl)
        for cx, cy in reversed(circle + [circle[0]]):
            xf(cx * sr, cy * sr, sl)
        glEnd()

        # ── Cone ───────────────────────────────────────────────────────
        cr = _CONE_RADIUS
        cl = _CONE_LENGTH
        tip_z = sl + cl

        glColor4f(rim_r, rim_g, rim_b, 1.0)
        glBegin(GL_TRIANGLE_FAN)
        xf(0, 0, tip_z)
        for cx, cy in circle + [circle[0]]:
            xf(cx * cr, cy * cr, sl)
        glEnd()

        # Cone base disc
        glColor4f(dark_r, dark_g, dark_b, 0.85)
        glBegin(GL_TRIANGLE_FAN)
        xf(0, 0, sl)
        for cx, cy in reversed(circle + [circle[0]]):
            xf(cx * cr, cy * cr, sl)
        glEnd()

        # ── Base disc (sits on the surface at lz=0) ────────────────────
        td = _BASE_DISC_RADIUS
        glColor4f(r, g, b, 0.55)
        glBegin(GL_TRIANGLE_FAN)
        xf(0, 0, 0)
        for cx, cy in circle + [circle[0]]:
            xf(cx * td, cy * td, 0.0)
        glEnd()

        # Outline ring for clarity
        glColor4f(0.0, 0.0, 0.0, 0.55)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        prev = circle[-1]
        for cx, cy in circle:
            xf(prev[0] * td, prev[1] * td, 0.0)
            xf(cx * td, cy * td, 0.0)
            prev = (cx, cy)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glDisable(GL_BLEND)

    # ------------------------------------------------------------------
    # Hit test
    # ------------------------------------------------------------------

    def hit_test(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                 origin: np.ndarray, direction: np.ndarray,
                 scale: float, surface_clearance: float = 0.0) -> float | None:
        """
        Test whether `ray` hits the arrow.
        `origin`, `direction`, `scale`, `surface_clearance` must match draw().
        Returns signed scalar `t` along `direction` if hit, else None.
        """
        o  = np.asarray(ray_origin,  dtype=float)
        d  = np.asarray(ray_dir,     dtype=float)
        ax = np.asarray(direction,   dtype=float)
        ax = ax / np.linalg.norm(ax)

        # shaft base = origin + clearance * scale * ax  (same as draw())
        a = np.asarray(origin, dtype=float) + ax * surface_clearance * scale

        # Hit radius = cone radius in world units (generous for usability)
        hit_r = _CONE_RADIUS * scale * 1.5
        total_len = (_SHAFT_LENGTH + _CONE_LENGTH) * scale

        # Relative ray origin in local frame
        rel = o - a
        d_ax   = float(np.dot(d,   ax))
        rel_ax = float(np.dot(rel, ax))

        d_perp   = d   - d_ax   * ax
        rel_perp = rel - rel_ax * ax

        # Closest approach of the ray to the arrow axis (cylinder test)
        a_coef = float(np.dot(d_perp,   d_perp))
        b_coef = 2.0 * float(np.dot(d_perp, rel_perp))
        c_coef = float(np.dot(rel_perp,  rel_perp)) - hit_r ** 2

        if a_coef < 1e-14:
            if c_coef > 0:
                return None
            ray_t = -rel_ax / d_ax if abs(d_ax) > 1e-10 else 0.0
        else:
            disc = b_coef * b_coef - 4.0 * a_coef * c_coef
            if disc < 0:
                return None
            ray_t = (-b_coef - math.sqrt(disc)) / (2.0 * a_coef)

        if ray_t < 0:
            if a_coef > 1e-14:
                disc = b_coef * b_coef - 4.0 * a_coef * c_coef
                ray_t = (-b_coef + math.sqrt(max(0.0, disc))) / (2.0 * a_coef)
            if ray_t < 0:
                return None

        hit_world = o + d * ray_t
        local_z = float(np.dot(hit_world - a, ax))

        if local_z < -hit_r or local_z > total_len + hit_r:
            return None

        return local_z / scale
