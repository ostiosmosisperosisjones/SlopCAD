"""
cad/sketch_tools/transform.py

Pure 2-D transform core shared by Mirror and Pattern tools.

A transform is expressed as an affine point map applied to an entity.  Two
flavours matter for sketch geometry:

  - orientation-preserving (translate, rotate): arc winding is unchanged, so
    we just map center + endpoints and keep the stored CCW sweep.
  - orientation-reversing (reflect): winding flips — handled in mirror.py,
    which predates this module and keeps its own reflect_entity.

`transform_entity(ent, fn, preserves_orientation=True)` maps a LineEntity or
ArcEntity through point-map `fn`.  Construction flag is carried over.
"""

from __future__ import annotations
import math
import numpy as np


def translate_fn(delta: np.ndarray):
    """Point map: p → p + delta."""
    delta = np.asarray(delta, dtype=np.float64)
    return lambda p: np.asarray(p, dtype=np.float64) + delta


def rotate_fn(center: np.ndarray, angle: float):
    """Point map: rotate p about center by angle (radians, CCW)."""
    center = np.asarray(center, dtype=np.float64)
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    def _fn(p):
        d = np.asarray(p, dtype=np.float64) - center
        return center + R @ d
    return _fn


def transform_entity(ent, fn, preserves_orientation: bool = True):
    """
    Map a LineEntity / ArcEntity through point-map `fn`.

    fn must be a rigid (isometric) map for arcs — translate or rotate — so the
    radius is preserved.  For orientation-preserving maps the CCW sweep is kept;
    callers needing reflection use mirror.reflect_entity instead.

    Returns a new entity, or None for unsupported types.
    """
    from cad.sketch import LineEntity, ArcEntity

    if isinstance(ent, LineEntity):
        return LineEntity(fn(ent.p0), fn(ent.p1),
                          construction=ent.construction)

    if isinstance(ent, ArcEntity):
        rc = fn(ent.center)
        full = abs((ent.end_angle - ent.start_angle) - 2 * math.pi) < 1e-9
        if full:
            return ArcEntity(rc, ent.radius, 0.0, 2 * math.pi,
                             construction=ent.construction)

        rp0 = fn(ent.p0)
        rp1 = fn(ent.p1)
        if preserves_orientation:
            # CCW p0→p1 stays CCW.
            a_from = math.atan2(float(rp0[1] - rc[1]), float(rp0[0] - rc[0]))
            a_to   = math.atan2(float(rp1[1] - rc[1]), float(rp1[0] - rc[0]))
        else:
            # Orientation flips: the traversal that was CCW becomes CW, so the
            # transformed arc sweeps CCW from rp1 back to rp0.
            a_from = math.atan2(float(rp1[1] - rc[1]), float(rp1[0] - rc[0]))
            a_to   = math.atan2(float(rp0[1] - rc[1]), float(rp0[0] - rc[0]))
        span = (a_to - a_from) % (2 * math.pi)
        if span < 1e-12:
            span = 2 * math.pi
        return ArcEntity(rc, ent.radius, a_from, a_from + span,
                         construction=ent.construction)

    return None
